// prompt_trim.go 服务端上下文超限自救：上游 11115「prompt is too long」时，把
// 归一化后 OpenAI body 的**中段历史**（最老的 assistant 工具调用 / tool 输出）裁掉
// 一次并重试，而不是把 400 透传回客户端。
//
// 动机（实测 2026-10-24）：Codex 长会话单回合可达数百次工具调用，历史随步数线性
// 膨胀；Codex 按自己估算的上下文窗口决定何时压缩，估算与上游真实上限（1048576）
// 之间存在漂移，中途撞线后**每个后续请求都原样重发同一份超长历史**——不裁剪的
// 话整个线程永久 400（网关侧还能看到确定性错误在账号池里空转）。
//
// 裁剪策略（对齐 Codex 自身压缩的形状：保头保尾、丢中间、留占位说明）：
//   - 头部连续 system/developer 块（客户端基础指令）逐字保留；
//   - 头部块后第一条非 system 消息（环境上下文 / 首条用户消息）保留；
//   - 尾部按字节预算保留最新历史（工具调用与结果的新鲜度对模型续跑最重要）；
//   - 中段整体删除，原位插入一条 user 角色占位说明（Codex 本地压缩摘要也是
//     user 消息形态；多 system 消息对上游是未实测行为，不引入）。
package server

import (
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strconv"
)

// promptTooLongRE 匹配上游 11115 报文里的 token 计数：
// "prompt is too long: 1103110 tokens > 1048576 maximum"。
var promptTooLongRE = regexp.MustCompile(`prompt is too long:\s*(\d+)\s*tokens?\s*>\s*(\d+)\s*maximum`)

// trimPromptOverflow 解析上游 11115 报文并裁剪 body 中段。
// 返回 (trimmed, true) = 裁剪成功可重试；(nil, false) = 无法裁剪（报文无计数 /
// 历史太短 / 可裁中段为空），调用方走原 400 透传。
func trimPromptOverflow(body []byte, upstreamBody string) ([]byte, bool) {
	var used, limit int64
	if m := promptTooLongRE.FindStringSubmatch(upstreamBody); m != nil {
		used, _ = strconv.ParseInt(m[1], 10, 64)
		limit, _ = strconv.ParseInt(m[2], 10, 64)
	}
	return trimPromptBody(body, used, limit)
}

// trimPromptBody 裁剪主体。used/limit 为上游实测值（0 = 解析失败，按保守比例裁）。
func trimPromptBody(body []byte, used, limit int64) ([]byte, bool) {
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, false
	}
	msgs, ok := obj["messages"].([]any)
	if !ok {
		return nil, false
	}
	// 头部连续 system/developer 块（developer 在 payload 层才归一为 system，
	// 此处 body 可能仍带 developer）。
	head := 0
	for head < len(msgs) {
		role := messageRole(msgs[head])
		if role != "system" && role != "developer" {
			break
		}
		head++
	}
	// 可裁区 = [head+1, len-1]：首条非 system 消息与最后一条恒保留。
	// 历史太短时裁剪无意义（省不出空间还破坏语境）。
	tailStart := len(msgs) - 1
	if tailStart-(head+1) < 8 {
		return nil, false
	}
	// 字节预算：目标保留量 = 0.7 * limit / used * 总量（留 30% 余量防估算漂移）；
	// 无计数时按保留 35% 裁。钳在 [5%, 90%]。
	keepRatio := 0.35
	if used > 0 && limit > 0 {
		keepRatio = 0.7 * float64(limit) / float64(used)
	}
	if keepRatio < 0.05 {
		keepRatio = 0.05
	}
	if keepRatio > 0.9 {
		keepRatio = 0.9
	}
	total := 0
	sizes := make([]int, len(msgs))
	for i, m := range msgs {
		b, err := json.Marshal(m)
		if err != nil {
			return nil, false
		}
		sizes[i] = len(b)
		total += sizes[i]
	}
	budget := int(keepRatio * float64(total))
	// 从尾向前累计，直到预算用尽；cutStart = 保留段的起点（不得侵入 head+1）。
	acc, cutStart := 0, tailStart
	for i := len(msgs) - 1; i > head+1 && acc < budget; i-- {
		acc += sizes[i]
		cutStart = i
	}
	if cutStart <= head+1 {
		return nil, false // 尾部保留段已顶到头部：无可裁中段
	}
	elided := cutStart - (head + 1)
	out := make([]any, 0, len(msgs)-elided+1)
	out = append(out, msgs[:head+1]...)
	out = append(out, map[string]any{
		"role": "user",
		"content": fmt.Sprintf(
			"[context notice] %d older conversation messages (mostly tool calls and their outputs) were elided by the gateway to fit the model context window (%d > %d tokens). The remaining recent history is complete and self-contained; do not attempt to recover the elided content.",
			elided, used, limit),
	})
	out = append(out, msgs[cutStart:]...)
	obj["messages"] = out
	trimmed, err := json.Marshal(obj)
	if err != nil {
		return nil, false
	}
	log.Printf("INFO: [server] prompt overflow trim: elided %d of %d messages (keep ratio %.2f, upstream %d > %d tokens)",
		elided, len(msgs), keepRatio, used, limit)
	return trimmed, true
}

// messageRole 安全取消息 role（非 map / 缺 role 返回空串）。
func messageRole(m any) string {
	msg, ok := m.(map[string]any)
	if !ok {
		return ""
	}
	role, _ := msg["role"].(string)
	return role
}
