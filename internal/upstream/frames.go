// frames.go 把「上游 SSE 帧处理」从 StreamHint 中抽出：协议无关的读取/解析/
// 归一化/标记修复在 PumpFrames 内完成，逐帧交付给 FrameSink；目标协议的编码与
// 写出由 sink 负责（OpenAI SSE 由 openAISSEWriter 实现，见 sse.go）。
package upstream

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ErrSinkWrite 标记 PumpFrames 返回的错误源自 FrameSink 写出失败（客户端断连/写
// 错误），以区别于上游读取错误。runTurn 据此区分：errors.Is(err, ErrSinkWrite) →
// 客户端已走（账号健康，记成功 + 粘性绑定）；否则是上游流中断（账号失败，不记成功、
// 不绑粘性）。此前两者共用同一 error 值，上游中途重置会被误判成客户端断连。
var ErrSinkWrite = errors.New("upstream: frame sink write failed")

// FrameSink 接收 PumpFrames 交付的协议无关帧/事件，负责目标协议的编码与写出。
type FrameSink interface {
	// Chunk 交付一帧已归一化的 chunk（已跑 stripToolCallNames/firstID/normalizeFrame）。
	// rawFallback 为编码失败时的原样兜底文本（与旧 writeChunk 的 marshal 失败路径一致）。
	Chunk(frame map[string]any, rawFallback string) error
	// Raw 交付需原样透传的非 JSON data 帧 payload（不加 hint）。
	Raw(payload string) error
	// ErrorFrame 交付上游 error 帧 payload（透传；OpenAI sink 会附加 gateway_hint）。
	ErrorFrame(payload string) error
	// Comment 交付非 data、非空的原始行（含换行，原样透传）。
	Comment(line string) error
	// Done 交付流结束事件（sink 写终止标记）。
	Done() error
}

// PumpFrames 读取上游 SSE，逐帧解析/归一化/修复后交付给 sink，返回终止错误。
// 空流（0 有效帧）时交付一帧空流 error 并返回 errEmptyStream。
func PumpFrames(r io.Reader, sink FrameSink, opts ...StreamOption) error {
	var o streamOptions
	for _, f := range opts {
		if f != nil {
			f(&o)
		}
	}
	return pumpFrames(r, sink, o)
}

// pumpFrames 是协议无关的帧处理主循环：拥有 toolCallSeen/firstID/maxToolIdx/
// synthIdx/toolFinishSent/repair 等全部跨帧状态，把结果经 sink 交付。
func pumpFrames(r io.Reader, sink FrameSink, o streamOptions) error {
	// toolCallSeen 跨帧记录 delta.tool_calls 里已发过首片的 index，
	// 供逐 chunk 透传时收敛 name 为「每 index 一次」（对齐 OpenAI 官方流）。
	toolCallSeen := map[int]bool{}

	// firstID 透传流的消息级 id 基准：缓存首个非空上游 id，后续帧缺失/空串时复用
	// （issue #35：同一条 SSE 消息所有帧共用一个真实 id，后台按 id 归并；此前中间帧
	// 一律补 chatcmpl-wb2api 哨兵，造成同流 id 分裂）。全流无真实 id → 才出现哨兵。
	firstID := ""

	// 标记修复（见 dsml.go）：把上游正文里的原生工具调用标记还原成 delta.tool_calls。
	// 客户端没声明过 tools 时 repair.Enabled() 为 false → 整条流零改写零分配。
	//   - synthIdx：本流已还原的调用数（合成的 index 从上游最大 index 之后接续）；
	//   - maxToolIdx：上游已用过的最大 tool_call index，避免合成的 index 与真实调用撞车；
	//   - toolFinishSent：是否已发出过 finish_reason: tool_calls，供流尾兜底。
	repair := o.markup
	synthIdx := 0
	maxToolIdx := -1
	toolFinishSent := false

	// 复读检测（见 loopguard.go）：**每条流一个实例**，只在配置开启时创建。
	// 未开启时为 nil，Observe 是 nil 安全的方法（不会进入任何判定分支）。
	var guard *LoopGuard
	if o.loopGuard {
		guard = &LoopGuard{}
	}

	// writeChunk 把一个 chunk 对象按规范白名单重建后交付给 sink，同时维护三件跨帧
	// 状态：tool_calls name 收敛（每 index 仅首片带 name）、消息 id 续传、以及上游
	// 已用过的最大 tool_call index（供标记修复接续编号）。
	// fallback 是规范化失败时的兜底原文（与修复前「marshal 失败即原样透传」一致）。
	writeChunk := func(obj map[string]any, fallback string) error {
		// 复读检测先于写出：命中即中止本流，不让后续重复内容继续流向客户端、
		// 也不让上游继续被读到（越早停越省钱）。
		if guard != nil {
			if txt := frameVisibleText(obj); txt != "" && guard.Observe(txt) {
				return ErrLoopDetected
			}
		}
		// 先按 index 收敛 tool_calls name（每 index 仅首片保留，后续分片删 name 键），再规范化透传。
		stripToolCallNames(obj, toolCallSeen)
		// id 续传：首帧非空真实 id 缓存；后续帧缺 id / 空 id 一律用缓存值，
		// 有自己 id 的帧保持原样（不同流分裂的帧允许各自 id）。
		if firstID == "" {
			if v, ok := obj["id"].(string); ok && v != "" {
				firstID = v
			}
		} else if v, ok := obj["id"].(string); !ok || v == "" {
			obj["id"] = firstID
		}
		if chs, ok := obj["choices"].([]any); ok {
			for _, ci := range chs {
				c, _ := ci.(map[string]any)
				if c == nil {
					continue
				}
				d, _ := c["delta"].(map[string]any)
				if d == nil {
					continue
				}
				tcs, _ := d["tool_calls"].([]any)
				for _, tci := range tcs {
					tc, _ := tci.(map[string]any)
					if tc == nil {
						continue
					}
					if v, ok := tc["index"].(float64); ok && int(v) > maxToolIdx {
						maxToolIdx = int(v)
					}
				}
			}
		}
		if err := sink.Chunk(normalizeFrame(obj), fallback); err != nil {
			return fmt.Errorf("%w: %w", ErrSinkWrite, err)
		}
		return nil
	}

	// repairFrame 在透传前跑标记修复，并把一帧展开成「可能多帧」：正文帧（标记被摘掉
	// 后的可见文本）+ N 个 delta.tool_calls 帧 + 收尾 finish_reason 帧。没有命中标记
	// 时原样返回入参帧（单元素），与修复前逐字节一致。
	repairFrame := func(obj map[string]any) []map[string]any {
		if !repair.Enabled() {
			return []map[string]any{obj}
		}
		chs, ok := obj["choices"].([]any)
		if !ok {
			return []map[string]any{obj}
		}
		var calls []MarkupCall
		for _, ci := range chs {
			c, ok := ci.(map[string]any)
			if !ok {
				continue
			}
			d, ok := c["delta"].(map[string]any)
			if !ok {
				continue
			}
			txt, ok := d["content"].(string)
			if !ok || txt == "" {
				continue
			}
			emit, got := repair.Feed(txt)
			calls = append(calls, got...)
			if emit == "" {
				delete(d, "content")
			} else {
				d["content"] = emit
			}
		}
		if len(calls) == 0 {
			// 本帧没命中标记；但本流此前还原过调用 → 把上游的 stop 收尾改写成
			// tool_calls，客户端据此知道「回合以工具调用结束」而不是「模型只说了话」。
			if synthIdx > 0 && rewriteFinishReason(obj, "tool_calls") {
				toolFinishSent = true
			}
			return []map[string]any{obj}
		}
		// 命中：finish_reason 必须挪到工具调用帧之后，否则客户端在读到调用之前就
		// 认为回合已经结束。注意「本帧本来没有 finish_reason」时**不得**补一帧收尾
		// ——上游的收尾帧随后就到，补了会变成两个 finish_reason（后续那个由
		// rewriteFinishReason 改写）。
		fr := takeFinishReason(obj)
		out := []map[string]any{obj}
		for _, call := range calls {
			idx := maxToolIdx + 1 + synthIdx
			synthIdx++
			out = append(out, markupCallFrame(obj, idx, call))
		}
		if fr != "" {
			if fr == "stop" {
				fr = "tool_calls"
			}
			out = append(out, finishOnlyFrame(obj, fr))
			toolFinishSent = fr == "tool_calls"
		}
		return out
	}

	// writeFrame 把 payload 按规范白名单重建后交付给 sink。
	// 仅 JSON 解析成功时计数记为一次有效转发（JSON 解析失败照常降级原样写出，但不计数）。
	writeFrame := func(payload string) (int, error) {
		var obj map[string]any
		if json.Unmarshal([]byte(payload), &obj) != nil {
			// 非 JSON 帧：原样降级写出，不计数（与修复前一致）。
			if werr := sink.Raw(payload); werr != nil {
				return 0, fmt.Errorf("%w: %w", ErrSinkWrite, werr)
			}
			return 0, nil
		}
		// 上游错误帧透传（error-passthrough）：带 error 键的帧**原样写出**，不走
		// normalizeFrame 白名单——白名单会剥掉 error 字段，客户端就看不到上游
		// code/msg/requestId。error.message 即上游原文（如 6004 限流、审核拦截），
		// 计入有效帧（避免误判空流补写 "empty upstream stream"）。
		if _, hasErr := obj["error"]; hasErr {
			// 旁路通知（账号处置用）：原始 payload 交给观察者，透传字节不变。
			if o.onErrorFrame != nil {
				o.onErrorFrame(payload)
			}
			if werr := sink.ErrorFrame(payload); werr != nil {
				return 0, fmt.Errorf("%w: %w", ErrSinkWrite, werr)
			}
			return 1, nil
		}
		for _, ch := range repairFrame(obj) {
			if werr := writeChunk(ch, payload); werr != nil {
				return 0, werr
			}
		}
		return 1, nil
	}

	br := bufio.NewReaderSize(r, 64*1024)
	validFrames := 0
readLoop:
	for {
		line, err := br.ReadString('\n')
		trimmed := strings.TrimRight(line, "\r\n")
		switch {
		case strings.HasPrefix(trimmed, "data: [DONE]"):
			// 上游显式结束：停止读取，DONE 之后的任何数据（含垃圾帧）一律不再透传。
			// [DONE] 统一在循环结束后写出，保证恰好一个。
			break readLoop
		case strings.HasPrefix(trimmed, "data: "):
			n, werr := writeFrame(strings.TrimPrefix(trimmed, "data: "))
			validFrames += n
			if werr != nil {
				return werr
			}
		case trimmed != "":
			// 注释/其他行：原样透传
			if werr := sink.Comment(line); werr != nil {
				return fmt.Errorf("%w: %w", ErrSinkWrite, werr)
			}
		}
		// 空行（帧分隔）吞掉：本函数自产 "\n\n"
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
	}
	// 标记修复的尾部收尾：流结束时仍未判定的字节一律原文回吐（绝不吞字节，未闭合
	// 的块连起始标记一起交还）；若本流还原过调用、但上游从未给出收尾帧，补一帧
	// finish_reason: tool_calls，保证客户端不会把「工具调用回合」读成「模型只说了话」。
	if repair.Enabled() {
		if tail := repair.Flush(); tail != "" {
			tailObj := map[string]any{
				"object": "chat.completion.chunk",
				"choices": []any{map[string]any{
					"index": 0,
					"delta": map[string]any{"content": tail},
				}},
			}
			if werr := writeChunk(tailObj, ""); werr != nil {
				return werr
			}
			validFrames++
		}
		if synthIdx > 0 && !toolFinishSent {
			if werr := writeChunk(finishOnlyFrame(map[string]any{}, "tool_calls"), ""); werr != nil {
				return werr
			}
		}
	}
	// 空流（0 有效帧）：先交付一帧 error（sink 原样保留 error 字段），
	// 再补 [DONE] 保证客户端能正常收尾，并返回非 nil error 供调用方记录。
	// 该兜底帧与上游 error 帧同路径交付：OpenAI sink 会按 hintFn 附加 gateway_hint，
	// 与旧 writeRaw 行为逐字节一致。
	if validFrames == 0 {
		_ = sink.ErrorFrame(`{"error":{"message":"empty upstream stream","type":"upstream_error","code":"upstream_parse"}}`)
	}
	// 保证恰好写一个 [DONE]（上游漏发时兜底补上）。
	if err := sink.Done(); err != nil {
		return fmt.Errorf("%w: %w", ErrSinkWrite, err)
	}
	if validFrames == 0 {
		return errEmptyStream
	}
	return nil
}
