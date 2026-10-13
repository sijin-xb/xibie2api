// loopguard.go 推理/正文复读（退化循环）检测。
//
// 为什么需要它：推理模型偶发退化时会陷入复读——同一句话或同一段"行动短句"被
// 无限重复，直到撞上上游输出上限或空闲超时。这不是报错：上游一路 200 正常吐帧，
// 网关侧的降级重试（content_blocked）与熔断都拦不住它，**积分按 token 实扣**。
// 一个号陷入循环就是一路静默烧钱，直到超时。
//
// 检测口径刻意做得极保守：只看**流尾窗口是否完全周期性**。
//
//	尾部 W=768 字节若能被某个 8..128 字节的单元整除且逐字节相同 → 判定复读。
//
// 也就是说要触发，必须出现「一段至少 6 次逐字节相同的重复」且它占满整个尾部窗口。
// 正常长文本（哪怕反复讨论同一话题、反复列同样的清单）不会逐字节重复 6 次，
// 因此误报概率极低；而真复读（同一句/同一段原样刷屏）会瞬间满足条件。
//
// 不做的事：不判"语义重复"、不做模糊匹配、不惩罚账号（复读是模型行为，不是账号
// 故障——与 11101/内容拦截同一哲学）。命中即中止该次上游读取，由调用方决定
// 如何向客户端交代。
package upstream

import (
	"errors"
)

// ErrLoopDetected 上游输出被判定为复读循环。属于**不罚号、不轮转**的请求级失败：
// 同一 body 换个号大概率复现同样退化，轮转只会把烧钱乘以号数。
var ErrLoopDetected = errors.New("upstream output loop detected")

const (
	// loopWindow 尾部检测窗口字节数。窗口内必须完全周期性才判定命中。
	loopWindow = 768
	// loopMinUnit / loopMinRepeats 周期长度下限与最少重复次数。
	// 周期上限由 loopWindow/loopMinRepeats 推出（768/6 = 128）：保证窗口内至少
	// 出现 6 次完整重复，"两次相同"这种偶然不算。
	//
	// 下限取 4 而不是 8：经典推理复读是反复刷 "Hmm." / "Wait." 这类**极短**单元
	// （5~6 字节），下限 8 会整类漏检。短周期带来的误报风险（装饰性分隔线之类）
	// 由 periodicTail 的"周期内至少两个不同字节"检查挡住——全是同一个字符的
	// 重复（"======"、"──────"）在语义上是 period=1，不是复读。
	loopMinUnit    = 4
	loopMinRepeats = 6
	// loopMinTotal 累计文本不足此长度不做判定：短回复不可能构成"复读"。
	loopMinTotal = 1024
	// loopCheckStep 每新增这么多字节才检测一次，控制 CPU（检测本身很便宜，
	// 但没必要每帧都跑）。
	loopCheckStep = 256
)

// LoopGuard 单条流的复读检测器。**每条流一个实例**（pumpFrames 内部创建），
// 不跨流复用——否则上一条流的累计文本会污染下一条的判定。
type LoopGuard struct {
	buf     []byte // 累计可见文本（只保留尾部窗口所需长度）
	total   int    // 累计字节数（含已丢弃的前缀）
	checked int    // 上次检测时的 total
	tripped bool
	period  int // 命中时的最小重复周期（字节），仅诊断用
}

// Observe 喂入一段新到的可见文本，返回是否已判定为复读。
//
// 命中后恒返回 true（粘滞）：调用方据此中止读取，后续帧不会再进入判定。
func (g *LoopGuard) Observe(txt string) bool {
	if g == nil || txt == "" {
		return false
	}
	if g.tripped {
		return true
	}
	g.total += len(txt)
	g.buf = append(g.buf, txt...)
	// 只留尾部窗口：检测只看流尾，更早的内容没有判定价值。
	if len(g.buf) > loopWindow {
		drop := len(g.buf) - loopWindow
		g.buf = append(g.buf[:0], g.buf[drop:]...)
	}
	if g.total < loopMinTotal || g.total-g.checked < loopCheckStep {
		return false
	}
	g.checked = g.total
	if len(g.buf) < loopWindow {
		// 窗口还没填满：周期性必须占满整个窗口才算数，否则"两次重复"就会命中。
		return false
	}
	if p, ok := periodicTail(g.buf); ok {
		g.tripped = true
		g.period = p
		return true
	}
	return false
}

// Tripped 报告是否已判定复读（供调用方在流尾统一判定，不必依赖 Observe 的返回值）。
func (g *LoopGuard) Tripped() bool { return g != nil && g.tripped }

// Period 返回判定命中时的最小重复周期（字节）；未命中为 0。仅供日志诊断。
func (g *LoopGuard) Period() int { return g.period }

// frameVisibleText 从一帧 OpenAI 形态的 chunk 里取出「可见文本」：正文 content 与
// 推理 reasoning_content 都算——推理模型的复读通常先发生在推理里，只盯正文会漏。
// 非 chunk 形态 / 无文本 → 返回空串（调用方跳过，不参与判定）。
func frameVisibleText(obj map[string]any) string {
	chs, ok := obj["choices"].([]any)
	if !ok {
		return ""
	}
	var out string
	for _, ci := range chs {
		c, ok := ci.(map[string]any)
		if !ok {
			continue
		}
		d, ok := c["delta"].(map[string]any)
		if !ok {
			continue
		}
		for _, key := range []string{"content", "reasoning_content"} {
			if s, ok := d[key].(string); ok && s != "" {
				out += s
			}
		}
	}
	return out
}

// periodicTail 判断 buf 的尾部窗口是否完全周期性，返回其最小周期（字节）。
//
// 求**真实最小周期**而不是拿一组候选周期去猜：复读单元的字节长度是任意的
// （一句中文短句 27 字节、一段英文 63 字节都常见），候选集合只要不巧包含真实
// 周期就会整条漏检——早期版本用几何级数候选，测试里 27 字节的重复就直接漏了。
//
// 成本：对每个候选周期 p 逐字节比对，遇到第一个不匹配立即返回。非周期窗口在
// 第一个字节就失败，因此平均成本 ≈ 候选数（约 121 次比较）；只有真周期窗口才
// 会走满 O(loopWindow)。每 loopCheckStep 字节才检测一次，整体开销可忽略。
func periodicTail(buf []byte) (int, bool) {
	if len(buf) < loopWindow {
		return 0, false
	}
	tail := buf[len(buf)-loopWindow:]
	maxUnit := loopWindow / loopMinRepeats
	for p := loopMinUnit; p <= maxUnit; p++ {
		if isPeriod(tail, p) && distinctBytes(tail[:p]) >= 2 {
			return p, true
		}
	}
	return 0, false
}

// isPeriod 报告 s 是否以 p 为周期（s[i] == s[i-p] 对所有 i>=p 成立）。
func isPeriod(s []byte, p int) bool {
	for i := p; i < len(s); i++ {
		if s[i] != s[i-p] {
			return false
		}
	}
	return true
}

// distinctBytes 统计 s 中不同字节值的个数（上限 3 即可判定，不必精确）。
func distinctBytes(s []byte) int {
	var seen [256]bool
	n := 0
	for _, b := range s {
		if !seen[b] {
			seen[b] = true
			n++
			if n >= 3 {
				return n
			}
		}
	}
	return n
}
