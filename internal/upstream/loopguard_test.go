package upstream

import (
	"errors"
	"strings"
	"testing"
)

// 复读检测的判定口径是「尾部窗口完全周期性」，因此测试要同时锁住两件事：
// 真复读必须命中，正常长文本（含反复讨论同一话题）必须不命中。

func feed(g *LoopGuard, s string, chunk int) bool {
	for i := 0; i < len(s); i += chunk {
		end := i + chunk
		if end > len(s) {
			end = len(s)
		}
		if g.Observe(s[i:end]) {
			return true
		}
	}
	return false
}

func TestDetectsVerbatimRepeat(t *testing.T) {
	cases := []struct {
		name string
		unit string
	}{
		{"短句复读", "Hmm. "},
		{"行动短句复读", "Let me check the file again. "},
		{"长段复读", strings.Repeat("I need to verify the configuration before proceeding. ", 2)},
		{"中文复读", "让我再确认一下这个配置是否正确。"},
		{"标记复读", "<|FunctionCallBegin|>run<|FunctionCallEnd|>"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := &LoopGuard{}
			// 先给一段**非周期性**的正常前缀（凑够 loopMinTotal），再无限复读。
			// 前缀必须逐字节变化：直接 Repeat 固定串本身就是复读，会自己命中。
			var pb strings.Builder
			for i := 0; i < 60; i++ {
				pb.WriteString("第")
				pb.WriteString(strings.Repeat("正", i%5+1))
				pb.WriteString("段正常说明，用于把累计长度推过判定门槛。\n")
			}
			if feed(g, pb.String(), 64) {
				t.Fatal("正常前缀不应命中")
			}
			repeated := strings.Repeat(tc.unit, 200)
			if !feed(g, repeated, 64) {
				t.Fatalf("复读单元 %q 应被检出", tc.unit)
			}
			if !g.Tripped() {
				t.Fatal("命中后 Tripped 应为真")
			}
			if g.Period() <= 0 {
				t.Fatal("命中后应记录重复周期")
			}
		})
	}
}

// TestIgnoresNormalLongText 是误报闸门：正常长文本（含反复使用同样的句式与词）
// 不得命中——误报会腰斩用户的正常回复。
func TestIgnoresNormalLongText(t *testing.T) {
	g := &LoopGuard{}
	var b strings.Builder
	// 构造"反复讨论同一话题、句式相似但不逐字节重复"的文本。
	for i := 0; i < 400; i++ {
		b.WriteString("第 ")
		b.WriteString(strings.Repeat("x", i%7+1))
		b.WriteString(" 项检查通过，接下来确认下一个环节是否符合预期。\n")
	}
	if feed(g, b.String(), 128) {
		t.Fatal("正常长文本被误判为复读")
	}
	// 结构化输出（重复的键名、缩进）同样不应命中。
	g2 := &LoopGuard{}
	var j strings.Builder
	for i := 0; i < 300; i++ {
		j.WriteString(`{"path": "/src/file`)
		j.WriteString(strings.Repeat("y", i%5+1))
		j.WriteString(`.go", "line": 12}` + "\n")
	}
	if feed(g2, j.String(), 128) {
		t.Fatal("结构化输出被误判为复读")
	}
	// 装饰性分隔线：同一字符重复在语义上是 period=1，不是复读（靠"周期内至少两个
	// 不同字节"挡住）。这是短周期下限从 8 降到 4 之后必须守住的边界。
	g3 := &LoopGuard{}
	var f3 strings.Builder
	for i := 0; i < 60; i++ {
		f3.WriteString("第")
		f3.WriteString(strings.Repeat("正", i%5+1))
		f3.WriteString("段正常说明。\n")
	}
	f3.WriteString(strings.Repeat("-", 2000))
	if feed(g3, f3.String(), 128) {
		t.Fatal("装饰性分隔线被误判为复读")
	}
}

func TestShortOutputNeverTrips(t *testing.T) {
	// 累计不足 loopMinTotal：即便完全周期性也不判定（短回复不可能构成复读）。
	g := &LoopGuard{}
	if feed(g, strings.Repeat("ab", 200), 16) { // 400 字节 < 1024
		t.Fatal("短输出不应判定复读")
	}
}

func TestTrippedIsSticky(t *testing.T) {
	g := &LoopGuard{}
	feed(g, strings.Repeat("正常内容。", 200), 64)
	feed(g, strings.Repeat("复读复读复读。", 200), 64)
	if !g.Tripped() {
		t.Fatal("应已命中")
	}
	// 命中后即便喂正常文本也保持 true（调用方据此终止流，不再翻转）。
	if !g.Observe("正常文本") {
		t.Fatal("命中后 Observe 应恒为 true（粘滞）")
	}
}

func TestNilGuardSafe(t *testing.T) {
	var g *LoopGuard
	if g.Observe("anything") {
		t.Fatal("nil guard 不应命中")
	}
	if g.Tripped() {
		t.Fatal("nil guard Tripped 应为假")
	}
}

func TestFrameVisibleText(t *testing.T) {
	// 正文与推理都算可见文本；非 chunk 形态返回空串。
	frame := map[string]any{"choices": []any{
		map[string]any{"delta": map[string]any{"content": "abc", "reasoning_content": "xyz"}},
	}}
	if got := frameVisibleText(frame); got != "abcxyz" {
		t.Fatalf("frameVisibleText=%q want abcxyz", got)
	}
	if got := frameVisibleText(map[string]any{"choices": []any{
		map[string]any{"delta": map[string]any{"role": "assistant"}},
	}}); got != "" {
		t.Fatalf("无文本帧应返回空串，got %q", got)
	}
	if got := frameVisibleText(map[string]any{}); got != "" {
		t.Fatalf("非 chunk 应返回空串，got %q", got)
	}
}

// TestPumpFramesAbortsOnLoop 端到端锁定：开启 WithLoopGuard 后，复读流会被中止
// 并返回 ErrLoopDetected；关闭时同一输入正常读完（零行为变化）。
func TestPumpFramesAbortsOnLoop(t *testing.T) {
	build := func() string {
		var b strings.Builder
		b.WriteString(`data: {"choices":[{"delta":{"content":"`)
		b.WriteString(strings.Repeat("正常开场内容。", 100))
		b.WriteString(`"}}]}` + "\n\n")
		// 之后是复读帧：每帧一段重复文本，累计超过窗口即命中。
		for i := 0; i < 40; i++ {
			b.WriteString(`data: {"choices":[{"delta":{"content":"`)
			b.WriteString(strings.Repeat("复读复读复读复读。", 8))
			b.WriteString(`"}}]}` + "\n\n")
		}
		b.WriteString("data: [DONE]\n\n")
		return b.String()
	}

	// 开启：应中止并返回 ErrLoopDetected。
	sink := &countingSink{}
	err := PumpFrames(strings.NewReader(build()), sink, WithLoopGuard())
	if !errors.Is(err, ErrLoopDetected) {
		t.Fatalf("应返回 ErrLoopDetected，got %v", err)
	}
	aborted := sink.frames

	// 关闭：同一输入应正常读完（帧数更多，且无错误）。
	sink2 := &countingSink{}
	if err := PumpFrames(strings.NewReader(build()), sink2); err != nil {
		t.Fatalf("未开启复读检测时不应报错: %v", err)
	}
	if sink2.frames <= aborted {
		t.Fatalf("未开启时应读完更多帧：opened=%d closed=%d", aborted, sink2.frames)
	}
}

// countingSink 只数帧数，不做任何编码。
type countingSink struct{ frames int }

func (s *countingSink) Chunk(map[string]any, string) error { s.frames++; return nil }
func (s *countingSink) Raw(string) error                   { s.frames++; return nil }
func (s *countingSink) ErrorFrame(string) error            { s.frames++; return nil }
func (s *countingSink) Comment(string) error               { return nil }
func (s *countingSink) Done() error                        { s.frames++; return nil }
