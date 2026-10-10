// pump_test.go 直接单测 PumpFrames：用一个记录型 FrameSink 捕获交付顺序与内容，
// 覆盖帧处理（读取/分类/解码/错误帧/标记修复/归一化）与 sink 交付契约。
package upstream

import (
	"encoding/json"
	"errors"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

// sinkCall 记录一次 sink 交付。
type sinkCall struct {
	kind     string // chunk / raw / error / comment / done
	frame    map[string]any
	fallback string
	payload  string
	line     string
}

// recordingSink 记录有序交付；failOn 命中的方法返回 failErr（用于错误路径测试）。
type recordingSink struct {
	calls   []sinkCall
	failOn  string
	failErr error
}

var errSinkFail = errors.New("sink failure")

func (s *recordingSink) record(c sinkCall) error {
	s.calls = append(s.calls, c)
	if s.failOn != "" && c.kind == s.failOn {
		return s.failErr
	}
	return nil
}

func (s *recordingSink) Chunk(frame map[string]any, rawFallback string) error {
	return s.record(sinkCall{kind: "chunk", frame: frame, fallback: rawFallback})
}

func (s *recordingSink) Raw(payload string) error {
	return s.record(sinkCall{kind: "raw", payload: payload})
}

func (s *recordingSink) ErrorFrame(payload string) error {
	return s.record(sinkCall{kind: "error", payload: payload})
}

func (s *recordingSink) Comment(line string) error {
	return s.record(sinkCall{kind: "comment", line: line})
}

func (s *recordingSink) Done() error {
	return s.record(sinkCall{kind: "done"})
}

// kinds 返回交付序列的 kind 列表。
func (s *recordingSink) kinds() []string {
	out := make([]string, 0, len(s.calls))
	for _, c := range s.calls {
		out = append(out, c.kind)
	}
	return out
}

// chunkContents 返回所有 chunk 帧的 delta.content（无 content 记空串）。
func chunkContents(s *recordingSink) []string {
	var out []string
	for _, c := range s.calls {
		if c.kind != "chunk" {
			continue
		}
		out = append(out, frameContent(c.frame))
	}
	return out
}

// frameContent 取 choices[0].delta.content（缺失返回空串）。
func frameContent(frame map[string]any) string {
	chs, _ := frame["choices"].([]any)
	if len(chs) == 0 {
		return ""
	}
	ch, _ := chs[0].(map[string]any)
	d, _ := ch["delta"].(map[string]any)
	v, _ := d["content"].(string)
	return v
}

// frameFinish 取 choices[0].finish_reason（缺失返回空串）。
func frameFinish(frame map[string]any) string {
	chs, _ := frame["choices"].([]any)
	if len(chs) == 0 {
		return ""
	}
	ch, _ := chs[0].(map[string]any)
	v, _ := ch["finish_reason"].(string)
	return v
}

// dataLine 拼一条 data 帧行（含空行分隔）。
func dataLine(payload string) string { return "data: " + payload + "\n\n" }

// 1. 多帧按源顺序交付。
func TestPumpFramesDeliversInOrder(t *testing.T) {
	raw := dataLine(`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}`) +
		dataLine(`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"a"}}]}`) +
		dataLine(`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"b"}}]}`) +
		dataLine("[DONE]")

	sink := &recordingSink{}
	if err := PumpFrames(strings.NewReader(raw), sink); err != nil {
		t.Fatalf("PumpFrames: %v", err)
	}
	if got := strings.Join(chunkContents(sink), ","); got != ",a,b" {
		t.Fatalf("chunk 顺序/内容 = %q，期望 \",a,b\"", got)
	}
	if k := sink.kinds(); strings.Join(k, ",") != "chunk,chunk,chunk,done" {
		t.Fatalf("交付序列 = %v", k)
	}
}

// 2. [DONE] 之后不再交付任何帧，Done 恰好一次。
func TestPumpFramesDoneStopsStream(t *testing.T) {
	raw := dataLine(`{"id":"x","choices":[{"index":0,"delta":{"content":"a"}}]}`) +
		dataLine("[DONE]") +
		dataLine(`{"id":"x","choices":[{"index":0,"delta":{"content":"ignored"}}]}`)

	sink := &recordingSink{}
	if err := PumpFrames(strings.NewReader(raw), sink); err != nil {
		t.Fatalf("PumpFrames: %v", err)
	}
	if got := chunkContents(sink); len(got) != 1 || got[0] != "a" {
		t.Fatalf("DONE 之后仍交付了帧: %v", got)
	}
	if n := countKind(sink, "done"); n != 1 {
		t.Fatalf("Done 调用 %d 次，期望 1", n)
	}
}

// 3. 无 [DONE] 的 EOF：仍交付 Done，且不误报空流。
func TestPumpFramesEOFWithoutDone(t *testing.T) {
	raw := dataLine(`{"id":"x","choices":[{"index":0,"delta":{"content":"a"}}]}`)

	sink := &recordingSink{}
	err := PumpFrames(strings.NewReader(raw), sink)
	if err != nil {
		t.Fatalf("无 [DONE] 的 EOF 不应报错: %v", err)
	}
	if n := countKind(sink, "done"); n != 1 {
		t.Fatalf("Done 调用 %d 次，期望 1", n)
	}
	if n := countKind(sink, "error"); n != 0 {
		t.Fatalf("非空流不应交付空流 error 帧，实际 %d 次", n)
	}
}

// 4. 空流：交付空流 error 帧 + Done，返回 errEmptyStream。
func TestPumpFramesEmptyStream(t *testing.T) {
	sink := &recordingSink{}
	err := PumpFrames(strings.NewReader(""), sink)
	if !IsEmptyStreamError(err) {
		t.Fatalf("期望 errEmptyStream，实际 %v", err)
	}
	if n := countKind(sink, "error"); n != 1 {
		t.Fatalf("空流 error 帧 %d 次，期望 1", n)
	}
	if n := countKind(sink, "done"); n != 1 {
		t.Fatalf("Done %d 次，期望 1", n)
	}
	if got := sink.calls[0].payload; !strings.Contains(got, "empty upstream stream") {
		t.Fatalf("空流 error payload = %q", got)
	}
	if k := sink.kinds(); strings.Join(k, ",") != "error,done" {
		t.Fatalf("空流交付序列 = %v", k)
	}
}

// 5. 跨帧 DSML 修复：sink 收到合成的 tool_call 帧与 finish_reason:tool_calls 帧。
func TestPumpFramesMarkupRepairCrossFrame(t *testing.T) {
	block := mkBlock(mkInvoke("exec_command", mkParam("cmd", "ls -la")))
	content := "准备执行。\n" + block
	runes := []rune(content)
	mid := len(runes) / 2
	chunks := []string{string(runes[:mid]), string(runes[mid:])}
	raw := upstreamSSEWithContent(chunks, "stop")

	repair := NewMarkupRepair(testTools, false)
	sink := &recordingSink{}
	if err := PumpFrames(strings.NewReader(raw), sink, WithMarkupRepair(repair)); err != nil {
		t.Fatalf("PumpFrames: %v", err)
	}

	var sawCall, sawFinish bool
	for _, c := range sink.calls {
		if c.kind != "chunk" {
			continue
		}
		if s := frameContent(c.frame); strings.Contains(s, markupPipe) {
			t.Fatalf("透出正文仍残留标记: %q", s)
		}
		if name := frameToolCallName(c.frame); name != "" {
			if name != "exec_command" {
				t.Fatalf("工具名 = %q，期望 exec_command", name)
			}
			sawCall = true
		}
		if frameFinish(c.frame) == "tool_calls" {
			sawFinish = true
		}
	}
	if !sawCall {
		t.Fatal("未交付合成的 tool_call 帧")
	}
	if !sawFinish {
		t.Fatal("未交付 finish_reason:tool_calls 帧")
	}
}

// 6. 未闭合 DSML 尾部：流尾把原始尾部文本作为 content 帧回吐。
func TestPumpFramesMarkupTailFlush(t *testing.T) {
	content := "正文" + markupOpen + "\n" + mkInvoke("exec_command", mkParam("cmd", "ls"))
	raw := upstreamSSEWithContent([]string{content}, "stop")

	repair := NewMarkupRepair(testTools, false)
	sink := &recordingSink{}
	if err := PumpFrames(strings.NewReader(raw), sink, WithMarkupRepair(repair)); err != nil {
		t.Fatalf("PumpFrames: %v", err)
	}
	var tail string
	for _, c := range sink.calls {
		if c.kind == "chunk" && strings.Contains(frameContent(c.frame), markupOpen) {
			tail = frameContent(c.frame)
		}
	}
	if tail == "" {
		t.Fatal("未交付未闭合标记的尾部 content 帧")
	}
	if !strings.Contains(tail, markupOpen) || !strings.Contains(tail, "exec_command") {
		t.Fatalf("尾部 content 不完整: %q", tail)
	}
}

// 7. sink.Chunk 返回错误 → PumpFrames 立即返回该错误并停止。
func TestPumpFramesChunkErrorStops(t *testing.T) {
	raw := dataLine(`{"id":"x","choices":[{"index":0,"delta":{"content":"a"}}]}`) +
		dataLine(`{"id":"x","choices":[{"index":0,"delta":{"content":"b"}}]}`) +
		dataLine("[DONE]")

	sink := &recordingSink{failOn: "chunk", failErr: errSinkFail}
	err := PumpFrames(strings.NewReader(raw), sink)
	if !errors.Is(err, errSinkFail) {
		t.Fatalf("期望 errSinkFail，实际 %v", err)
	}
	if len(sink.calls) != 1 {
		t.Fatalf("出错后仍继续交付: %v", sink.kinds())
	}
}

// errReader 恒定返回非 EOF 错误。
type errReader struct{ err error }

func (r errReader) Read([]byte) (int, error) { return 0, r.err }

// 8. 上游读取错误原样返回。
func TestPumpFramesReaderError(t *testing.T) {
	boom := errors.New("boom")
	sink := &recordingSink{}
	err := PumpFrames(errReader{err: boom}, sink)
	if !errors.Is(err, boom) {
		t.Fatalf("期望 boom，实际 %v", err)
	}
}

// 9. error 帧：观察者恰好一次（原始 payload），sink.ErrorFrame 恰好一次（同 payload）。
func TestPumpFramesErrorFrameObserver(t *testing.T) {
	payload := `{"error":{"message":"rate limited","code":"6004"}}`
	raw := dataLine(payload) + dataLine("[DONE]")

	var observed []string
	sink := &recordingSink{}
	err := PumpFrames(strings.NewReader(raw), sink, WithErrorFrameObserver(func(p string) {
		observed = append(observed, p)
	}))
	if err != nil {
		t.Fatalf("PumpFrames: %v", err)
	}
	if len(observed) != 1 || observed[0] != payload {
		t.Fatalf("观察者调用 = %v，期望 [%s]", observed, payload)
	}
	if n := countKind(sink, "error"); n != 1 {
		t.Fatalf("ErrorFrame %d 次，期望 1", n)
	}
	if got := firstPayload(sink, "error"); got != payload {
		t.Fatalf("ErrorFrame payload = %q，期望 %q", got, payload)
	}
}

// 10. 注释/非 data 行 → sink.Comment 收到原始行（含换行）。
func TestPumpFramesCommentPassthrough(t *testing.T) {
	raw := ": keepalive\n\n" +
		dataLine(`{"id":"x","choices":[{"index":0,"delta":{"content":"a"}}]}`) +
		dataLine("[DONE]")

	sink := &recordingSink{}
	if err := PumpFrames(strings.NewReader(raw), sink); err != nil {
		t.Fatalf("PumpFrames: %v", err)
	}
	if n := countKind(sink, "comment"); n != 1 {
		t.Fatalf("Comment %d 次，期望 1", n)
	}
	for _, c := range sink.calls {
		if c.kind == "comment" {
			if c.line != ": keepalive\n" {
				t.Fatalf("Comment 行 = %q，期望 %q", c.line, ": keepalive\n")
			}
		}
	}
}

// 11. 归一化在交付前完成：finish_reason:"" → nil，顶层未知字段被剔除。
func TestPumpFramesNormalizesBeforeDelivery(t *testing.T) {
	raw := dataLine(`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":""}],"unknown_field":"z"}`) +
		dataLine("[DONE]")

	sink := &recordingSink{}
	if err := PumpFrames(strings.NewReader(raw), sink); err != nil {
		t.Fatalf("PumpFrames: %v", err)
	}
	var got map[string]any
	for _, c := range sink.calls {
		if c.kind == "chunk" {
			got = c.frame
		}
	}
	if got == nil {
		t.Fatal("未交付 chunk")
	}
	if _, ok := got["unknown_field"]; ok {
		t.Fatalf("未知字段未被剔除: %v", got)
	}
	chs, _ := got["choices"].([]any)
	ch, _ := chs[0].(map[string]any)
	if v, ok := ch["finish_reason"]; !ok || v != nil {
		t.Fatalf("finish_reason 应为 nil，实际 %#v", ch["finish_reason"])
	}
	if !strings.Contains(chunkJSON(t, got), `"content":"hi"`) {
		t.Fatalf("content 丢失: %v", got)
	}
}

// 12. 旧入口 StreamHint 仍与 PumpFrames+openAISSEWriter / Stream 输出逐字节一致。
func TestStreamHintDelegatesByteIdentical(t *testing.T) {
	raw := dataLine(`{"id":"up-1","object":"chat.completion.chunk","created":111,"model":"m1","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"}}]}`) +
		": keepalive\n\n" +
		dataLine(`{"id":"up-1","choices":[{"index":0,"delta":{"content":"，世界"}}]}`) +
		dataLine(`{"id":"up-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`) +
		dataLine("[DONE]")

	// StreamHint（无 opts）
	recHint := httptest.NewRecorder()
	if err := StreamHint(recHint, strings.NewReader(raw), nil); err != nil {
		t.Fatalf("StreamHint: %v", err)
	}
	// Stream（= StreamHint(nil)）
	recStream := httptest.NewRecorder()
	if err := Stream(recStream, strings.NewReader(raw)); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	// PumpFrames + openAISSEWriter
	recPump := httptest.NewRecorder()
	if err := pumpFrames(strings.NewReader(raw), &openAISSEWriter{w: recPump}, streamOptions{}); err != nil {
		t.Fatalf("pumpFrames: %v", err)
	}

	hintBody, streamBody, pumpBody := recHint.Body.String(), recStream.Body.String(), recPump.Body.String()
	if hintBody != streamBody {
		t.Fatalf("StreamHint 与 Stream 输出不一致:\n%s\n---\n%s", hintBody, streamBody)
	}
	if hintBody != pumpBody {
		t.Fatalf("StreamHint 与 PumpFrames+openAISSEWriter 输出不一致:\n%s\n---\n%s", hintBody, pumpBody)
	}
	if !strings.Contains(hintBody, "data: [DONE]\n\n") {
		t.Fatalf("缺少 [DONE]: %q", hintBody)
	}
}

// --- 辅助 -------------------------------------------------------------------

func countKind(s *recordingSink, kind string) int {
	n := 0
	for _, c := range s.calls {
		if c.kind == kind {
			n++
		}
	}
	return n
}

func firstPayload(s *recordingSink, kind string) string {
	for _, c := range s.calls {
		if c.kind == kind {
			return c.payload
		}
	}
	return ""
}

// frameToolCallName 取 choices[0].delta.tool_calls[0].function.name。
func frameToolCallName(frame map[string]any) string {
	chs, _ := frame["choices"].([]any)
	if len(chs) == 0 {
		return ""
	}
	ch, _ := chs[0].(map[string]any)
	d, _ := ch["delta"].(map[string]any)
	tcs, _ := d["tool_calls"].([]any)
	if len(tcs) == 0 {
		return ""
	}
	tc, _ := tcs[0].(map[string]any)
	fn, _ := tc["function"].(map[string]any)
	name, _ := fn["name"].(string)
	return name
}

func chunkJSON(t *testing.T, frame map[string]any) string {
	t.Helper()
	b, err := json.Marshal(frame)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

var _ FrameSink = (*recordingSink)(nil)
var _ io.Reader = errReader{}
