package server

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// messages_stream_fixes_test.go 覆盖流式/请求归一化的 7 个已修复缺陷。
//
// 这些都是「现有测试覆盖盲区」里的真实 bug：上游交错分片、部分 usage 帧、
// error 帧后仍吐帧、Anthropic SDK 合法但网关未识别的工具类型等。

// 1. 并行 tool_call 交错分片：两个 index 的 arguments 必须各自完整，不丢分片。
func TestMessagesStreamParallelToolCallsInterleaved(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"write_stdin","arguments":"{\"session\":"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls -la\"}"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"s1\"}"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	events := parseSSEEvents(t, rec.Body.String())

	// 每个 tool_use 块：start(id/name) → delta* → stop，按 index 顺序且参数完整。
	type blk struct {
		id, name string
		args     string
		stopped  bool
	}
	blocks := map[float64]*blk{}
	for _, e := range events {
		switch e.name {
		case "content_block_start":
			cb, _ := e.data["content_block"].(map[string]any)
			if cb["type"] != "tool_use" {
				continue
			}
			idx := e.data["index"].(float64)
			blocks[idx] = &blk{id: cb["id"].(string), name: cb["name"].(string)}
			if blocks[idx].id == "" || blocks[idx].name == "" {
				t.Errorf("block %.0f opened with empty id/name: %v", idx, cb)
			}
		case "content_block_delta":
			d := e.data["delta"].(map[string]any)
			if d["type"] != "input_json_delta" {
				continue
			}
			idx := e.data["index"].(float64)
			if b := blocks[idx]; b != nil {
				b.args += d["partial_json"].(string)
			}
		case "content_block_stop":
			idx := e.data["index"].(float64)
			if b := blocks[idx]; b != nil {
				b.stopped = true
			}
		}
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks=%d want 2: %v", len(blocks), blocks)
	}
	if blocks[0].args != `{"cmd":"ls -la"}` {
		t.Errorf("block0 args=%q want %q", blocks[0].args, `{"cmd":"ls -la"}`)
	}
	if blocks[1].args != `{"session":"s1"}` {
		t.Errorf("block1 args=%q want %q", blocks[1].args, `{"session":"s1"}`)
	}
	for idx, b := range blocks {
		if !b.stopped {
			t.Errorf("block %.0f never stopped", idx)
		}
	}
	// 每个块的 JSON 必须可解析（截断会失败）。
	for idx, b := range blocks {
		var v map[string]any
		if err := json.Unmarshal([]byte(b.args), &v); err != nil {
			t.Errorf("block %.0f args not valid JSON: %q (%v)", idx, b.args, err)
		}
	}
}

// 2. 正文块已打开后到达的 reasoning_content：不得发出 index:-1 的 thinking_delta。
func TestMessagesStreamThinkingAfterTextNoNegativeIndex(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"reasoning_content":"late think"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	for _, e := range parseSSEEvents(t, rec.Body.String()) {
		if e.name != "content_block_delta" {
			continue
		}
		idx := e.data["index"].(float64)
		if idx < 0 {
			t.Fatalf("content_block_delta with negative index %.0f: %v", idx, e.data)
		}
		if d := e.data["delta"].(map[string]any); d["type"] == "thinking_delta" {
			t.Fatalf("thinking_delta emitted after text block opened: %v", e.data)
		}
	}
}

// 3. error 帧之后不得再发任何 content 事件。
func TestMessagesStreamNoContentAfterError(t *testing.T) {
	const sse = `data: {"error":{"message":"boom","type":"upstream_error"}}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"leaked"}}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	sawError := false
	for _, e := range parseSSEEvents(t, rec.Body.String()) {
		if e.name == "error" {
			sawError = true
			continue
		}
		if sawError {
			t.Fatalf("event %q emitted after error", e.name)
		}
	}
	if !sawError {
		t.Fatal("no error event seen")
	}
}

// 4. usage 分帧到达（末帧只带 completion_tokens）：message_delta 不得丢掉 input_tokens。
func TestMessagesStreamUsagePartialFrames(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":100,"cache_read_input_tokens":60}}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":7}}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	md := eventsNamed(parseSSEEvents(t, rec.Body.String()), "message_delta")[0].data
	usage := md["usage"].(map[string]any)
	if got := usage["input_tokens"].(float64); got != 40 {
		t.Fatalf("input_tokens=%v want 40 (100-60, from earlier frame)", got)
	}
	if got := usage["cache_read_input_tokens"].(float64); got != 60 {
		t.Fatalf("cache_read_input_tokens=%v want 60 (from earlier frame)", got)
	}
	if got := usage["output_tokens"].(float64); got != 7 {
		t.Fatalf("output_tokens=%v want 7", got)
	}
}

// 5. 客户端工具的合法 type:"custom" 必须保留为 function 工具，不得丢弃。
func TestMessagesCustomTypedToolKept(t *testing.T) {
	raw := []byte(`{"model":"m","max_tokens":100,"tools":[{"type":"custom","name":"apply_patch","description":"edit","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`)
	openAI, _, merr := normalizeMessagesRequest(raw)
	if merr != nil {
		t.Fatalf("normalize error: %v", merr.message)
	}
	var body map[string]any
	if err := json.Unmarshal(openAI, &body); err != nil {
		t.Fatal(err)
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools=%v want 1 (custom type must be kept)", body["tools"])
	}
	fn, _ := tools[0].(map[string]any)["function"].(map[string]any)
	if fn == nil || fn["name"] != "apply_patch" {
		t.Fatalf("tool not mapped to function: %v", tools[0])
	}
}

// 6. 检索失败 → error_code 必须是 Anthropic 合法枚举（unavailable）。
func TestMessagesWebSearchErrorCodeValid(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up}) // web_search 未启用 → ws.err 非空

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"m","max_tokens":100,"tools":[{"type":"web_search_20260209","name":"web_search"}],"messages":[{"role":"user","content":"hello"}]}`)))

	resp := mustDecodeJSON(t, rec.Body.Bytes())
	content, _ := resp["content"].([]any)
	if len(content) < 2 {
		t.Fatalf("content=%v", resp["content"])
	}
	c, _ := content[1].(map[string]any)["content"].(map[string]any)
	if c["type"] != "web_search_tool_result_error" {
		t.Fatalf("expected error block, got %v", content[1])
	}
	if code, _ := c["error_code"].(string); code != "unavailable" {
		t.Fatalf("error_code=%q want unavailable", code)
	}
}

// 8. thinking → tool → text 交错：块顺序必须严格闭合（不得在 thinking 未闭合时插 tool）。
func TestMessagesStreamBlockOrderStrictlyClosed(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"content":"after tool"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	// 状态机：任何时刻最多一个块处于「已 start 未 stop」状态；stop 的 index 必须
	// 等于当前打开的 index。用栈校验严格闭合。
	open := map[float64]bool{}
	for _, e := range parseSSEEvents(t, rec.Body.String()) {
		switch e.name {
		case "content_block_start":
			if len(open) != 0 {
				t.Fatalf("content_block_start while another block open: %v", e.data)
			}
			open[e.data["index"].(float64)] = true
		case "content_block_delta":
			idx := e.data["index"].(float64)
			if !open[idx] {
				t.Fatalf("delta for non-open block %.0f: %v", idx, e.data)
			}
		case "content_block_stop":
			idx := e.data["index"].(float64)
			if !open[idx] {
				t.Fatalf("stop for non-open block %.0f", idx)
			}
			delete(open, idx)
		}
	}
	if len(open) != 0 {
		t.Fatalf("blocks left open at stream end: %v", open)
	}
}

// 9. lastUserText 必须按 rune 截断，不切断 UTF-8 多字节字符。
func TestLastUserTextUTF8SafeTruncation(t *testing.T) {
	long := strings.Repeat("汉", 300) // 900 字节 / 300 rune
	got := lastUserText([]any{map[string]any{"role": "user", "content": long}})
	if !utf8.ValidString(got) {
		t.Fatalf("truncated string is invalid UTF-8: len=%d", len(got))
	}
	if n := utf8.RuneCountInString(got); n != 300 {
		t.Fatalf("rune count=%d want 300 (no truncation needed)", n)
	}
	// 超长输入按 rune 截到 512。
	huge := strings.Repeat("汉", 1000)
	got2 := lastUserText([]any{map[string]any{"role": "user", "content": huge}})
	if !utf8.ValidString(got2) {
		t.Fatal("truncated huge string invalid UTF-8")
	}
	if n := utf8.RuneCountInString(got2); n != 512 {
		t.Fatalf("rune count=%d want 512", n)
	}
}
