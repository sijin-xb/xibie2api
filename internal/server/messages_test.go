// messages_test.go —— /v1/messages（Anthropic Messages API 子集）的行为测试。
//
// 覆盖：请求归一化（支持矩阵 + 拒绝项）、system 两种形态、tool_use/tool_result 多轮
// 往返、非流式转换（文本/工具/stop_reason/usage）、流式事件生命周期、上游 error 帧、
// 空流、sink 写失败、路由/鉴权/校验错误、模型别名、提示词模式。
// 复用 handler_test.go / prompt_modes_test.go / chat_regression_test.go 的夹具
// （newFakeUpstream、testPoolWith、sseOK、captureUpstream、outboundMessages、
// staticRegressionUpstream、regNonStreamSSE、regFragmentedToolSSE、parseSSEEvents）。
package server

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// msgFinishSSE 构造「一段正文 + 指定 finish_reason + usage」的上游 SSE。
func msgFinishSSE(finish string) string {
	fr := "null"
	if finish != "" {
		fr = `"` + finish + `"`
	}
	return "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":" + fr + "}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":5,\"total_tokens\":12}}\n\n" +
		"data: [DONE]\n\n"
}

// msgBadToolArgsSSE：tool_call 的 arguments 是非法 JSON（非流式转换必须 502）。
const msgBadToolArgsSSE = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_bad\",\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{not json\"}}]}}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":9,\"total_tokens\":12}}\n\n" +
	"data: [DONE]\n\n"

// msgTextThenToolSSE：先文本后工具调用（验证 index 单调 + 切换块先 stop）。
const msgTextThenToolSSE = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"let me run\"}}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"run\"}}]}}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\n\n" +
	"data: [DONE]\n\n"

// failingWriter 在首次 Write 时报错（sink 写失败传播测试）。
type failingWriter struct{}

func (failingWriter) Header() http.Header       { return http.Header{} }
func (failingWriter) Write([]byte) (int, error) { return 0, errors.New("write failed") }
func (failingWriter) WriteHeader(int)           {}

// ---------------------------------------------------------------------------
// 1. normalizeMessagesRequest
// ---------------------------------------------------------------------------

func TestNormalizeMessagesRequest(t *testing.T) {
	t.Run("text_string", func(t *testing.T) {
		body, meta, err := normalizeMessagesRequest([]byte(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if meta.model != "glm-5.2" || meta.stream {
			t.Fatalf("meta=%+v", meta)
		}
		obj := mustDecodeJSON(t, body)
		if obj["max_tokens"].(float64) != 100 {
			t.Fatalf("max_tokens=%v", obj["max_tokens"])
		}
		msgs := obj["messages"].([]any)
		if len(msgs) != 1 {
			t.Fatalf("messages=%v", msgs)
		}
		if m := msgs[0].(map[string]any); m["role"] != "user" || m["content"] != "hello" {
			t.Fatalf("message=%v", m)
		}
	})

	t.Run("content_block_array_joined", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		m := mustDecodeJSON(t, body)["messages"].([]any)[0].(map[string]any)
		if m["content"] != "ab" {
			t.Fatalf("content=%v want joined \"ab\"", m["content"])
		}
	})

	t.Run("system_string", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"system":"SYS","messages":[{"role":"user","content":"hi"}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		msgs := mustDecodeJSON(t, body)["messages"].([]any)
		if len(msgs) != 2 {
			t.Fatalf("messages=%v want 2", msgs)
		}
		if m := msgs[0].(map[string]any); m["role"] != "system" || m["content"] != "SYS" {
			t.Fatalf("msgs[0]=%v", m)
		}
	})

	t.Run("system_array_of_text_blocks", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"system":[{"type":"text","text":"A"},{"type":"text","text":"B"}],"messages":[{"role":"user","content":"hi"}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		m := mustDecodeJSON(t, body)["messages"].([]any)[0].(map[string]any)
		if m["role"] != "system" || m["content"] != "AB" {
			t.Fatalf("system=%v want AB", m)
		}
	})

	t.Run("tool_use_tool_result_roundtrip", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"messages":[` +
			`{"role":"user","content":"hi"},` +
			`{"role":"assistant","content":[{"type":"text","text":"calling"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"sunny"}]}]}` +
			`]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		msgs := mustDecodeJSON(t, body)["messages"].([]any)
		if len(msgs) != 3 {
			t.Fatalf("messages=%v want 3", msgs)
		}
		asst := msgs[1].(map[string]any)
		if asst["role"] != "assistant" || asst["content"] != "calling" {
			t.Fatalf("assistant=%v", asst)
		}
		tc := asst["tool_calls"].([]any)[0].(map[string]any)
		if tc["id"] != "toolu_1" || tc["type"] != "function" {
			t.Fatalf("tool_call=%v", tc)
		}
		if fn := tc["function"].(map[string]any); fn["name"] != "get_weather" || fn["arguments"] != `{"city":"SF"}` {
			t.Fatalf("function=%v", fn)
		}
		tool := msgs[2].(map[string]any)
		if tool["role"] != "tool" || tool["tool_call_id"] != "toolu_1" || tool["content"] != "sunny" {
			t.Fatalf("tool msg=%v", tool)
		}
	})

	t.Run("user_text_then_tool_result_order", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[` +
			`{"type":"text","text":"note"},` +
			`{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		msgs := mustDecodeJSON(t, body)["messages"].([]any)
		if len(msgs) != 2 {
			t.Fatalf("messages=%v want 2 (user text + tool)", msgs)
		}
		if m := msgs[0].(map[string]any); m["role"] != "user" || m["content"] != "note" {
			t.Fatalf("msgs[0]=%v", m)
		}
		if m := msgs[1].(map[string]any); m["role"] != "tool" || m["tool_call_id"] != "t1" {
			t.Fatalf("msgs[1]=%v", m)
		}
	})

	t.Run("image_base64", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		m := mustDecodeJSON(t, body)["messages"].([]any)[0].(map[string]any)
		parts := m["content"].([]any)
		p0 := parts[0].(map[string]any)
		if p0["type"] != "image_url" {
			t.Fatalf("part=%v", p0)
		}
		if url := p0["image_url"].(map[string]any)["url"]; url != "data:image/png;base64,AAAA" {
			t.Fatalf("url=%v", url)
		}
	})

	t.Run("image_url_source", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		p0 := mustDecodeJSON(t, body)["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
		if url := p0["image_url"].(map[string]any)["url"]; url != "https://x/y.png" {
			t.Fatalf("url=%v", url)
		}
	})

	t.Run("tools_and_stop", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"stop_sequences":["STOP"],"messages":[{"role":"user","content":"hi"}],` +
			`"tools":[{"name":"f","description":"d","input_schema":{"type":"object"}}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		obj := mustDecodeJSON(t, body)
		stop := obj["stop"].([]any)
		if len(stop) != 1 || stop[0] != "STOP" {
			t.Fatalf("stop=%v", stop)
		}
		tools := obj["tools"].([]any)
		fn := tools[0].(map[string]any)["function"].(map[string]any)
		if fn["name"] != "f" || fn["description"] != "d" {
			t.Fatalf("fn=%v", fn)
		}
		if _, ok := fn["parameters"]; !ok {
			t.Fatalf("parameters missing: %v", fn)
		}
	})

	t.Run("tool_choice_mappings", func(t *testing.T) {
		cases := []struct {
			in   string
			want any
		}{
			{`{"type":"auto"}`, "auto"},
			{`{"type":"any"}`, "required"},
			{`{"type":"none"}`, "none"},
		}
		for _, tc := range cases {
			raw := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tool_choice":` + tc.in + `}`
			body, _, err := normalizeMessagesRequest([]byte(raw))
			if err != nil {
				t.Fatalf("%s: %+v", tc.in, err)
			}
			if got := mustDecodeJSON(t, body)["tool_choice"]; got != tc.want {
				t.Fatalf("%s: tool_choice=%v want %v", tc.in, got, tc.want)
			}
		}
		raw := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"tool","name":"f"}}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("tool choice: %+v", err)
		}
		tc := mustDecodeJSON(t, body)["tool_choice"].(map[string]any)
		if tc["type"] != "function" || tc["function"].(map[string]any)["name"] != "f" {
			t.Fatalf("tool_choice=%v", tc)
		}
	})

	t.Run("metadata_ignored", func(t *testing.T) {
		raw := `{"model":"m","max_tokens":10,"metadata":{"user_id":"u1"},"messages":[{"role":"user","content":"hi"}]}`
		body, _, err := normalizeMessagesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("metadata must be accepted+ignored: %+v", err)
		}
		if strings.Contains(string(body), "user_id") {
			t.Fatalf("metadata must not be forwarded: %s", body)
		}
	})

	rejectCases := []struct {
		name  string
		raw   string
		field string
	}{
		{"missing_model", `{"max_tokens":10,"messages":[]}`, "model"},
		{"model_wrong_type", `{"model":1,"max_tokens":10,"messages":[]}`, "model"},
		{"missing_max_tokens", `{"model":"m","messages":[]}`, "max_tokens"},
		{"zero_max_tokens", `{"model":"m","max_tokens":0,"messages":[]}`, "max_tokens"},
		{"negative_max_tokens", `{"model":"m","max_tokens":-5,"messages":[]}`, "max_tokens"},
		{"fractional_max_tokens", `{"model":"m","max_tokens":1.5,"messages":[]}`, "max_tokens"},
		{"string_max_tokens", `{"model":"m","max_tokens":"10","messages":[]}`, "max_tokens"},
		{"missing_messages", `{"model":"m","max_tokens":10}`, "messages"},
		{"messages_wrong_type", `{"model":"m","max_tokens":10,"messages":"hi"}`, "messages"},
		{"stream_wrong_type", `{"model":"m","max_tokens":10,"stream":"yes","messages":[]}`, "stream"},
		{"unknown_content_block", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[{"type":"bogus"}]}]}`, "type"},
		{"unknown_role", `{"model":"m","max_tokens":10,"messages":[{"role":"system","content":"x"}]}`, "role"},
		{"unknown_tool_choice", `{"model":"m","max_tokens":10,"messages":[],"tool_choice":{"type":"weird"}}`, "tool_choice.type"},
		{"top_k", `{"model":"m","max_tokens":10,"messages":[],"top_k":5}`, "top_k"},
		{"thinking", `{"model":"m","max_tokens":10,"messages":[],"thinking":{"type":"enabled"}}`, "thinking"},
		{"output_config", `{"model":"m","max_tokens":10,"messages":[],"output_config":{}}`, "output_config"},
		{"container", `{"model":"m","max_tokens":10,"messages":[],"container":"c"}`, "container"},
		{"mcp_servers", `{"model":"m","max_tokens":10,"messages":[],"mcp_servers":[]}`, "mcp_servers"},
		{"context_management", `{"model":"m","max_tokens":10,"messages":[],"context_management":{}}`, "context_management"},
		{"service_tier", `{"model":"m","max_tokens":10,"messages":[],"service_tier":"auto"}`, "service_tier"},
	}
	for _, tc := range rejectCases {
		t.Run("reject_"+tc.name, func(t *testing.T) {
			_, _, err := normalizeMessagesRequest([]byte(tc.raw))
			if err == nil {
				t.Fatalf("want error for %s", tc.name)
			}
			if err.status != http.StatusBadRequest || err.typ != "invalid_request_error" {
				t.Fatalf("envelope=%+v", err)
			}
			if !strings.Contains(err.message, tc.field) {
				t.Fatalf("error message %q must name field %q", err.message, tc.field)
			}
		})
	}

	t.Run("bad_json", func(t *testing.T) {
		if _, _, err := normalizeMessagesRequest([]byte(`{not json`)); err == nil {
			t.Fatal("want error for malformed JSON")
		}
	})
}

// ---------------------------------------------------------------------------
// 2. 非流式 /v1/messages
// ---------------------------------------------------------------------------

func TestMessagesNonStreamText(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regNonStreamSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["type"] != "message" || obj["role"] != "assistant" {
		t.Fatalf("envelope=%v", obj)
	}
	if obj["model"] != "glm-5.2" {
		t.Fatalf("model=%v want glm-5.2", obj["model"])
	}
	if obj["stop_reason"] != "end_turn" || obj["stop_sequence"] != nil {
		t.Fatalf("stop=%v/%v", obj["stop_reason"], obj["stop_sequence"])
	}
	content := obj["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content=%v", content)
	}
	c0 := content[0].(map[string]any)
	if c0["type"] != "text" || c0["text"] != "你好，世界" {
		t.Fatalf("content[0]=%v", c0)
	}
	usage := obj["usage"].(map[string]any)
	if usage["input_tokens"].(float64) != 7 || usage["output_tokens"].(float64) != 5 {
		t.Fatalf("usage=%v", usage)
	}
	if _, ok := usage["cache_read_input_tokens"]; ok {
		t.Fatalf("must not fabricate cache fields: %v", usage)
	}
	if !strings.HasPrefix(obj["id"].(string), "msg_") {
		t.Fatalf("id=%v want msg_ prefix", obj["id"])
	}
}

func TestMessagesNonStreamToolCall(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regFragmentedToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["stop_reason"] != "tool_use" {
		t.Fatalf("stop_reason=%v want tool_use", obj["stop_reason"])
	}
	content := obj["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content=%v", content)
	}
	tu := content[0].(map[string]any)
	if tu["type"] != "tool_use" || tu["id"] != "call_frag" || tu["name"] != "exec_command" {
		t.Fatalf("tool_use=%v", tu)
	}
	input := tu["input"].(map[string]any)
	if input["cmd"] != "ls -la" {
		t.Fatalf("input=%v want parsed object", input)
	}
	usage := obj["usage"].(map[string]any)
	if usage["input_tokens"].(float64) != 3 || usage["output_tokens"].(float64) != 9 {
		t.Fatalf("usage=%v", usage)
	}
}

func TestMessagesNonStreamInvalidToolArgs(t *testing.T) {
	up, _ := staticRegressionUpstream(200, msgBadToolArgsSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("code=%d want 502 body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["type"] != "error" {
		t.Fatalf("envelope=%v", obj)
	}
	e := obj["error"].(map[string]any)
	if e["type"] != "api_error" || !strings.Contains(e["message"].(string), "invalid tool arguments") {
		t.Fatalf("error=%v", e)
	}
}

func TestMessagesStopReasonMapping(t *testing.T) {
	cases := []struct {
		finish string
		want   string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"content_filter", "refusal"},
		{"", "end_turn"},
	}
	for _, tc := range cases {
		t.Run("finish_"+tc.finish, func(t *testing.T) {
			up, _ := staticRegressionUpstream(200, msgFinishSSE(tc.finish), true)
			p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
			h := NewHandler(Config{Pool: p, Upstream: up})
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
				strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))
			if rec.Code != http.StatusOK {
				t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
			}
			obj := mustDecodeJSON(t, rec.Body.Bytes())
			if obj["stop_reason"] != tc.want {
				t.Fatalf("stop_reason=%v want %v", obj["stop_reason"], tc.want)
			}
			usage := obj["usage"].(map[string]any)
			if usage["input_tokens"].(float64) != 7 || usage["output_tokens"].(float64) != 5 {
				t.Fatalf("usage=%v", usage)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 3. 流式 /v1/messages 事件生命周期
// ---------------------------------------------------------------------------

func TestMessagesStreamTextSequence(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regNonStreamSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type=%q", ct)
	}
	events := parseSSEEvents(t, rec.Body.String())

	want := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
		"message_delta", "message_stop",
	}
	got := eventNames(events)
	if len(got) != len(want) {
		t.Fatalf("events=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event[%d]=%q want %q (all=%v)", i, got[i], want[i], got)
		}
	}
	// data.type 必须与 event 名一致。
	for _, e := range events {
		if e.data["type"] != e.name {
			t.Fatalf("event %q data.type=%v mismatch", e.name, e.data["type"])
		}
	}
	// message_start 恰好一次，且 message.id 带 msg_ 前缀。
	starts := eventsNamed(events, "message_start")
	if len(starts) != 1 {
		t.Fatalf("message_start count=%d want 1", len(starts))
	}
	if id := starts[0].data["message"].(map[string]any)["id"].(string); !strings.HasPrefix(id, "msg_") {
		t.Fatalf("message.id=%q", id)
	}
	// deltas 拼接 == 完整正文。
	var text strings.Builder
	for _, e := range eventsNamed(events, "content_block_delta") {
		text.WriteString(e.data["delta"].(map[string]any)["text"].(string))
	}
	if text.String() != "你好，世界" {
		t.Fatalf("concatenated text=%q", text.String())
	}
	// message_delta 的 stop_reason / usage。
	md := eventsNamed(events, "message_delta")[0].data
	if md["delta"].(map[string]any)["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason=%v", md["delta"])
	}
	if md["usage"].(map[string]any)["output_tokens"].(float64) != 5 {
		t.Fatalf("usage=%v", md["usage"])
	}
}

func TestMessagesStreamToolUseFragmented(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regFragmentedToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())

	want := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
		"message_delta", "message_stop",
	}
	got := eventNames(events)
	if len(got) != len(want) {
		t.Fatalf("events=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event[%d]=%q want %q", i, got[i], want[i])
		}
	}
	start := eventsNamed(events, "content_block_start")[0].data
	cb := start["content_block"].(map[string]any)
	if cb["type"] != "tool_use" || cb["id"] != "call_frag" || cb["name"] != "exec_command" {
		t.Fatalf("content_block=%v", cb)
	}
	if start["index"].(float64) != 0 {
		t.Fatalf("start index=%v want 0", start["index"])
	}
	var args strings.Builder
	for _, e := range eventsNamed(events, "content_block_delta") {
		d := e.data["delta"].(map[string]any)
		if d["type"] != "input_json_delta" {
			t.Fatalf("delta type=%v", d["type"])
		}
		if e.data["index"].(float64) != 0 {
			t.Fatalf("delta index=%v want 0", e.data["index"])
		}
		args.WriteString(d["partial_json"].(string))
	}
	if args.String() != `{"cmd":"ls -la"}` {
		t.Fatalf("assembled arguments=%q", args.String())
	}
	if md := eventsNamed(events, "message_delta")[0].data; md["delta"].(map[string]any)["stop_reason"] != "tool_use" {
		t.Fatalf("stop_reason=%v", md["delta"])
	}
}

func TestMessagesStreamTextThenToolIndexes(t *testing.T) {
	up, _ := staticRegressionUpstream(200, msgTextThenToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())

	want := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_stop",
		"content_block_start", "content_block_delta", "content_block_stop",
		"message_delta", "message_stop",
	}
	got := eventNames(events)
	if len(got) != len(want) {
		t.Fatalf("events=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event[%d]=%q want %q (all=%v)", i, got[i], want[i], got)
		}
	}
	// 文本块 index 0，工具块 index 1（单调）。
	starts := eventsNamed(events, "content_block_start")
	if starts[0].data["index"].(float64) != 0 {
		t.Fatalf("text block index=%v want 0", starts[0].data["index"])
	}
	if starts[0].data["content_block"].(map[string]any)["type"] != "text" {
		t.Fatalf("first block not text: %v", starts[0].data)
	}
	if starts[1].data["index"].(float64) != 1 {
		t.Fatalf("tool block index=%v want 1", starts[1].data["index"])
	}
	if starts[1].data["content_block"].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("second block not tool_use: %v", starts[1].data)
	}
	// 文本块必须先 stop（index 0），再开工具块。
	stops := eventsNamed(events, "content_block_stop")
	if stops[0].data["index"].(float64) != 0 || stops[1].data["index"].(float64) != 1 {
		t.Fatalf("stops=%v", stops)
	}
}

func TestMessagesStreamErrorFrame(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respErrorSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	events := parseSSEEvents(t, rec.Body.String())
	errs := eventsNamed(events, "error")
	if len(errs) != 1 {
		t.Fatalf("error event count=%d want 1 (%v)", len(errs), eventNames(events))
	}
	e := errs[0].data["error"].(map[string]any)
	if e["type"] != "api_error" || e["message"] != "rate limited" {
		t.Fatalf("error event=%v", e)
	}
	// 终态是 error 事件本身：失败流不得再补 message_delta/message_stop 成功收尾
	// （与 Responses 的 sawError → response.failed 同语义）。
	if last := events[len(events)-1]; last.name != "error" {
		t.Fatalf("terminal event=%q want error (all=%v)", last.name, eventNames(events))
	}
	if n := len(eventsNamed(events, "message_delta")); n != 0 {
		t.Fatalf("error stream must NOT emit message_delta (all=%v)", eventNames(events))
	}
	if n := len(eventsNamed(events, "message_stop")); n != 0 {
		t.Fatalf("error stream must NOT emit message_stop (all=%v)", eventNames(events))
	}
}

// TestMessagesStreamErrorAfterPartialText 上游先给正文、再以 error 帧报错 → 终态是
// error 事件，之后不得出现任何成功终态（message_delta / message_stop）。
func TestMessagesStreamErrorAfterPartialText(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respErrorSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	events := parseSSEEvents(t, rec.Body.String())
	if n := len(eventsNamed(events, "error")); n != 1 {
		t.Fatalf("error event count=%d want 1 (%v)", n, eventNames(events))
	}
	// 部分正文已交付（error 之前的 content_block_delta）。
	if n := len(eventsNamed(events, "content_block_delta")); n == 0 {
		t.Fatalf("partial text should be delivered before the error (all=%v)", eventNames(events))
	}
	if last := events[len(events)-1]; last.name != "error" {
		t.Fatalf("terminal event=%q want error (all=%v)", last.name, eventNames(events))
	}
	if n := len(eventsNamed(events, "message_delta")); n != 0 {
		t.Fatalf("must NOT emit message_delta after error (all=%v)", eventNames(events))
	}
	if n := len(eventsNamed(events, "message_stop")); n != 0 {
		t.Fatalf("must NOT emit message_stop after error (all=%v)", eventNames(events))
	}
}

// TestMessagesStreamNormalTerminal 正常 [DONE] 流 → 恰好一个 message_stop 成功终态。
func TestMessagesStreamNormalTerminal(t *testing.T) {
	up, _ := staticRegressionUpstream(200, msgFinishSSE("stop"), true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	events := parseSSEEvents(t, rec.Body.String())
	if n := len(eventsNamed(events, "message_stop")); n != 1 {
		t.Fatalf("message_stop count=%d want 1 (all=%v)", n, eventNames(events))
	}
	if n := len(eventsNamed(events, "message_delta")); n != 1 {
		t.Fatalf("message_delta count=%d want 1 (all=%v)", n, eventNames(events))
	}
	if last := events[len(events)-1]; last.name != "message_stop" {
		t.Fatalf("terminal event=%q want message_stop (all=%v)", last.name, eventNames(events))
	}
}

func TestMessagesEmptyStream(t *testing.T) {
	up, _ := staticRegressionUpstream(200, "data: [DONE]\n\n", true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	events := parseSSEEvents(t, rec.Body.String())
	if len(events) == 0 {
		t.Fatalf("empty stream must still produce a coherent lifecycle: %q", rec.Body.String())
	}
	if events[0].name != "message_start" {
		t.Fatalf("first event=%q want message_start", events[0].name)
	}
	// 空流兜底 error 帧 → 终态是 error 事件本身，不再补成功终态。
	if last := events[len(events)-1]; last.name != "error" {
		t.Fatalf("terminal event=%q want error (all=%v)", last.name, eventNames(events))
	}
	if len(eventsNamed(events, "error")) == 0 {
		t.Fatalf("empty stream should surface an error event (all=%v)", eventNames(events))
	}
	if n := len(eventsNamed(events, "message_stop")); n != 0 {
		t.Fatalf("empty stream must NOT emit message_stop (all=%v)", eventNames(events))
	}
	if n := len(eventsNamed(events, "message_delta")); n != 0 {
		t.Fatalf("empty stream must NOT emit message_delta (all=%v)", eventNames(events))
	}
	// 失败流不再补空 text 块（Done 早退），无 content_block_start/stop。
	if n := len(eventsNamed(events, "content_block_start")); n != 0 {
		t.Fatalf("empty stream must NOT open content blocks (all=%v)", eventNames(events))
	}
}

func TestMessagesSinkWriteFailure(t *testing.T) {
	sink := newMessagesSink(failingWriter{}, nil, messagesMeta{model: "m"})
	frame := map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": "x"}}}}
	if err := sink.Chunk(frame, ""); err == nil {
		t.Fatal("sink write failure must propagate as an error")
	}
}

// ---------------------------------------------------------------------------
// 4. 路由 / 鉴权 / 校验错误
// ---------------------------------------------------------------------------

func TestMessagesRouteAndAuth(t *testing.T) {
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})

	t.Run("unauthorized", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sseOK, true)
		h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "secret"})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
			strings.NewReader(`{"model":"glm-5.2","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("code=%d want 401 body=%s", rec.Code, rec.Body)
		}
		obj := mustDecodeJSON(t, rec.Body.Bytes())
		if obj["type"] != "error" {
			t.Fatalf("envelope=%v", obj)
		}
		if e := obj["error"].(map[string]any); e["type"] != "authentication_error" {
			t.Fatalf("error=%v", e)
		}
	})

	t.Run("wrong_method_not_routed", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sseOK, true)
		h := NewHandler(Config{Pool: p, Upstream: up})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/messages", nil))
		if rec.Code != http.StatusMethodNotAllowed {
			t.Fatalf("code=%d want 405 body=%s", rec.Code, rec.Body)
		}
	})

	t.Run("validation_error_envelope", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sseOK, true)
		h := NewHandler(Config{Pool: p, Upstream: up})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
			strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("code=%d want 400 body=%s", rec.Code, rec.Body)
		}
		obj := mustDecodeJSON(t, rec.Body.Bytes())
		e := obj["error"].(map[string]any)
		if e["type"] != "invalid_request_error" || !strings.Contains(e["message"].(string), "max_tokens") {
			t.Fatalf("error envelope=%v", e)
		}
	})
}

// ---------------------------------------------------------------------------
// 5. 模型别名 + 提示词模式（出站 body）
// ---------------------------------------------------------------------------

func TestMessagesModelAliasOutbound(t *testing.T) {
	tbl, err := NewAliasTable([]ModelAlias{{From: "claude-sonnet-4-5", To: "cn:glm-5.2"}})
	if err != nil {
		t.Fatalf("build alias table: %v", err)
	}
	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, ModelAliases: tbl})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"claude-sonnet-4-5","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) != 1 {
		t.Fatalf("upstream attempts=%d want 1", len(bodies))
	}
	out := mustDecodeJSON(t, bodies[0])
	if out["model"] != "glm-5.2" {
		t.Fatalf("outbound model=%v want glm-5.2 (alias resolved to bare name)", out["model"])
	}
	if strings.Contains(string(bodies[0]), "claude-sonnet-4-5") {
		t.Fatalf("alias source must not leak upstream: %s", bodies[0])
	}
	resp := mustDecodeJSON(t, rec.Body.Bytes())
	if resp["model"] != "claude-sonnet-4-5" {
		t.Fatalf("response model=%v want claude-sonnet-4-5 (echo client model)", resp["model"])
	}
}

func TestMessagesPromptModes(t *testing.T) {
	run := func(promptMode, promptText string) []byte {
		var bodies [][]byte
		up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
		p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
		h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: promptMode, PromptText: promptText})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
			strings.NewReader(`{"model":"glm-5.2","max_tokens":10,"system":"CLIENT_SYS","messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
		}
		if len(bodies) != 1 {
			t.Fatalf("upstream attempts=%d want 1", len(bodies))
		}
		return bodies[0]
	}

	// passthrough：客户端 system 原样出站。
	pt := outboundMessages(t, run("passthrough", ""))
	if n := countSystemContent(pt, "CLIENT_SYS"); n != 1 {
		t.Fatalf("passthrough CLIENT_SYS count=%d want 1 (msgs=%v)", n, pt)
	}

	// custom：客户端 system 被网关 PromptText 替换。
	cu := run("custom", "GATEWAY_SYS")
	if strings.Contains(string(cu), "CLIENT_SYS") {
		t.Fatalf("custom must replace client system: %s", cu)
	}
	if n := countSystemContent(outboundMessages(t, cu), "GATEWAY_SYS"); n != 1 {
		t.Fatalf("custom GATEWAY_SYS count=%d want 1", n)
	}
}
