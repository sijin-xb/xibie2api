// responses_test.go —— /v1/responses（OpenAI Responses API）的行为测试。
//
// 覆盖：请求归一化（支持矩阵 + 拒绝项）、非流式转换、流式事件序列、工具调用事件、
// 上游 error 帧、空流终态、模型别名、提示词模式、路由/鉴权/校验错误。
// 复用 handler_test.go / prompt_modes_test.go 的夹具（newFakeUpstream、
// testPoolWith、sseOK、captureUpstream、outboundMessages、roundTripFunc）。
package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/reqlog"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// respTextSSE：多段正文 + stop + usage（流式/非流式共用）。
const respTextSSE = regNonStreamSSE

// respToolSSE：function_call 的 name + arguments 分片（流式工具调用事件输入）。
const respToolSSE = regFragmentedToolSSE

// respErrorSSE：一段正文后上游以 error 帧报错（6004 限流形态）。
const respErrorSSE = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"}}]}\n\n" +
	"data: {\"error\":{\"message\":\"rate limited\",\"code\":6004}}\n\n" +
	"data: [DONE]\n\n"

type sseEvent struct {
	name string
	data map[string]any
}

// parseSSEEvents 把 SSE 响应体解析为 (event, data) 列表；跳过无 event: 行的块。
func parseSSEEvents(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var name, dataLine string
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(line, "event: ") {
				name = strings.TrimPrefix(line, "event: ")
			} else if strings.HasPrefix(line, "data: ") {
				dataLine = strings.TrimPrefix(line, "data: ")
			}
		}
		if name == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(dataLine), &m); err != nil {
			t.Fatalf("event %s: data not JSON: %v (%q)", name, err, dataLine)
		}
		events = append(events, sseEvent{name: name, data: m})
	}
	return events
}

func eventNames(events []sseEvent) []string {
	names := make([]string, 0, len(events))
	for _, e := range events {
		names = append(names, e.name)
	}
	return names
}

func eventsNamed(events []sseEvent, name string) []sseEvent {
	var out []sseEvent
	for _, e := range events {
		if e.name == name {
			out = append(out, e)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// 1. normalizeResponsesRequest
// ---------------------------------------------------------------------------

func TestNormalizeResponsesRequest(t *testing.T) {
	t.Run("string_input", func(t *testing.T) {
		body, meta, err := normalizeResponsesRequest([]byte(`{"model":"glm-5.2","input":"hello"}`))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if meta.model != "glm-5.2" || meta.stream {
			t.Fatalf("meta=%+v", meta)
		}
		obj := mustDecodeJSON(t, body)
		msgs := obj["messages"].([]any)
		if len(msgs) != 1 {
			t.Fatalf("messages=%v", msgs)
		}
		m := msgs[0].(map[string]any)
		if m["role"] != "user" || m["content"] != "hello" {
			t.Fatalf("message=%v", m)
		}
	})

	t.Run("message_array_with_parts", func(t *testing.T) {
		raw := `{"model":"glm-5.2","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`
		body, _, err := normalizeResponsesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		m := mustDecodeJSON(t, body)["messages"].([]any)[0].(map[string]any)
		if m["role"] != "user" {
			t.Fatalf("role=%v", m["role"])
		}
		parts := m["content"].([]any)
		p0 := parts[0].(map[string]any)
		if p0["type"] != "text" || p0["text"] != "hi" {
			t.Fatalf("part=%v", p0)
		}
	})

	t.Run("instructions_prepended_as_system", func(t *testing.T) {
		raw := `{"model":"glm-5.2","instructions":"SYS","input":"hi"}`
		body, meta, err := normalizeResponsesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if meta.instructions != "SYS" {
			t.Fatalf("meta.instructions=%v", meta.instructions)
		}
		msgs := mustDecodeJSON(t, body)["messages"].([]any)
		if len(msgs) != 2 {
			t.Fatalf("messages=%v want 2", msgs)
		}
		if m := msgs[0].(map[string]any); m["role"] != "system" || m["content"] != "SYS" {
			t.Fatalf("msgs[0]=%v", m)
		}
		if m := msgs[1].(map[string]any); m["role"] != "user" || m["content"] != "hi" {
			t.Fatalf("msgs[1]=%v", m)
		}
	})

	t.Run("tools_flat_to_nested", func(t *testing.T) {
		raw := `{"model":"glm-5.2","input":"hi","tools":[{"type":"function","name":"f","description":"d","parameters":{"type":"object"}}]}`
		body, meta, err := normalizeResponsesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		tools := mustDecodeJSON(t, body)["tools"].([]any)
		fn := tools[0].(map[string]any)["function"].(map[string]any)
		if fn["name"] != "f" || fn["description"] != "d" {
			t.Fatalf("tool fn=%v", fn)
		}
		if len(meta.tools) != 1 {
			t.Fatalf("meta.tools=%v", meta.tools)
		}
	})

	t.Run("function_call_and_output_roundtrip", func(t *testing.T) {
		raw := `{"model":"glm-5.2","input":[` +
			`{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"x\":1}"},` +
			`{"type":"function_call_output","call_id":"c1","output":"ok"}]}`
		body, _, err := normalizeResponsesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		msgs := mustDecodeJSON(t, body)["messages"].([]any)
		if len(msgs) != 2 {
			t.Fatalf("messages=%v want 2", msgs)
		}
		asst := msgs[0].(map[string]any)
		if asst["role"] != "assistant" {
			t.Fatalf("asst=%v", asst)
		}
		tc := asst["tool_calls"].([]any)[0].(map[string]any)
		if tc["id"] != "c1" || tc["type"] != "function" {
			t.Fatalf("tool_call=%v", tc)
		}
		if fn := tc["function"].(map[string]any); fn["name"] != "f" || fn["arguments"] != `{"x":1}` {
			t.Fatalf("fn=%v", fn)
		}
		tool := msgs[1].(map[string]any)
		if tool["role"] != "tool" || tool["tool_call_id"] != "c1" || tool["content"] != "ok" {
			t.Fatalf("tool msg=%v", tool)
		}
	})

	t.Run("reasoning_item_ignored", func(t *testing.T) {
		raw := `{"model":"glm-5.2","input":[{"type":"reasoning","summary":[]},{"type":"message","role":"user","content":"hi"}]}`
		body, _, err := normalizeResponsesRequest([]byte(raw))
		if err != nil {
			t.Fatalf("unexpected error: %+v", err)
		}
		if msgs := mustDecodeJSON(t, body)["messages"].([]any); len(msgs) != 1 {
			t.Fatalf("reasoning must be ignored, messages=%v", msgs)
		}
	})

	t.Run("reject_unsupported_item_type", func(t *testing.T) {
		raw := `{"model":"glm-5.2","input":[{"type":"bogus"}]}`
		_, _, err := normalizeResponsesRequest([]byte(raw))
		if err == nil || err.param != "input" {
			t.Fatalf("want input param error, got %+v", err)
		}
	})

	rejectCases := []struct {
		name  string
		raw   string
		param string
	}{
		{"previous_response_id", `{"model":"m","input":"hi","previous_response_id":"resp_1"}`, "previous_response_id"},
		{"conversation", `{"model":"m","input":"hi","conversation":"conv_1"}`, "conversation"},
		{"background", `{"model":"m","input":"hi","background":true}`, "background"},
		{"truncation_auto", `{"model":"m","input":"hi","truncation":"auto"}`, "truncation"},
		{"missing_model", `{"input":"hi"}`, "model"},
		{"missing_input", `{"model":"m"}`, "input"},
		{"model_wrong_type", `{"model":123,"input":"hi"}`, "model"},
	}
	for _, tc := range rejectCases {
		t.Run("reject_"+tc.name, func(t *testing.T) {
			_, _, err := normalizeResponsesRequest([]byte(tc.raw))
			if err == nil {
				t.Fatalf("want error for %s", tc.name)
			}
			if err.param != tc.param {
				t.Fatalf("param=%q want %q", err.param, tc.param)
			}
			if err.status != http.StatusBadRequest || err.typ != "invalid_request_error" {
				t.Fatalf("envelope=%+v", err)
			}
		})
	}

	t.Run("bad_json", func(t *testing.T) {
		_, _, err := normalizeResponsesRequest([]byte(`{not json`))
		if err == nil {
			t.Fatal("want error for malformed JSON")
		}
	})

	t.Run("truncation_disabled_accepted", func(t *testing.T) {
		if _, _, err := normalizeResponsesRequest([]byte(`{"model":"m","input":"hi","truncation":"disabled"}`)); err != nil {
			t.Fatalf("truncation disabled must be accepted: %+v", err)
		}
	})
}

// ---------------------------------------------------------------------------
// 2. 非流式 /v1/responses
// ---------------------------------------------------------------------------

func TestResponsesNonStream(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respTextSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","input":"hi"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["object"] != "response" || obj["status"] != "completed" {
		t.Fatalf("envelope=%v", obj)
	}
	if obj["model"] != "glm-5.2" {
		t.Fatalf("model=%v want glm-5.2", obj["model"])
	}
	output := obj["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output=%v want 1 item", output)
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" || item["status"] != "completed" || item["role"] != "assistant" {
		t.Fatalf("item=%v", item)
	}
	content := item["content"].([]any)[0].(map[string]any)
	if content["type"] != "output_text" || content["text"] != "你好，世界" {
		t.Fatalf("content=%v", content)
	}
	usage := obj["usage"].(map[string]any)
	if usage["input_tokens"].(float64) != 7 || usage["output_tokens"].(float64) != 5 || usage["total_tokens"].(float64) != 12 {
		t.Fatalf("usage=%v", usage)
	}
	if _, ok := obj["output_text"]; ok {
		t.Fatalf("output_text is an SDK-computed property, must not be a wire field: %v", obj["output_text"])
	}
}

// ---------------------------------------------------------------------------
// 3. 流式 /v1/responses 事件序列
// ---------------------------------------------------------------------------

func TestResponsesStreamEventSequence(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respTextSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type=%q", ct)
	}
	events := parseSSEEvents(t, rec.Body.String())

	want := []string{
		"response.created", "response.in_progress",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta", "response.output_text.delta",
		"response.output_text.done", "response.content_part.done",
		"response.output_item.done", "response.completed",
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

	// data.type 必须与 event 名一致；sequence_number 单调从 1 起。
	var text strings.Builder
	for i, e := range events {
		if e.data["type"] != e.name {
			t.Fatalf("event %q data.type=%v mismatch", e.name, e.data["type"])
		}
		if got := e.data["sequence_number"].(float64); int(got) != i+1 {
			t.Fatalf("sequence_number=%v want %d", got, i+1)
		}
		if e.name == "response.output_text.delta" {
			text.WriteString(e.data["delta"].(string))
		}
	}
	if text.String() != "你好，世界" {
		t.Fatalf("concatenated deltas=%q want 你好，世界", text.String())
	}

	// response.completed 的 output 含完整 message item。
	completed := eventsNamed(events, "response.completed")[0].data
	resp := completed["response"].(map[string]any)
	if resp["status"] != "completed" {
		t.Fatalf("completed status=%v", resp["status"])
	}
	output := resp["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("completed output=%v", output)
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" {
		t.Fatalf("completed item=%v", item)
	}
	if c := item["content"].([]any)[0].(map[string]any); c["text"] != "你好，世界" {
		t.Fatalf("completed content=%v", c)
	}
}

// ---------------------------------------------------------------------------
// 4. 工具调用事件
// ---------------------------------------------------------------------------

func TestResponsesStreamToolCall(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())

	added := eventsNamed(events, "response.output_item.added")
	if len(added) != 1 {
		t.Fatalf("output_item.added count=%d want 1 (%v)", len(added), eventNames(events))
	}
	item := added[0].data["item"].(map[string]any)
	if item["type"] != "function_call" || item["call_id"] != "call_frag" || item["name"] != "exec_command" {
		t.Fatalf("function_call item=%v", item)
	}

	deltas := eventsNamed(events, "response.function_call_arguments.delta")
	if len(deltas) != 2 {
		t.Fatalf("arguments.delta count=%d want 2 (%v)", len(deltas), eventNames(events))
	}
	var args strings.Builder
	for _, d := range deltas {
		args.WriteString(d.data["delta"].(string))
	}
	if args.String() != `{"cmd":"ls -la"}` {
		t.Fatalf("accumulated arguments=%q", args.String())
	}
	done := eventsNamed(events, "response.function_call_arguments.done")
	if len(done) != 1 || done[0].data["arguments"] != `{"cmd":"ls -la"}` {
		t.Fatalf("arguments.done=%v", done)
	}

	completed := eventsNamed(events, "response.completed")[0].data["response"].(map[string]any)
	output := completed["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("completed output=%v", output)
	}
	fc := output[0].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "call_frag" ||
		fc["name"] != "exec_command" || fc["arguments"] != `{"cmd":"ls -la"}` {
		t.Fatalf("completed function_call=%v", fc)
	}
}

// Codex 依赖 item.phase 区分「回合最终回答」与「过程播报」：纯文本回合必须是
// final_answer，否则客户端会把进度播报当正式回答。
func TestResponsesMessagePhaseFinalAnswer(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respTextSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	resp := eventsNamed(events, "response.completed")[0].data["response"].(map[string]any)
	item := resp["output"].([]any)[0].(map[string]any)
	if item["phase"] != "final_answer" {
		t.Fatalf("phase=%v want final_answer", item["phase"])
	}
}

// 出现工具调用的回合，工具前正文必须是 commentary，工具后正文必须是 final_answer。
func TestResponsesMessagePhaseAcrossToolBoundary(t *testing.T) {
	// 正文先于工具调用出现，构造「先说明、再调工具」的真实形态。
	const textThenToolSSE = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"我先看看目录。\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_t1\",\"type\":\"function\",\"function\":{\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\":\\\"ls -la\\\"}\"}}]}}]}\n\n" +
		"data: [DONE]\n\n"

	up, _ := staticRegressionUpstream(200, textThenToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	assertResponsesMessagePhases(t, events, []string{"commentary"})
}

// 同一上游回合在工具结果之后继续输出正文时，必须拆成两个 message item：
// 工具前是 commentary，工具后是 final_answer。合并会让 Codex 在工具前停止，
// 或把最终回答误当成过程播报而不结束回合。
func TestResponsesMessagePhaseSplitsAcrossToolResult(t *testing.T) {
	const commentaryThenToolThenFinalSSE = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"我先检查配置。\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_t1\",\"type\":\"function\",\"function\":{\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\":\\\"cat config\\\"}\"}}]}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"tool\",\"content\":\"配置正常。\"}}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"最终答案是 42。\"}}]}\n\n" +
		"data: [DONE]\n\n"

	up, _ := staticRegressionUpstream(200, commentaryThenToolThenFinalSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	assertResponsesMessagePhases(t, events, []string{"commentary", "final_answer"})
}

// Codex 回放历史时会携带 phase。最终回答出现在工具结果之后，而输入里仍保留
// 工具前的 commentary；响应必须保留这个边界，不能把历史 phase 当成当前正文。
func TestResponsesMessagePhaseWithToolHistory(t *testing.T) {
	const finalAfterToolSSE = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1753600000,\"model\":\"glm-5.2\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"最终答案是 42。\"}}]}\n\n" +
		"data: [DONE]\n\n"

	up, _ := staticRegressionUpstream(200, finalAfterToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	body := `{"model":"glm-5.2","stream":true,"input":[` +
		`{"type":"message","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"我先检查配置。"}]},` +
		`{"type":"function_call","call_id":"call_t1","name":"exec_command","arguments":"{\"cmd\":\"cat config\"}"},` +
		`{"type":"function_call_output","call_id":"call_t1","output":"配置正常。"},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"继续"}]}` +
		`]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	assertResponsesMessagePhases(t, events, []string{"final_answer"})
}

// 上游已经以 stop 正常结束当前响应时，网关必须如实结束该响应。
// 正文是否像“接下来会……”属于模型语义，不是协议终止信号；网关自行再发一次
// 上游请求会绕过 Codex 的回合循环，并与客户端的重试边界冲突。
func TestResponsesStopDoesNotTriggerGatewayContinuation(t *testing.T) {
	promised := `data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"接下来我会检查本机工具。"}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	up, calls := newRegressionUpstream(func(int, string, []byte) (int, string, bool) {
		return 200, promised, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"检查工具","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if got := len(calls()); got != 1 {
		t.Fatalf("upstream calls=%d want 1", got)
	}
	events := parseSSEEvents(t, rec.Body.String())
	if got := len(eventsNamed(events, "response.created")); got != 1 {
		t.Fatalf("response.created=%d want 1; events=%v", got, eventNames(events))
	}
	if got := len(eventsNamed(events, "response.completed")); got != 1 {
		t.Fatalf("response.completed=%d want 1; events=%v", got, eventNames(events))
	}
	var text strings.Builder
	for _, e := range eventsNamed(events, "response.output_text.delta") {
		text.WriteString(e.data["delta"].(string))
	}
	if got := text.String(); got != "接下来我会检查本机工具。" {
		t.Fatalf("wire text=%q want original upstream text", got)
	}
}

// 工具结果之后、最终阶段尚未执行时，模型只发一段“现在创建…”的纯文本。
// 这是过程播报，不能被标成 final_answer；否则 Codex 会把它当最终回答并结束回合。
func TestResponsesPendingActionTextIsCommentary(t *testing.T) {
	const pendingActionSSE = `data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"阶段 5 的验证器已成功输出 LONG_TASK_OK。现在创建阶段 6 的 complete.txt，并核对六个目标文件都存在且内容准确。"}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, pendingActionSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"finish all phases"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	completed := eventsNamed(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("response.completed count=%d want 1 (all=%v)", len(completed), eventNames(events))
	}
	resp := completed[0].data["response"].(map[string]any)
	item := resp["output"].([]any)[0].(map[string]any)
	if item["phase"] != "commentary" {
		t.Fatalf("phase=%v want commentary for pending action", item["phase"])
	}
}

func TestResponsesToolCallFragmentsMergeByCallID(t *testing.T) {
	sse := `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_same","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":"}}]}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_same","type":"function","function":{"arguments":"\"pwd\"}"}}]}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"m","stream":true,"input":"x"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	if got := len(eventsNamed(events, "response.output_item.added")); got != 1 {
		t.Fatalf("output_item.added=%d want 1; events=%v", got, eventNames(events))
	}
	resp := eventsNamed(events, "response.completed")[0].data["response"].(map[string]any)
	output := resp["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output=%v want one function_call", output)
	}
	item := output[0].(map[string]any)
	if item["call_id"] != "call_same" || item["arguments"] != `{"cmd":"pwd"}` {
		t.Fatalf("function_call=%v", item)
	}
}

func assertResponsesMessagePhases(t *testing.T, events []sseEvent, want []string) {
	t.Helper()
	added := make([]string, 0, len(want))
	current := make(map[string]string)
	for _, e := range eventsNamed(events, "response.output_item.added") {
		item, _ := e.data["item"].(map[string]any)
		if item == nil || item["type"] != "message" {
			continue
		}
		phase, _ := item["phase"].(string)
		if phase == "" {
			t.Fatalf("added message item has no phase: %v", item)
		}
		added = append(added, phase)
		if id, _ := item["id"].(string); id != "" {
			current[id] = phase
		}
	}
	if len(added) != len(want) {
		t.Fatalf("added message phases=%v want %v", added, want)
	}
	for _, e := range eventsNamed(events, "response.output_item.updated") {
		item, _ := e.data["item"].(map[string]any)
		if item == nil || item["type"] != "message" {
			continue
		}
		if id, _ := item["id"].(string); id != "" {
			phase, _ := item["phase"].(string)
			current[id] = phase
		}
	}

	done := make([]string, 0, len(want))
	for _, e := range eventsNamed(events, "response.output_item.done") {
		item, _ := e.data["item"].(map[string]any)
		if item == nil || item["type"] != "message" {
			continue
		}
		phase, _ := item["phase"].(string)
		done = append(done, phase)
		if id, _ := item["id"].(string); id != "" && current[id] != phase {
			t.Fatalf("done message %s phase=%q, last update=%q", id, phase, current[id])
		}
	}
	if len(done) != len(want) {
		t.Fatalf("done message phases=%v want %v", done, want)
	}
	for i := range want {
		if done[i] != want[i] {
			t.Fatalf("done message[%d] phase=%q want %q (all=%v)", i, done[i], want[i], done)
		}
	}

	completed := eventsNamed(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("response.completed count=%d", len(completed))
	}
	resp := completed[0].data["response"].(map[string]any)
	var final []string
	for _, raw := range resp["output"].([]any) {
		item, _ := raw.(map[string]any)
		if item == nil || item["type"] != "message" {
			continue
		}
		phase, _ := item["phase"].(string)
		final = append(final, phase)
	}
	if len(final) != len(want) {
		t.Fatalf("final message phases=%v want %v", final, want)
	}
	for i := range want {
		if final[i] != want[i] {
			t.Fatalf("final message[%d] phase=%q want %q (all=%v)", i, final[i], want[i], final)
		}
	}
}

// ---------------------------------------------------------------------------
// 5. 上游 error 帧
// ---------------------------------------------------------------------------

func TestResponsesStreamErrorFrame(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respErrorSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))

	events := parseSSEEvents(t, rec.Body.String())
	errs := eventsNamed(events, "error")
	if len(errs) != 1 {
		t.Fatalf("error event count=%d want 1 (%v)", len(errs), eventNames(events))
	}
	if errs[0].data["type"] != "error" {
		t.Fatalf("error data.type=%v", errs[0].data["type"])
	}
	if errs[0].data["code"] != "6004" || errs[0].data["message"] != "rate limited" {
		t.Fatalf("error event=%v", errs[0].data)
	}
	// 终态必须是 response.failed（而非 response.completed）——失败请求不得被读成成功。
	if n := len(eventsNamed(events, "response.completed")); n != 0 {
		t.Fatalf("error stream must NOT emit response.completed (all=%v)", eventNames(events))
	}
	failed := eventsNamed(events, "response.failed")
	if len(failed) != 1 {
		t.Fatalf("response.failed count=%d want 1 (all=%v)", len(failed), eventNames(events))
	}
	if last := events[len(events)-1]; last.name != "response.failed" {
		t.Fatalf("terminal event=%q want response.failed (all=%v)", last.name, eventNames(events))
	}
	resp := failed[0].data["response"].(map[string]any)
	if resp["status"] != "failed" {
		t.Fatalf("response.status=%v want failed", resp["status"])
	}
	e := resp["error"].(map[string]any)
	if e["code"] != "6004" || e["message"] != "rate limited" {
		t.Fatalf("response.error=%v", e)
	}
	if n := countResponsesTerminals(events); n != 1 {
		t.Fatalf("terminal count=%d want exactly 1 (all=%v)", n, eventNames(events))
	}
}

// ---------------------------------------------------------------------------
// 6. 空流
// ---------------------------------------------------------------------------

func TestResponsesEmptyStream(t *testing.T) {
	up, _ := staticRegressionUpstream(200, "data: [DONE]\n\n", true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))

	events := parseSSEEvents(t, rec.Body.String())
	if len(events) == 0 {
		t.Fatalf("empty stream must still produce a coherent event stream: %q", rec.Body.String())
	}
	// 空流由 pumpFrames 交付兜底 error 帧 → 终态必须是 response.failed，而非 completed。
	last := events[len(events)-1]
	if last.name != "response.failed" {
		t.Fatalf("terminal event=%q want response.failed (all=%v)", last.name, eventNames(events))
	}
	if n := len(eventsNamed(events, "response.completed")); n != 0 {
		t.Fatalf("empty stream must NOT emit response.completed (all=%v)", eventNames(events))
	}
	if len(eventsNamed(events, "error")) == 0 {
		t.Fatalf("empty stream should surface an error event (all=%v)", eventNames(events))
	}
	if n := countResponsesTerminals(events); n != 1 {
		t.Fatalf("terminal count=%d want exactly 1 (all=%v)", n, eventNames(events))
	}
}

// countResponsesTerminals 统计 Responses 流里的终态事件（response.completed /
// response.failed）。用于断言「恰好一个终态」且「绝不成功+失败并存」。
func countResponsesTerminals(events []sseEvent) int {
	n := 0
	for _, e := range events {
		if e.name == "response.completed" || e.name == "response.failed" {
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// 6b. 流终态语义：失败绝不 response.completed
// ---------------------------------------------------------------------------

// TestResponsesStreamErrorAfterPartialContent 上游先给正文、再以 error 帧报错 →
// 终态 response.failed，且 response.output 仍携带已产生的部分正文（不丢）。
func TestResponsesStreamErrorAfterPartialContent(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respErrorSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))

	events := parseSSEEvents(t, rec.Body.String())
	// 部分正文已交付。
	var text strings.Builder
	for _, e := range eventsNamed(events, "response.output_text.delta") {
		text.WriteString(e.data["delta"].(string))
	}
	if text.String() != "partial" {
		t.Fatalf("partial text=%q want \"partial\"", text.String())
	}
	failed := eventsNamed(events, "response.failed")
	if len(failed) != 1 {
		t.Fatalf("response.failed count=%d want 1 (all=%v)", len(failed), eventNames(events))
	}
	resp := failed[0].data["response"].(map[string]any)
	output := resp["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output=%v want 1 partial message item", output)
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" {
		t.Fatalf("partial item=%v", item)
	}
	if c := item["content"].([]any)[0].(map[string]any); c["text"] != "partial" {
		t.Fatalf("partial output text=%v want \"partial\"", c["text"])
	}
	if len(eventsNamed(events, "response.completed")) != 0 {
		t.Fatalf("must NOT emit response.completed (all=%v)", eventNames(events))
	}
}

// TestResponsesStreamNormalCompleted 正常 [DONE] → 恰好一个 response.completed，
// status completed，且无 response.failed。
func TestResponsesStreamNormalCompleted(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))

	events := parseSSEEvents(t, rec.Body.String())
	completed := eventsNamed(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("response.completed count=%d want 1 (all=%v)", len(completed), eventNames(events))
	}
	if last := events[len(events)-1]; last.name != "response.completed" {
		t.Fatalf("terminal event=%q want response.completed (all=%v)", last.name, eventNames(events))
	}
	resp := completed[0].data["response"].(map[string]any)
	if resp["status"] != "completed" {
		t.Fatalf("status=%v want completed", resp["status"])
	}
	if len(eventsNamed(events, "response.failed")) != 0 {
		t.Fatalf("must NOT emit response.failed (all=%v)", eventNames(events))
	}
	if n := countResponsesTerminals(events); n != 1 {
		t.Fatalf("terminal count=%d want exactly 1 (all=%v)", n, eventNames(events))
	}
}

// 上游达到 max_tokens 时必须以 response.incomplete 收尾，不能伪装成 completed。
func TestResponsesStreamLengthIsIncomplete(t *testing.T) {
	const lengthSSE = `data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"部分输出"}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, lengthSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	incomplete := eventsNamed(events, "response.incomplete")
	if len(incomplete) != 1 {
		t.Fatalf("response.incomplete count=%d want 1 (all=%v)", len(incomplete), eventNames(events))
	}
	if last := events[len(events)-1]; last.name != "response.incomplete" {
		t.Fatalf("terminal event=%q want response.incomplete (all=%v)", last.name, eventNames(events))
	}
	if n := len(eventsNamed(events, "response.completed")); n != 0 {
		t.Fatalf("length must NOT emit response.completed (all=%v)", eventNames(events))
	}
	resp := incomplete[0].data["response"].(map[string]any)
	if resp["status"] != "incomplete" {
		t.Fatalf("status=%v want incomplete", resp["status"])
	}
	details, _ := resp["incomplete_details"].(map[string]any)
	if details == nil || details["reason"] != "max_output_tokens" {
		t.Fatalf("incomplete_details=%v want reason=max_output_tokens", resp["incomplete_details"])
	}
}

// content_filter 同样是模型未正常完成，必须映射为 incomplete + content_filter。
func TestResponsesStreamContentFilterIsIncomplete(t *testing.T) {
	const filteredSSE = `data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"部分输出"}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, filteredSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	incomplete := eventsNamed(events, "response.incomplete")
	if len(incomplete) != 1 {
		t.Fatalf("response.incomplete count=%d want 1 (all=%v)", len(incomplete), eventNames(events))
	}
	resp := incomplete[0].data["response"].(map[string]any)
	details, _ := resp["incomplete_details"].(map[string]any)
	if resp["status"] != "incomplete" || details == nil || details["reason"] != "content_filter" {
		t.Fatalf("status=%v incomplete_details=%v want incomplete/content_filter",
			resp["status"], resp["incomplete_details"])
	}
}

func TestResponsesNonStreamLengthIsIncomplete(t *testing.T) {
	const lengthSSE = `data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"部分输出"}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, lengthSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["status"] != "incomplete" {
		t.Fatalf("status=%v want incomplete", obj["status"])
	}
	details, _ := obj["incomplete_details"].(map[string]any)
	if details == nil || details["reason"] != "max_output_tokens" {
		t.Fatalf("incomplete_details=%v want reason=max_output_tokens", obj["incomplete_details"])
	}
}

// errAfterReader 首次 Read 返回全部 data，随后 Read 返回 err（模拟上游流中途读错误）。
type errAfterReader struct {
	data []byte
	err  error
}

func (r *errAfterReader) Read(p []byte) (int, error) {
	if len(r.data) > 0 {
		n := copy(p, r.data)
		r.data = r.data[n:]
		return n, nil
	}
	return 0, r.err
}

// TestResponsesStreamUpstreamReadError 上游流中途读错误 → pumpFrames 立即返回，
// Done 不被调用 → 不写任何成功终态（也不写失败终态）。
func TestResponsesStreamUpstreamReadError(t *testing.T) {
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: io.NopCloser(&errAfterReader{
					data: []byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n"),
					err:  errors.New("upstream read failed"),
				}),
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))

	events := parseSSEEvents(t, rec.Body.String())
	if n := len(eventsNamed(events, "response.completed")); n != 0 {
		t.Fatalf("read error must NOT emit response.completed (all=%v)", eventNames(events))
	}
	if n := countResponsesTerminals(events); n != 0 {
		t.Fatalf("read error must not emit any terminal (all=%v)", eventNames(events))
	}
}

// failAfterContentWriter 记录写入；在写入包含 response.output_text.delta 的那次之后，
// 后续写入一律失败——模拟客户端在流中途断连。
type failAfterContentWriter struct {
	buf    bytes.Buffer
	failed bool
}

func (w *failAfterContentWriter) Header() http.Header { return http.Header{} }
func (w *failAfterContentWriter) WriteHeader(int)     {}
func (w *failAfterContentWriter) Write(p []byte) (int, error) {
	if w.failed {
		return 0, errors.New("write failed")
	}
	n, _ := w.buf.Write(p)
	if strings.Contains(string(p), "response.output_text.delta") {
		w.failed = true
	}
	return n, nil
}

// TestResponsesStreamSinkWriteFailure sink 写失败 → pumpFrames 立即停止，Done 不被调用
// → 失败之后不写任何终态（正文已交付但无 response.completed / response.failed）。
func TestResponsesStreamSinkWriteFailure(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	w := &failAfterContentWriter{}
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))

	body := w.buf.String()
	if !strings.Contains(body, "response.output_text.delta") {
		t.Fatalf("content delta should have been delivered before failure: %q", body)
	}
	if strings.Contains(body, "response.completed") || strings.Contains(body, "response.failed") {
		t.Fatalf("no terminal may be written after a sink write failure: %q", body)
	}
}

// ---------------------------------------------------------------------------
// 7. 模型别名
// ---------------------------------------------------------------------------

func TestResponsesModelAliasOutbound(t *testing.T) {
	tbl, err := NewAliasTable([]ModelAlias{{From: "claude-sonnet-4-5", To: "cn:glm-5.2"}})
	if err != nil {
		t.Fatalf("build alias table: %v", err)
	}
	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, ModelAliases: tbl})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"claude-sonnet-4-5","input":"hi"}`)))
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
	// 响应回显的是**原始客户端 model**（meta.model），不是解析后的裸名。
	resp := mustDecodeJSON(t, rec.Body.Bytes())
	if resp["model"] != "claude-sonnet-4-5" {
		t.Fatalf("response model=%v want claude-sonnet-4-5 (echo client model)", resp["model"])
	}
}

// 未列入 model_aliases 的模型必须在网关本地被拒绝：不选号、不出站、不返回可重试
// 的 503。否则 Codex 会把一个永久无效的模型名重试到上限，表现为回合提前结束。
func TestResponsesRejectsUnconfiguredModel(t *testing.T) {
	up, calls := staticRegressionUpstream(200, respTextSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{
		Pool: p, Upstream: up,
		AllowedModels: []string{"global:deepseek-v4.1-flash"},
	})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"gpt-5.6-luna","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400; body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	errObj, _ := obj["error"].(map[string]any)
	if errObj == nil || errObj["code"] != "model_not_allowed" {
		t.Fatalf("error=%v want model_not_allowed", obj["error"])
	}
	if got := len(calls()); got != 0 {
		t.Fatalf("upstream calls=%d want 0", got)
	}
}

// ---------------------------------------------------------------------------
// 8. 提示词模式 + instructions
// ---------------------------------------------------------------------------

func TestResponsesPromptModes(t *testing.T) {
	run := func(promptMode, promptText string) []byte {
		var bodies [][]byte
		up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
		p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
		h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: promptMode, PromptText: promptText})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
			strings.NewReader(`{"model":"glm-5.2","instructions":"INSTR","input":"hi"}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
		}
		if len(bodies) != 1 {
			t.Fatalf("upstream attempts=%d want 1", len(bodies))
		}
		return bodies[0]
	}

	// passthrough：instructions → system 原样出站。
	pt := outboundMessages(t, run("passthrough", ""))
	if n := countSystemContent(pt, "INSTR"); n != 1 {
		t.Fatalf("passthrough INSTR system count=%d want 1 (msgs=%v)", n, pt)
	}
	if pt[len(pt)-1]["role"] != "user" || pt[len(pt)-1]["content"] != "hi" {
		t.Fatalf("passthrough last msg=%v", pt[len(pt)-1])
	}

	// custom：客户端 system（来自 instructions）被网关 PromptText 替换。
	cu := run("custom", "GATEWAY_SYS")
	if strings.Contains(string(cu), "INSTR") {
		t.Fatalf("custom must replace client system (instructions): %s", cu)
	}
	if n := countSystemContent(outboundMessages(t, cu), "GATEWAY_SYS"); n != 1 {
		t.Fatalf("custom GATEWAY_SYS count=%d want 1", n)
	}
}

// ---------------------------------------------------------------------------
// 9. 路由 / 鉴权 / 校验错误
// ---------------------------------------------------------------------------

func TestResponsesRouteAndAuth(t *testing.T) {
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})

	t.Run("unauthorized", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sseOK, true)
		h := NewHandler(Config{Pool: p, Upstream: up, APIKey: "secret"})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
			strings.NewReader(`{"model":"glm-5.2","input":"hi"}`)))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("code=%d want 401 body=%s", rec.Code, rec.Body)
		}
	})

	t.Run("wrong_method_not_routed", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sseOK, true)
		h := NewHandler(Config{Pool: p, Upstream: up})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/responses", nil))
		if rec.Code != http.StatusMethodNotAllowed {
			t.Fatalf("code=%d want 405 body=%s", rec.Code, rec.Body)
		}
	})

	t.Run("validation_error_envelope", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sseOK, true)
		h := NewHandler(Config{Pool: p, Upstream: up})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
			strings.NewReader(`{"input":"hi"}`)))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("code=%d want 400 body=%s", rec.Code, rec.Body)
		}
		obj := mustDecodeJSON(t, rec.Body.Bytes())
		e := obj["error"].(map[string]any)
		if e["type"] != "invalid_request_error" || e["param"] != "model" {
			t.Fatalf("error envelope=%v", e)
		}
	})

	t.Run("request_log_path_and_client_info", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sseOK, true)
		reqLog := reqlog.New(reqlog.Config{})
		h := NewHandler(Config{
			Pool:             p,
			Upstream:         up,
			RequestLog:       reqLog,
			RecordClientInfo: true,
		})
		req := httptest.NewRequest("POST", "/v1/responses",
			strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`))
		req.Header.Set("X-Forwarded-For", "203.0.113.9")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
		}
		if got := rec.Header().Get("X-Request-Id"); !strings.HasPrefix(got, "req-") {
			t.Fatalf("X-Request-Id=%q", got)
		}
		s := reqLog.Snapshot()
		if len(s.Recent) != 1 {
			t.Fatalf("recent=%+v", s.Recent)
		}
		e := s.Recent[0]
		if e.Path != "/v1/responses" {
			t.Fatalf("event path=%q want /v1/responses", e.Path)
		}
		if e.ClientIP != "203.0.113.9" {
			t.Fatalf("client ip=%q want 203.0.113.9", e.ClientIP)
		}
	})
}

// ---------------------------------------------------------------------------
// 10. 非流式工具调用（P1 回归）：[]any 与 []map[string]any 两种内存形态
// ---------------------------------------------------------------------------

// responsesAggregateWithToolCalls 构造一个带指定 tool_calls 形态的聚合响应。
func responsesAggregateWithToolCalls(toolCalls any) map[string]any {
	return map[string]any{
		"id":      "chatcmpl-x",
		"object":  "chat.completion",
		"created": 1753600001,
		"model":   "glm-5.2",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":       "assistant",
					"content":    "",
					"tool_calls": toolCalls,
				},
				"finish_reason": "tool_calls",
			},
		},
	}
}

func toolCallObj(id, name, args string) map[string]any {
	return map[string]any{
		"id":   id,
		"type": "function",
		"function": map[string]any{
			"name":      name,
			"arguments": args,
		},
	}
}

// findFunctionCall 从 output[] 找出（唯一的）function_call item。
func findFunctionCall(t *testing.T, output []any) map[string]any {
	t.Helper()
	var fc map[string]any
	for _, it := range output {
		m, ok := it.(map[string]any)
		if !ok {
			continue
		}
		if m["type"] == "function_call" {
			if fc != nil {
				t.Fatalf("more than one function_call item: %v", output)
			}
			fc = m
		}
	}
	return fc
}

// TestResponsesFromAggregateToolCallsShape 锁定 P1 缺陷的根因：聚合响应在内存里
// tool_calls 是 []map[string]any（upstream.Aggregate 构造），JSON 解码路径才是 []any。
// 两种形态都必须产出 function_call item（此前只认 []any → 断言失败 → 静默丢弃）。
func TestResponsesFromAggregateToolCallsShape(t *testing.T) {
	meta := responsesMeta{model: "glm-5.2"}

	t.Run("slice_of_any", func(t *testing.T) {
		resp := responsesAggregateWithToolCalls([]any{
			toolCallObj("call_1", "exec_command", `{"cmd":"ls -la"}`),
		})
		obj := responsesFromAggregate(resp, meta)
		fc := findFunctionCall(t, obj["output"].([]any))
		if fc == nil {
			t.Fatalf("function_call missing for []any shape: output=%v", obj["output"])
		}
		if fc["call_id"] != "call_1" || fc["name"] != "exec_command" || fc["arguments"] != `{"cmd":"ls -la"}` {
			t.Fatalf("function_call=%v", fc)
		}
		if fc["status"] != "completed" {
			t.Fatalf("status=%v", fc["status"])
		}
	})

	t.Run("slice_of_map", func(t *testing.T) {
		resp := responsesAggregateWithToolCalls([]map[string]any{
			toolCallObj("call_1", "exec_command", `{"cmd":"ls -la"}`),
		})
		obj := responsesFromAggregate(resp, meta)
		fc := findFunctionCall(t, obj["output"].([]any))
		if fc == nil {
			t.Fatalf("function_call missing for []map[string]any shape (the real Aggregate shape): output=%v", obj["output"])
		}
		if fc["call_id"] != "call_1" || fc["name"] != "exec_command" || fc["arguments"] != `{"cmd":"ls -la"}` {
			t.Fatalf("function_call=%v", fc)
		}
	})
}

// TestResponsesFromAggregateMultipleToolCallsOrder 多个工具调用顺序保持。
func TestResponsesFromAggregateMultipleToolCallsOrder(t *testing.T) {
	resp := responsesAggregateWithToolCalls([]map[string]any{
		toolCallObj("call_a", "first", `{"a":1}`),
		toolCallObj("call_b", "second", `{"b":2}`),
	})
	obj := responsesFromAggregate(resp, responsesMeta{model: "m"})
	output := obj["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output=%v want 2 function_call items", output)
	}
	first := output[0].(map[string]any)
	second := output[1].(map[string]any)
	if first["type"] != "function_call" || first["call_id"] != "call_a" || first["name"] != "first" || first["arguments"] != `{"a":1}` {
		t.Fatalf("first=%v", first)
	}
	if second["type"] != "function_call" || second["call_id"] != "call_b" || second["name"] != "second" || second["arguments"] != `{"b":2}` {
		t.Fatalf("second=%v", second)
	}
}

// TestResponsesFromAggregateTextUnchanged 纯文本完成：单个 message/output_text，无 function_call。
func TestResponsesFromAggregateTextUnchanged(t *testing.T) {
	resp := map[string]any{
		"id": "chatcmpl-t", "object": "chat.completion", "created": 1753600000, "model": "glm-5.2",
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": "hello"},
			"finish_reason": "stop",
		}},
	}
	obj := responsesFromAggregate(resp, responsesMeta{model: "glm-5.2"})
	output := obj["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output=%v want 1 message item", output)
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Fatalf("item=%v", item)
	}
	if c := item["content"].([]any)[0].(map[string]any); c["type"] != "output_text" || c["text"] != "hello" {
		t.Fatalf("content=%v", c)
	}
	if findFunctionCall(t, output) != nil {
		t.Fatalf("text-only response must not contain function_call: %v", output)
	}
}

// TestResponsesNonStreamToolCallsEndToEnd 端到端：fake 上游经 upstream.Aggregate 产出
// 真实 []map[string]any 形态 → /v1/responses 非流式响应必须含 function_call item。
// 该用例在修复前会失败（静默丢弃），修复后通过。
func TestResponsesNonStreamToolCallsEndToEnd(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regFragmentedToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["status"] != "completed" {
		t.Fatalf("status=%v", obj["status"])
	}
	output := obj["output"].([]any)
	fc := findFunctionCall(t, output)
	if fc == nil {
		t.Fatalf("non-stream responses must include a function_call item; output=%v", output)
	}
	if fc["call_id"] != "call_frag" || fc["name"] != "exec_command" || fc["arguments"] != `{"cmd":"ls -la"}` {
		t.Fatalf("function_call=%v", fc)
	}
}

// ---------------------------------------------------------------------------
// 10. 思维链（reasoning item）
// ---------------------------------------------------------------------------

// respReasoningSSE：先思维链后正文，末帧 stop（上游 deepseek 系真实形态）。
const respReasoningSSE = `data: {"id":"chatcmpl-rs","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"先算"}}]}

data: {"id":"chatcmpl-rs","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"reasoning_content":"再验算"}}]}

data: {"id":"chatcmpl-rs","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"content":"答案是 4"}}]}

data: {"id":"chatcmpl-rs","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":6,"total_tokens":15}}

data: [DONE]

`

// TestResponsesStreamReasoning 锁定：delta.reasoning_content → 独立 reasoning item，
// 事件序列为 summary_part.added → text.delta×N → text.done → part.done → item.done，
// 且 reasoning item 的 output_index 必须小于 message item（Responses 顺序约束）。
func TestResponsesStreamReasoning(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respReasoningSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"deepseek-v4.1-flash","stream":true,"input":"2+2"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())

	want := []string{
		"response.created", "response.in_progress",
		"response.output_item.added", "response.reasoning_summary_part.added",
		"response.reasoning_summary_text.delta", "response.reasoning_summary_text.delta",
		"response.reasoning_summary_text.done", "response.reasoning_summary_part.done",
		"response.output_item.done",
		"response.output_item.added", "response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done", "response.content_part.done",
		"response.output_item.done", "response.completed",
	}
	got := eventNames(events)
	if len(got) != len(want) {
		t.Fatalf("events=%v\nwant %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event[%d]=%q want %q (all=%v)", i, got[i], want[i], got)
		}
	}

	// summary_index 恒为 0；sequence_number 单调。
	var reasoning strings.Builder
	for i, e := range events {
		if n, ok := e.data["sequence_number"].(float64); !ok || int(n) != i+1 {
			t.Fatalf("event %s sequence_number=%v want %d", e.name, e.data["sequence_number"], i+1)
		}
		if strings.HasPrefix(e.name, "response.reasoning_summary") {
			if n, ok := e.data["summary_index"].(float64); !ok || n != 0 {
				t.Fatalf("event %s summary_index=%v want 0", e.name, e.data["summary_index"])
			}
		}
		if e.name == "response.reasoning_summary_text.delta" {
			reasoning.WriteString(e.data["delta"].(string))
		}
	}
	if reasoning.String() != "先算再验算" {
		t.Fatalf("concatenated reasoning=%q want 先算再验算", reasoning.String())
	}

	completed := eventsNamed(events, "response.completed")[0].data
	resp := completed["response"].(map[string]any)
	output := resp["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output=%v want [reasoning, message]", output)
	}
	rs := output[0].(map[string]any)
	if rs["type"] != "reasoning" || rs["status"] != "completed" {
		t.Fatalf("reasoning item=%v", rs)
	}
	if s := rs["summary"].([]any)[0].(map[string]any); s["type"] != "summary_text" || s["text"] != "先算再验算" {
		t.Fatalf("reasoning summary=%v", s)
	}
	msg := output[1].(map[string]any)
	if msg["type"] != "message" {
		t.Fatalf("second item=%v want message", msg)
	}
	if c := msg["content"].([]any)[0].(map[string]any); c["text"] != "答案是 4" {
		t.Fatalf("message content=%v", c)
	}

	// reasoning 的 output_index 必须小于 message 的。
	added := eventsNamed(events, "response.output_item.added")
	if len(added) != 2 {
		t.Fatalf("output_item.added=%v", added)
	}
	ri := added[0].data["output_index"].(float64)
	mi := added[1].data["output_index"].(float64)
	if !(ri < mi) {
		t.Fatalf("reasoning output_index=%v must be < message output_index=%v", ri, mi)
	}
}

// TestResponsesNonStreamReasoning 非流式：message.reasoning_content → reasoning item。
func TestResponsesNonStreamReasoning(t *testing.T) {
	up, _ := staticRegressionUpstream(200, respReasoningSSE, false)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"deepseek-v4.1-flash","input":"2+2"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	output := obj["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output=%v want [reasoning, message]", output)
	}
	rs := output[0].(map[string]any)
	if rs["type"] != "reasoning" {
		t.Fatalf("first item=%v want reasoning", rs)
	}
	if s := rs["summary"].([]any)[0].(map[string]any); s["text"] != "先算再验算" {
		t.Fatalf("reasoning summary=%v", s)
	}
	if msg := output[1].(map[string]any); msg["type"] != "message" {
		t.Fatalf("second item=%v want message", msg)
	}
}

// TestResponsesReasoningOnlyNoEmptyMessage 纯思维链回合（无正文无工具）不补空
// message item，也不退回空流兜底分支。
func TestResponsesReasoningOnlyNoEmptyMessage(t *testing.T) {
	const sse = `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"只有思考"}}]}

data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"m","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	for _, e := range events {
		if e.name == "response.output_text.delta" {
			t.Fatalf("reasoning-only turn must not emit output_text: %v", eventNames(events))
		}
	}
	completed := eventsNamed(events, "response.completed")
	if len(completed) != 1 {
		t.Fatalf("events=%v", eventNames(events))
	}
	output := completed[0].data["response"].(map[string]any)["output"].([]any)
	if len(output) != 1 || output[0].(map[string]any)["type"] != "reasoning" {
		t.Fatalf("output=%v want single reasoning item", output)
	}
}
