// responses_tools_test.go —— /v1/responses 工具形态适配测试（Codex 内置工具）。
//
// 覆盖：namespace 工具摊平、custom(freeform) 工具降级、web_search 跳过，
// 以及回程把上游 function_call 还原为 namespace / custom item。
package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// codexToolsRequest 构造一个带 Codex 四类工具的最小请求体。
const codexToolsRequest = `{"model":"glm-5.2","input":"hi","tools":[` +
	`{"type":"function","name":"exec_command","description":"run","parameters":{"type":"object"}},` +
	`{"type":"custom","name":"apply_patch","description":"edit files","format":{"type":"grammar","syntax":"lark","definition":"start: x"}},` +
	`{"type":"namespace","name":"multi_agent_v1","description":"sub-agents","tools":[` +
	`{"type":"function","name":"spawn_agent","description":"spawn","parameters":{"type":"object"}},` +
	`{"type":"function","name":"wait_agent","description":"wait","parameters":{"type":"object"}}]},` +
	`{"type":"namespace","name":"mcp__node_repl","description":"node repl","tools":[` +
	`{"type":"function","name":"js","description":"eval","parameters":{"type":"object"}}]},` +
	`{"type":"web_search","external_web_access":false,"search_content_types":["text","image"]}]}`

// outboundTools 解析出站 body 的 tools 数组（缺失返回 nil）。
func outboundTools(t *testing.T, body []byte) []map[string]any {
	t.Helper()
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		t.Fatalf("outbound body not JSON: %v", err)
	}
	raw, _ := obj["tools"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, t2 := range raw {
		m, _ := t2.(map[string]any)
		out = append(out, m)
	}
	return out
}

// TestResponsesToolsNamespaceAndCustomFlattened 断言 namespace 子工具被摊平成
// "<ns>__<tool>"、custom 工具降级为单参数 function、web_search 被跳过而非 400。
func TestResponsesToolsNamespaceAndCustomFlattened(t *testing.T) {
	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, regNonStreamSSE, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(codexToolsRequest)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) == 0 {
		t.Fatal("no outbound request captured")
	}
	tools := outboundTools(t, bodies[0])

	names := map[string]map[string]any{}
	for _, tl := range tools {
		if tl["type"] != "function" {
			t.Fatalf("outbound tool type=%v want function: %v", tl["type"], tl)
		}
		fn, _ := tl["function"].(map[string]any)
		name, _ := fn["name"].(string)
		names[name] = fn
	}

	for _, want := range []string{"exec_command", "apply_patch", "multi_agent_v1__spawn_agent", "multi_agent_v1__wait_agent", "mcp__node_repl__js"} {
		if _, ok := names[want]; !ok {
			t.Fatalf("missing flattened tool %q (got %v)", want, names)
		}
	}
	// namespace 子工具保留参数 schema，并带上命名空间描述。
	if _, ok := names["multi_agent_v1__spawn_agent"]["parameters"]; !ok {
		t.Fatalf("namespace sub-tool lost parameters: %v", names["multi_agent_v1__spawn_agent"])
	}
	// custom 工具降级为单字符串参数 input。
	ap := names["apply_patch"]
	params, _ := ap["parameters"].(map[string]any)
	props, _ := params["properties"].(map[string]any)
	if _, ok := props["input"]; !ok {
		t.Fatalf("custom tool not degraded to single input param: %v", ap)
	}
	// web_search 无法表达，必须跳过（不得出现同名 function）。
	if _, ok := names["web_search"]; ok {
		t.Fatalf("web_search should be skipped, got %v", names["web_search"])
	}
}

// TestResponsesNamespaceToolCallRendered 断言上游返回摊平名 function_call 时，
// 响应 item 还原为带 namespace 的 function_call（流式与非流式一致）。
func TestResponsesNamespaceToolCallRendered(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_ns","type":"function","function":{"name":"multi_agent_v1__spawn_agent","arguments":"{\"task\":\"x\"}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n" +
		"data: [DONE]\n\n"

	t.Run("stream", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sse, true)
		p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
		h := NewHandler(Config{Pool: p, Upstream: up})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
			strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi","tools":[`+
				`{"type":"namespace","name":"multi_agent_v1","tools":[`+
				`{"type":"function","name":"spawn_agent","parameters":{"type":"object"}}]}]}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
		}
		events := parseSSEEvents(t, rec.Body.String())
		added := eventsNamed(events, "response.output_item.added")
		if len(added) != 1 {
			t.Fatalf("output_item.added=%v", eventNames(events))
		}
		item := added[0].data["item"].(map[string]any)
		if item["namespace"] != "multi_agent_v1" || item["name"] != "spawn_agent" {
			t.Fatalf("namespace item=%v", item)
		}
		completed := eventsNamed(events, "response.completed")[0].data["response"].(map[string]any)
		done := completed["output"].([]any)[0].(map[string]any)
		if done["namespace"] != "multi_agent_v1" || done["name"] != "spawn_agent" || done["arguments"] != `{"task":"x"}` {
			t.Fatalf("completed namespace item=%v", done)
		}
	})

	t.Run("non_stream", func(t *testing.T) {
		up, _ := staticRegressionUpstream(200, sse, false)
		p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
		h := NewHandler(Config{Pool: p, Upstream: up})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(codexToolsRequest)))
		if rec.Code != http.StatusOK {
			t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
		}
		obj := mustDecodeJSON(t, rec.Body.Bytes())
		item := obj["output"].([]any)[0].(map[string]any)
		if item["namespace"] != "multi_agent_v1" || item["name"] != "spawn_agent" {
			t.Fatalf("non-stream namespace item=%v", item)
		}
	})
}

// TestResponsesCustomToolCallRendered 断言 custom 工具调用还原为 custom_tool_call，
// input 从上游单参数 JSON 中解出。
func TestResponsesCustomToolCallRendered(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_c","type":"function","function":{"name":"apply_patch","arguments":"{\"input\":\"*** Begin Patch\\n*** End Patch\"}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\n" +
		"data: [DONE]\n\n"

	up, _ := staticRegressionUpstream(200, sse, false)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(codexToolsRequest)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	item := obj["output"].([]any)[0].(map[string]any)
	if item["type"] != "custom_tool_call" || item["name"] != "apply_patch" {
		t.Fatalf("custom item=%v", item)
	}
	if item["input"] != "*** Begin Patch\n*** End Patch" {
		t.Fatalf("custom input=%q", item["input"])
	}
}

// TestResponsesCustomToolHistoryRoundTrip 断言历史里的 custom_tool_call /
// custom_tool_call_output 被归一化，且 namespace function_call 还原为摊平名。
func TestResponsesCustomToolHistoryRoundTrip(t *testing.T) {
	raw := `{"model":"glm-5.2","input":[` +
		`{"type":"custom_tool_call","call_id":"c1","name":"apply_patch","input":"PATCH"},` +
		`{"type":"custom_tool_call_output","call_id":"c1","output":"ok"},` +
		`{"type":"function_call","call_id":"c2","namespace":"multi_agent_v1","name":"spawn_agent","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"c2","output":"done"}]}`
	body, _, err := normalizeResponsesRequest([]byte(raw))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	msgs := mustDecodeJSON(t, body)["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("messages=%v want 4", msgs)
	}
	customCall := msgs[0].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if fn := customCall["function"].(map[string]any); fn["name"] != "apply_patch" || fn["arguments"] != `{"input":"PATCH"}` {
		t.Fatalf("custom call fn=%v", fn)
	}
	if tool := msgs[1].(map[string]any); tool["role"] != "tool" || tool["content"] != "ok" {
		t.Fatalf("custom output msg=%v", tool)
	}
	nsCall := msgs[2].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	if fn := nsCall["function"].(map[string]any); fn["name"] != "multi_agent_v1__spawn_agent" {
		t.Fatalf("namespace history call fn=%v", fn)
	}
}
