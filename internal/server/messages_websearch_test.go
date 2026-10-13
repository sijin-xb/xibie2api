// messages_websearch_test.go —— /v1/messages 的 web_search 服务端工具端到端测试。
//
// 覆盖：归一化剥离服务端工具并提取查询串；流式补发 server_tool_use +
// web_search_tool_result 内容块（索引单调、query 出现在 input_json_delta）；
// 非流式前置两个内容块；检索结果注入出站 prompt；网关未启用时回
// web_search_tool_result_error。复用 handler_test.go / chat_regression_test.go
// 的夹具（newRegressionUpstream / staticRegressionUpstream / testPoolWith /
// parseSSEEvents / mustDecodeJSON）。
package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/websearch"
)

// newSearxngTestClient 起一个返回固定结果的假 SearXNG，返回指向它的检索客户端。
func newSearxngTestClient(t *testing.T, urls ...string) *websearch.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		var sb strings.Builder
		sb.WriteString(`{"results":[`)
		for i, u := range urls {
			if i > 0 {
				sb.WriteString(",")
			}
			b, _ := json.Marshal(map[string]any{"url": u, "title": "T:" + u, "content": "snippet for " + u})
			sb.Write(b)
		}
		sb.WriteString(`]}`)
		_, _ = w.Write([]byte(sb.String()))
	}))
	t.Cleanup(srv.Close)
	return websearch.NewClient(websearch.Config{Enabled: true, Backend: "searxng", BaseURL: srv.URL, MaxResults: 5})
}

func TestNormalizeMessagesWebSearchTool(t *testing.T) {
	body, meta, err := normalizeMessagesRequest([]byte(`{
		"model":"glm-5.2","max_tokens":100,
		"tools":[
			{"type":"web_search_20260209","name":"web_search","max_uses":8},
			{"name":"get_weather","description":"d","input_schema":{"type":"object"}}
		],
		"messages":[{"role":"user","content":"北京今天天气怎么样"}]}`))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if meta.webSearch == nil {
		t.Fatal("webSearch is nil; server tool not detected")
	}
	if meta.webSearch.query != "北京今天天气怎么样" {
		t.Fatalf("query=%q", meta.webSearch.query)
	}
	if meta.webSearch.maxUses != 8 {
		t.Fatalf("maxUses=%d", meta.webSearch.maxUses)
	}
	obj := mustDecodeJSON(t, body)
	tools, _ := obj["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("server tool leaked into function tools: %v", tools)
	}
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Fatalf("remaining tool=%v", fn)
	}
}

func TestNormalizeMessagesWebSearchOnlyTool(t *testing.T) {
	body, meta, err := normalizeMessagesRequest([]byte(`{
		"model":"glm-5.2","max_tokens":100,
		"tools":[{"type":"web_search_20250305","name":"web_search"}],
		"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("unexpected error: %+v", err)
	}
	if meta.webSearch == nil {
		t.Fatal("webSearch is nil")
	}
	obj := mustDecodeJSON(t, body)
	if _, ok := obj["tools"]; ok {
		t.Fatalf("tools should be absent when only server tool declared: %v", obj["tools"])
	}
}

func TestMessagesWebSearchStream(t *testing.T) {
	up, calls := newRegressionUpstream(func(int, string, []byte) (int, string, bool) {
		return 200, regNonStreamSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: newSearxngTestClient(t, "https://a.example/1", "https://b.example/2")})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"glm-5.2","max_tokens":100,"stream":true,"tools":[{"type":"web_search_20260209","name":"web_search"}],"messages":[{"role":"user","content":"golang 1.24 新特性"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	starts := eventsNamed(events, "content_block_start")
	if len(starts) < 3 {
		t.Fatalf("content_block_start count=%d events=%v", len(starts), eventNames(events))
	}
	// 前两块：server_tool_use + web_search_tool_result；索引单调 0,1,2。
	b0 := starts[0].data["content_block"].(map[string]any)
	if b0["type"] != "server_tool_use" || b0["name"] != "web_search" {
		t.Fatalf("block0=%v", b0)
	}
	b1 := starts[1].data["content_block"].(map[string]any)
	if b1["type"] != "web_search_tool_result" {
		t.Fatalf("block1=%v", b1)
	}
	b2 := starts[2].data["content_block"].(map[string]any)
	if b2["type"] != "text" {
		t.Fatalf("block2=%v", b2)
	}
	for i, want := range []float64{0, 1, 2} {
		if got := starts[i].data["index"].(float64); got != want {
			t.Fatalf("index[%d]=%v want %v", i, got, want)
		}
	}
	// server_tool_use 的 query 经 input_json_delta 下发。
	var partial string
	for _, e := range eventsNamed(events, "content_block_delta") {
		d, _ := e.data["delta"].(map[string]any)
		if d["type"] == "input_json_delta" {
			partial += d["partial_json"].(string)
		}
	}
	if !strings.Contains(partial, "golang 1.24 新特性") {
		t.Fatalf("partial_json=%q", partial)
	}
	// 结果内容为两条 web_search_result。
	cont, _ := b1["content"].([]any)
	if len(cont) != 2 {
		t.Fatalf("tool_result content=%v", b1["content"])
	}
	first := cont[0].(map[string]any)
	if first["type"] != "web_search_result" || first["url"] != "https://a.example/1" {
		t.Fatalf("result[0]=%v", first)
	}
	// 检索结果注入出站 prompt（system 消息含结果 URL）。
	cs := calls()
	if len(cs) != 1 {
		t.Fatalf("upstream calls=%d", len(cs))
	}
	injected := false
	for _, m := range outboundMessages(t, cs[0].body) {
		if m["role"] == "system" {
			if s, _ := m["content"].(string); strings.Contains(s, "https://a.example/1") {
				injected = true
			}
		}
	}
	if !injected {
		t.Fatalf("search context not injected into outbound body: %s", cs[0].body)
	}
	// 服务端工具未被当成 function 转发。
	if obj := mustDecodeJSON(t, cs[0].body); obj["tools"] != nil {
		t.Fatalf("server tool leaked as function: %v", obj["tools"])
	}
}

// newSearxngCapture 同 newSearxngTestClient，但记录每次检索收到的 query。
func newSearxngCapture(t *testing.T, urls ...string) (*websearch.Client, *[]string) {
	t.Helper()
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		queries = append(queries, r.URL.Query().Get("q"))
		w.Header().Set("Content-Type", "application/json")
		var sb strings.Builder
		sb.WriteString(`{"results":[`)
		for i, u := range urls {
			if i > 0 {
				sb.WriteString(",")
			}
			b, _ := json.Marshal(map[string]any{"url": u, "title": "T:" + u, "content": "s"})
			sb.Write(b)
		}
		sb.WriteString(`]}`)
		_, _ = w.Write([]byte(sb.String()))
	}))
	t.Cleanup(srv.Close)
	return websearch.NewClient(websearch.Config{Enabled: true, Backend: "searxng", BaseURL: srv.URL, MaxResults: 5}), &queries
}

func TestMessagesWebSearchModelQuery(t *testing.T) {
	const querySSE = "data: {\"choices\":[{\"delta\":{\"content\":\"Go 1.24 release notes\"}}]}\n\ndata: [DONE]\n\n"
	up, calls := newRegressionUpstream(func(n int, _ string, _ []byte) (int, string, bool) {
		if n == 1 {
			return 200, querySSE, true // 第一次调用 = 查询提炼
		}
		return 200, regNonStreamSSE, true // 第二次 = 正式回答
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	client, queries := newSearxngCapture(t, "https://go.dev/doc/go1.24")
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: client, WebSearchQueryMode: "model"})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"glm-5.2","max_tokens":100,"tools":[{"type":"web_search_20260209","name":"web_search"}],"messages":[{"role":"user","content":"Go 1.24 有哪些新特性？"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(*queries) != 1 || (*queries)[0] != "Go 1.24 release notes" {
		t.Fatalf("search queries=%v want [Go 1.24 release notes]", *queries)
	}
	if len(calls()) != 2 {
		t.Fatalf("upstream calls=%d want 2 (query-gen + answer)", len(calls()))
	}
	// server_tool_use.input.query 应是提炼后的查询串。
	resp := mustDecodeJSON(t, rec.Body.Bytes())
	content, _ := resp["content"].([]any)
	stu := content[0].(map[string]any)
	if q := stu["input"].(map[string]any)["query"]; q != "Go 1.24 release notes" {
		t.Fatalf("server_tool_use input=%v", stu["input"])
	}
}

func TestMessagesWebSearchNonStream(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regNonStreamSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: newSearxngTestClient(t, "https://a.example/1")})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"glm-5.2","max_tokens":100,"tools":[{"type":"web_search_20260209","name":"web_search"}],"messages":[{"role":"user","content":"hello"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	resp := mustDecodeJSON(t, rec.Body.Bytes())
	content, _ := resp["content"].([]any)
	if len(content) != 3 {
		t.Fatalf("content=%v", resp["content"])
	}
	if content[0].(map[string]any)["type"] != "server_tool_use" {
		t.Fatalf("content[0]=%v", content[0])
	}
	c1 := content[1].(map[string]any)
	if c1["type"] != "web_search_tool_result" {
		t.Fatalf("content[1]=%v", c1)
	}
	if content[2].(map[string]any)["type"] != "text" {
		t.Fatalf("content[2]=%v", content[2])
	}
}

func TestMessagesWebSearchDisabled(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regNonStreamSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up}) // 未配置 WebSearch

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"glm-5.2","max_tokens":100,"tools":[{"type":"web_search_20260209","name":"web_search"}],"messages":[{"role":"user","content":"hello"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	resp := mustDecodeJSON(t, rec.Body.Bytes())
	content, _ := resp["content"].([]any)
	if len(content) != 3 {
		t.Fatalf("content=%v", resp["content"])
	}
	c1 := content[1].(map[string]any)
	if c1["type"] != "web_search_tool_result" {
		t.Fatalf("content[1]=%v", c1)
	}
	errBlock, _ := c1["content"].(map[string]any)
	if errBlock["type"] != "web_search_tool_result_error" {
		t.Fatalf("expected tool_result_error, got %v", c1["content"])
	}
}
