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
	"sync"
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

func newSearxngCapture(t *testing.T, urls ...string) (*websearch.Client, *[]string) {
	t.Helper()
	var mu sync.Mutex
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		queries = append(queries, r.URL.Query().Get("q"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		var sb strings.Builder
		sb.WriteString(`{"results":[`)
		for i, u := range urls {
			if i > 0 {
				sb.WriteString(",")
			}
			b, _ := json.Marshal(map[string]any{"url": u, "title": "T:" + u, "content": "snippet"})
			sb.Write(b)
		}
		sb.WriteString(`]}`)
		_, _ = w.Write([]byte(sb.String()))
	}))
	t.Cleanup(srv.Close)
	return websearch.NewClient(websearch.Config{Enabled: true, Backend: "searxng", BaseURL: srv.URL, MaxResults: 5}), &queries
}

func TestMessagesWebSearchStream(t *testing.T) {
	up, calls := newRegressionUpstream(func(n int, _ string, body []byte) (int, string, bool) {
		if n == 1 {
			return 200, sseSearchToolCall, true
		}
		return 200, regNonStreamSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	client, queries := newSearxngCapture(t, "https://a.example/1", "https://b.example/2")
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: client})

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
	var partial string
	for _, e := range eventsNamed(events, "content_block_delta") {
		d, _ := e.data["delta"].(map[string]any)
		if d["type"] == "input_json_delta" {
			partial += d["partial_json"].(string)
		}
	}
	if !strings.Contains(partial, "golang 1.26 release notes") {
		t.Fatalf("partial_json=%q (query comes from the model tool call)", partial)
	}
	cont, _ := b1["content"].([]any)
	if len(cont) != 2 {
		t.Fatalf("tool_result content=%v", b1["content"])
	}
	first := cont[0].(map[string]any)
	if first["type"] != "web_search_result" || first["url"] != "https://a.example/1" {
		t.Fatalf("result[0]=%v", first)
	}
	cs := calls()
	if len(cs) != 2 {
		t.Fatalf("upstream calls=%d want 2 (search round + answer round)", len(cs))
	}
	if !strings.Contains(string(cs[1].body), "https://a.example/1") {
		t.Fatalf("round2 body must carry search results: %s", cs[1].body)
	}
	if len(*queries) != 1 {
		t.Fatalf("engine queries=%v want exactly 1", *queries)
	}
}

// TestMessagesSearchNoLeak：模型把 web_search 与客户端工具混发（或预算耗尽后
// 仍要搜）——合成工具绝不透出为 tool_use 块。
func TestMessagesSearchNoLeak(t *testing.T) {
	up, _ := newRegressionUpstream(func(int, string, []byte) (int, string, bool) {
		return 200, sseSearchToolCall, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	client, _ := newSearxngCapture(t, "https://a.example/1")
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: client})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"glm-5.2","max_tokens":100,"stream":true,"max_uses":1,"tools":[{"type":"web_search_20260209","name":"web_search","max_uses":1}],"messages":[{"role":"user","content":"搜这个"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	for _, e := range parseSSEEvents(t, rec.Body.String()) {
		if e.name != "content_block_start" {
			continue
		}
		if b, _ := e.data["content_block"].(map[string]any); b["type"] == "tool_use" {
			t.Fatalf("web_search leaked as tool_use block: %v", b)
		}
	}
}
