// responses_codex_test.go —— Codex 客户端适配的行为测试。
//
// 覆盖：11115 超限中段裁剪重试（prompt_trim.go）、web_search 服务端工具接入
// responses 路径（含 TTL 缓存的 fresh 语义与 web_search_call item 事件）。
// 复用 chat_regression_test.go / messages_websearch_test.go 的夹具。
package server

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/websearch"
)

// buildCodexHistory 构造 Codex 长会话形态的归一化 body：system 基础指令 + 首条
// user（环境上下文）+ N 组 assistant(tool_calls)/tool 往返 + 尾部新 user 消息。
func buildCodexHistory(n int) []byte {
	var sb strings.Builder
	sb.WriteString(`{"model":"glm-5.2","stream":true,"messages":[`)
	sb.WriteString(`{"role":"system","content":"base instructions"},{"role":"user","content":"env context"}`)
	for i := 0; i < n; i++ {
		sb.WriteString(fmt.Sprintf(`,{"role":"assistant","tool_calls":[{"id":"c%d","type":"function","function":{"name":"exec","arguments":"{\"cmd\":\"%d\"}"}}]}`, i, i))
		sb.WriteString(fmt.Sprintf(`,{"role":"tool","tool_call_id":"c%d","content":"output %d padding padding padding padding padding"}`, i, i))
	}
	sb.WriteString(`,{"role":"user","content":"continue the task"}]}`)
	return []byte(sb.String())
}

func TestPromptTrimBody(t *testing.T) {
	t.Run("elides_middle_keeps_head_and_tail", func(t *testing.T) {
		body := buildCodexHistory(40)
		trimmed, ok := trimPromptBody(body, 1_103_110, 1_048_576)
		if !ok {
			t.Fatal("trim must succeed for a 40-round history")
		}
		msgs := outboundMessages(t, trimmed)
		if msgs[0]["role"] != "system" || msgs[0]["content"] != "base instructions" {
			t.Fatalf("head system must be preserved, got %v", msgs[0])
		}
		if msgs[1]["role"] != "user" || msgs[1]["content"] != "env context" {
			t.Fatalf("first user message must be preserved, got %v", msgs[1])
		}
		last := msgs[len(msgs)-1]
		if last["role"] != "user" || last["content"] != "continue the task" {
			t.Fatalf("tail user message must be preserved, got %v", last)
		}
		// 占位说明存在且为 user 角色。
		var notice bool
		for _, m := range msgs {
			if c, _ := m["content"].(string); strings.Contains(c, "elided by the gateway") {
				if m["role"] != "user" {
					t.Fatalf("notice role=%v want user", m["role"])
				}
				notice = true
			}
		}
		if !notice {
			t.Fatal("elision notice missing")
		}
		// 中段确实被裁掉：消息数显著减少。
		if want := len(outboundMessages(t, body)); len(msgs) >= want-8 {
			t.Fatalf("trimmed len=%d original=%d (must elide a meaningful middle)", len(msgs), want)
		}
		// 尾部保留段连续完整（tool_call 与其输出成对出现）。
		if len(msgs) < 6 {
			t.Fatalf("tail too short: %d", len(msgs))
		}
	})

	t.Run("short_history_refuses", func(t *testing.T) {
		body := []byte(`{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","content":"v"}]}`)
		if _, ok := trimPromptBody(body, 100, 50); ok {
			t.Fatal("short history must not be trimmed")
		}
	})

	t.Run("overflow_message_parsing", func(t *testing.T) {
		body := buildCodexHistory(30)
		trimmed, ok := trimPromptOverflow(body,
			`{"code":11115,"msg":"prompt is too long: 1103110 tokens > 1048576 maximum"}`)
		if !ok || trimmed == nil {
			t.Fatal("trimPromptOverflow must parse the upstream message and trim")
		}
		if _, ok := trimPromptOverflow([]byte(`{not json`), "anything"); ok {
			t.Fatal("bad body json must refuse")
		}
	})
}

func TestWebSearchCache(t *testing.T) {
	c := newWebSearchCache(20*time.Millisecond, 2)
	if _, ok := c.get("q"); ok {
		t.Fatal("empty cache must miss")
	}
	c.put("q", nil, "boom")
	e, ok := c.get("q")
	if !ok || e.err != "boom" {
		t.Fatalf("entry=%v ok=%v want cached error", e, ok)
	}
	time.Sleep(25 * time.Millisecond)
	if _, ok := c.get("q"); ok {
		t.Fatal("expired entry must miss")
	}
	// 容量上限：3 个键超过 maxKeys=2，整表淘汰后仍可写入。
	c.put("a", nil, "")
	c.put("b", nil, "")
	c.put("c", nil, "")
	if _, ok := c.get("a"); ok {
		t.Fatal("evicted key must miss")
	}
	if _, ok := c.get("c"); !ok {
		t.Fatal("latest key must hit")
	}
}

// sseSearchToolCall：模型调用 web_search 合成工具的流（聚合后含 tool_calls）。
const sseSearchToolCall = `data: {"id":"chatcmpl-ws","object":"chat.completion.chunk","created":1753600000,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_ws1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"golang 1.26 release notes\"}"}}]}}]}

data: {"id":"chatcmpl-ws","object":"chat.completion.chunk","created":1753600000,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":50,"completion_tokens":20,"total_tokens":70}}

data: [DONE]

`

// countingSearxngClient 起一个计数命中的假 SearXNG，返回 (client, 命中数取值函数)。
func countingSearxngClient(t *testing.T, url string) (*websearch.Client, func() int) {
	t.Helper()
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"url":"` + url + `","title":"T","content":"snippet"}]}`))
	}))
	t.Cleanup(srv.Close)
	return websearch.NewClient(websearch.Config{
		Enabled: true, Backend: "searxng", BaseURL: srv.URL, MaxResults: 5,
	}), func() int { return int(atomic.LoadInt32(&hits)) }
}

// TestResponsesModelInvokedSearch：模型主动调用模式——模型调 web_search →
// 网关检索回填 → 重发上游 → 客户端看到 web_search_call item + 最终回答。
func TestResponsesModelInvokedSearch(t *testing.T) {
	up, calls := newRegressionUpstream(func(n int, _ string, body []byte) (int, string, bool) {
		if n == 1 {
			return 200, sseSearchToolCall, true
		}
		return 200, respTextSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	ws, hits := countingSearxngClient(t, "https://a.example/1")
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: ws})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
		`{"model":"glm-5.2","stream":true,"tools":[{"type":"web_search"}],"input":"golang 1.26 发布了什么"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	var sawSearch, sawMessage bool
	for _, e := range eventsNamed(events, "response.output_item.added") {
		item, _ := e.data["item"].(map[string]any)
		switch item["type"] {
		case "web_search_call":
			sawSearch = true
			action, _ := item["action"].(map[string]any)
			if action["query"] != "golang 1.26 release notes" {
				t.Fatalf("web_search_call action=%v", action)
			}
		case "message":
			sawMessage = true
		}
	}
	if !sawSearch || !sawMessage {
		t.Fatalf("sawSearch=%v sawMessage=%v events=%v", sawSearch, sawMessage, eventNames(events))
	}
	if lastEventNamed(t, rec.Body.String(), "response.completed") == nil {
		t.Fatal("stream must complete")
	}
	// 引擎只打一次；上游两轮：第一轮带合成工具，第二轮带检索结果。
	if n := hits(); n != 1 {
		t.Fatalf("searxng hits=%d want 1", n)
	}
	cs := calls()
	if len(cs) != 2 {
		t.Fatalf("upstream calls=%d want 2", len(cs))
	}
	if !strings.Contains(string(cs[0].body), `"web_search"`) {
		t.Fatalf("round1 body must declare the synthetic tool: %s", cs[0].body)
	}
	if !strings.Contains(string(cs[1].body), "https://a.example/1") {
		t.Fatalf("round2 body must carry search results: %s", cs[1].body)
	}
}

// TestResponsesModelInvokedSearchNoSearchNeeded：模型直接回答（编码任务常态）——
// 不触发任何检索（旧预检索模式对整段任务书搜索的浪费点），合成工具已声明。
func TestResponsesModelInvokedSearchNoSearchNeeded(t *testing.T) {
	up, calls := newRegressionUpstream(func(int, string, []byte) (int, string, bool) {
		return 200, respTextSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	ws, hits := countingSearxngClient(t, "https://a.example/1")
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: ws})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
		`{"model":"glm-5.2","stream":true,"tools":[{"type":"web_search"}],"input":"在本地加一个搜索统计视图"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	for _, e := range parseSSEEvents(t, rec.Body.String()) {
		if e.name != "response.output_item.added" && e.name != "response.output_item.done" {
			continue
		}
		if item, _ := e.data["item"].(map[string]any); item["type"] == "web_search_call" {
			t.Fatal("no search ran; web_search_call item must not appear")
		}
	}
	if n := hits(); n != 0 {
		t.Fatalf("searxng hits=%d want 0 (model did not ask to search)", n)
	}
	cs := calls()
	if len(cs) != 1 {
		t.Fatalf("upstream calls=%d want 1", len(cs))
	}
	if !strings.Contains(string(cs[0].body), `"web_search"`) {
		t.Fatalf("body must declare the synthetic tool so the model CAN search: %s", cs[0].body)
	}
}

// TestResponsesPromptTooLongTrimRetry：上游首个 11115 → 中段裁剪 → 同请求内重试成功。
func TestResponsesPromptTooLongTrimRetry(t *testing.T) {
	up, calls := newRegressionUpstream(func(n int, _ string, body []byte) (int, string, bool) {
		if n == 1 {
			return 400, `{"code":11115,"msg":"prompt is too long: 1103110 tokens > 1048576 maximum"}`, false
		}
		return 200, respTextSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(string(buildCodexResponsesBody(40)))))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s (trim retry must succeed)", rec.Code, rec.Body)
	}
	cs := calls()
	if len(cs) != 2 {
		t.Fatalf("upstream calls=%d want 2 (original + trimmed retry)", len(cs))
	}
	first := outboundMessages(t, cs[0].body)
	second := outboundMessages(t, cs[1].body)
	if len(second) >= len(first) {
		t.Fatalf("retry body must be trimmed: %d vs %d", len(second), len(first))
	}
	if !strings.Contains(rec.Body.String(), "event: response.completed") {
		t.Fatalf("stream must complete, body=%s", rec.Body)
	}
}

// buildCodexResponsesBody 构造带 40 组工具往返的 /v1/responses 请求体。
func buildCodexResponsesBody(n int) string {
	var sb strings.Builder
	sb.WriteString(`{"model":"glm-5.2","stream":true,"input":[{"type":"message","role":"user","content":"env context"}`)
	for i := 0; i < n; i++ {
		sb.WriteString(fmt.Sprintf(`,{"type":"function_call","call_id":"c%d","name":"exec","arguments":"{\"cmd\":\"%d\"}"}`, i, i))
		sb.WriteString(fmt.Sprintf(`,{"type":"function_call_output","call_id":"c%d","output":"output %d padding padding padding"}`, i, i))
	}
	sb.WriteString(`,{"type":"message","role":"user","content":"continue"}]}`)
	return sb.String()
}

// ssePendingNarration：「现在检查…」播报收尾 + 无工具调用 + stop——end_turn:false
// 强制续跑的触发形态。
const ssePendingNarration = `data: {"id":"chatcmpl-pending","object":"chat.completion.chunk","created":1753600000,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"定向测试已通过，现在检查全量测试。"}}]}

data: {"id":"chatcmpl-pending","object":"chat.completion.chunk","created":1753600000,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":12,"total_tokens":112}}

data: [DONE]

`

// TestResponsesEndTurnForcedFollowUp：播报收尾且无工具调用 → response.completed
// 附 end_turn:false（Codex 自动 follow-up）；历史末尾已有 2 条连续 assistant
// 消息（防抖上限）→ 不附。
func TestResponsesEndTurnForcedFollowUp(t *testing.T) {
	up, _ := staticRegressionUpstream(200, ssePendingNarration, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	freshInput := `{"model":"glm-5.2","stream":true,"input":[{"type":"message","role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(freshInput)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	completed := lastEventNamed(t, rec.Body.String(), "response.completed")
	resp := completed["response"].(map[string]any)
	if resp["end_turn"] != false {
		t.Fatalf("end_turn=%v want false on pending narration", resp["end_turn"])
	}

	saturatedInput := `{"model":"glm-5.2","stream":true,"input":[` +
		`{"type":"message","role":"user","content":"hi"},` +
		`{"type":"message","role":"assistant","content":"现在检查一下。"},` +
		`{"type":"message","role":"assistant","content":"现在运行测试。"}]}`
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(saturatedInput)))
	if rec2.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec2.Code, rec2.Body)
	}
	completed2 := lastEventNamed(t, rec2.Body.String(), "response.completed")
	resp2 := completed2["response"].(map[string]any)
	if _, has := resp2["end_turn"]; has {
		t.Fatalf("end_turn must be absent when trailing assistant msgs >= %d", maxForcedFollowUps)
	}

	// 归一化侧：末尾连续 assistant 计数正确（中间夹 user 消息即中断）。
	_, meta, err := normalizeResponsesRequest([]byte(saturatedInput))
	if err != nil {
		t.Fatalf("normalize: %+v", err)
	}
	if meta.trailingAssistantMsgs != 2 {
		t.Fatalf("trailingAssistantMsgs=%d want 2", meta.trailingAssistantMsgs)
	}
	mixed := `{"model":"m","input":[{"type":"message","role":"assistant","content":"a"},{"type":"message","role":"user","content":"u"},{"type":"message","role":"assistant","content":"b"}]}`
	_, meta2, _ := normalizeResponsesRequest([]byte(mixed))
	if meta2.trailingAssistantMsgs != 1 {
		t.Fatalf("trailingAssistantMsgs=%d want 1 (user breaks streak)", meta2.trailingAssistantMsgs)
	}
}

// lastEventNamed 取最后一个指定名称的 SSE 事件 data。
func lastEventNamed(t *testing.T, body, name string) map[string]any {
	t.Helper()
	var last map[string]any
	for _, e := range parseSSEEvents(t, body) {
		if e.name == name {
			last = e.data
		}
	}
	if last == nil {
		t.Fatalf("no %s event in stream", name)
	}
	return last
}

// TestResponsesSearchBudgetExhaustedNoLeak：模型每轮都想搜（耗尽降级路径）——
// 降级后的最终响应里绝不能出现 web_search function_call（Codex 无法执行）。
func TestResponsesSearchBudgetExhaustedNoLeak(t *testing.T) {
	up, calls := newRegressionUpstream(func(int, string, []byte) (int, string, bool) {
		return 200, sseSearchToolCall, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	ws, hits := countingSearxngClient(t, "https://a.example/1")
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: ws})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
		`{"model":"glm-5.2","stream":true,"tools":[{"type":"web_search"}],"input":"再搜点别的"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	for _, e := range parseSSEEvents(t, rec.Body.String()) {
		if e.name != "response.output_item.added" && e.name != "response.output_item.done" {
			continue
		}
		item, _ := e.data["item"].(map[string]any)
		if item["type"] == "function_call" {
			t.Fatalf("web_search function_call leaked to client: %v", item)
		}
	}
	// 上游调用数 = 搜索轮（maxSearchRounds+1）+ 降级 runTurn 一跳；引擎命中被 TTL 缓存去重为 1。
	if n := len(calls()); n < maxSearchRounds+1 {
		t.Fatalf("upstream calls=%d want >= %d", n, maxSearchRounds+1)
	}
	if n := hits(); n != 1 {
		t.Fatalf("searxng hits=%d want 1 (TTL dedup)", n)
	}
	// 降级 body 带引导 system 提示。
	lastBody := string(calls()[len(calls())-1].body)
	if !strings.Contains(lastBody, "no longer available") {
		t.Fatalf("fallback body must carry steering note")
	}
}
