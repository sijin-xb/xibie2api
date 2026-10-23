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
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
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

// TestResponsesWebSearchStream：声明 web_search 工具的 /v1/responses 请求——
// 首次：检索执行（fresh），SSE 先发 web_search_call item，检索结果注入出站 prompt；
// 同查询第二次（TTL 缓存命中）：不再发 web_search_call item，但结果照常注入。
func TestResponsesWebSearchStream(t *testing.T) {
	up, calls := newRegressionUpstream(func(int, string, []byte) (int, string, bool) {
		return 200, respTextSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, WebSearch: newSearxngTestClient(t, "https://a.example/1")})

	reqBody := `{"model":"glm-5.2","stream":true,"tools":[{"type":"web_search"}],"input":"golang 1.24 release notes"}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(reqBody)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	added := eventsNamed(events, "response.output_item.added")
	if len(added) == 0 {
		t.Fatalf("no output_item.added events: %v", eventNames(events))
	}
	wsItem, _ := added[0].data["item"].(map[string]any)
	if wsItem["type"] != "web_search_call" {
		t.Fatalf("first item=%v want web_search_call", wsItem)
	}
	// done 事件带 action.query。
	for _, e := range eventsNamed(events, "response.output_item.done") {
		if item, _ := e.data["item"].(map[string]any); item["type"] == "web_search_call" {
			action, _ := item["action"].(map[string]any)
			if action["type"] != "search" || !strings.Contains(action["query"].(string), "golang") {
				t.Fatalf("web_search_call action=%v", action)
			}
		}
	}
	// 检索结果注入出站 prompt。
	cs := calls()
	if len(cs) != 1 {
		t.Fatalf("upstream calls=%d", len(cs))
	}
	if !strings.Contains(string(cs[0].body), "https://a.example/1") {
		t.Fatalf("search results not injected into outbound body: %s", cs[0].body)
	}

	// 第二次同查询：缓存命中 → 无 web_search_call item，注入照常。
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(reqBody)))
	if rec2.Code != http.StatusOK {
		t.Fatalf("second code=%d body=%s", rec2.Code, rec2.Body)
	}
	for _, e := range parseSSEEvents(t, rec2.Body.String()) {
		if e.name != "response.output_item.added" && e.name != "response.output_item.done" {
			continue
		}
		if item, _ := e.data["item"].(map[string]any); item["type"] == "web_search_call" {
			t.Fatal("cached search must not re-emit web_search_call item")
		}
	}
	cs2 := calls()
	if len(cs2) != 2 || !strings.Contains(string(cs2[1].body), "https://a.example/1") {
		t.Fatalf("cached results must still be injected (calls=%d)", len(cs2))
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
