package server

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// messages_cache_usage_test.go 覆盖 Anthropic usage 的缓存字段拆分。
//
// 背景：ZCode 等客户端按 total = input_tokens + cache_read_input_tokens +
// cache_creation_input_tokens 计算上下文占用，命中率取 cache_read。此前 /v1/messages
// 只发 input_tokens(= 上游含缓存的 prompt_tokens) 与 output_tokens，导致：
//   - 命中率恒为 0（cache 字段缺失）
//   - 上下文占用偏高（把缓存命中部分也算作新输入，且无法扣除）
//
// 修复：input_tokens = prompt_tokens − cache_read − cache_write，并补发两个 cache 字段
// （上游未提供时省略，不伪造）。

// msgCacheSSE 上游 SSE：prompt=100 含 cache_read=60、cache_write=10，completion=5。
// 期望 input_tokens = 100−60−10 = 30。
const msgCacheSSE = `data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"cache_read_input_tokens":60,"cache_creation_input_tokens":10}}

data: [DONE]

`

// TestMessagesNonStreamUsageSplitsCache：非流式 usage 必须拆出 cache 两件套，
// 且 input_tokens 不含缓存部分。
func TestMessagesNonStreamUsageSplitsCache(t *testing.T) {
	up, _ := staticRegressionUpstream(200, msgCacheSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	usage := obj["usage"].(map[string]any)

	if got := usage["input_tokens"].(float64); got != 30 {
		t.Fatalf("input_tokens=%v want 30 (100-60-10)", got)
	}
	if got := usage["output_tokens"].(float64); got != 5 {
		t.Fatalf("output_tokens=%v want 5", got)
	}
	if got, ok := usage["cache_read_input_tokens"].(float64); !ok || got != 60 {
		t.Fatalf("cache_read_input_tokens=%v ok=%v want 60", usage["cache_read_input_tokens"], ok)
	}
	if got, ok := usage["cache_creation_input_tokens"].(float64); !ok || got != 10 {
		t.Fatalf("cache_creation_input_tokens=%v ok=%v want 10", usage["cache_creation_input_tokens"], ok)
	}
	// 客户端上下文占用 = 三件套之和，必须等于上游 prompt_tokens（不因拆分而改变总量）。
	total := usage["input_tokens"].(float64) +
		usage["cache_read_input_tokens"].(float64) +
		usage["cache_creation_input_tokens"].(float64)
	if total != 100 {
		t.Fatalf("context total=%v want 100 (== upstream prompt_tokens)", total)
	}
}

// TestMessagesNonStreamUsageNoCacheFields：上游未给缓存维度时，不伪造 0，
// input_tokens 等于完整 prompt_tokens。
func TestMessagesNonStreamUsageNoCacheFields(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regNonStreamSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))

	obj := mustDecodeJSON(t, rec.Body.Bytes())
	usage := obj["usage"].(map[string]any)
	if got := usage["input_tokens"].(float64); got != 7 {
		t.Fatalf("input_tokens=%v want 7", got)
	}
	if _, ok := usage["cache_read_input_tokens"]; ok {
		t.Fatalf("must not fabricate cache_read when upstream omits it: %v", usage)
	}
	if _, ok := usage["cache_creation_input_tokens"]; ok {
		t.Fatalf("must not fabricate cache_creation when upstream omits it: %v", usage)
	}
}

// TestMessagesStreamUsageSplitsCache：流式 message_delta 必须拆出 cache 两件套，
// 且 input_tokens 不含缓存部分——客户端靠这里更新上下文占用与命中率。
func TestMessagesStreamUsageSplitsCache(t *testing.T) {
	up, _ := staticRegressionUpstream(200, msgCacheSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != 200 {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	md := eventsNamed(events, "message_delta")
	if len(md) != 1 {
		t.Fatalf("message_delta count=%d want 1", len(md))
	}
	usage := md[0].data["usage"].(map[string]any)
	if got := usage["input_tokens"].(float64); got != 30 {
		t.Fatalf("input_tokens=%v want 30", got)
	}
	if got := usage["output_tokens"].(float64); got != 5 {
		t.Fatalf("output_tokens=%v want 5", got)
	}
	if got, ok := usage["cache_read_input_tokens"].(float64); !ok || got != 60 {
		t.Fatalf("cache_read_input_tokens=%v ok=%v want 60", usage["cache_read_input_tokens"], ok)
	}
	if got, ok := usage["cache_creation_input_tokens"].(float64); !ok || got != 10 {
		t.Fatalf("cache_creation_input_tokens=%v ok=%v want 10", usage["cache_creation_input_tokens"], ok)
	}
}

// TestMessagesUsageFieldsInconsistentUpstream：上游字段不自洽（命中+写入 > 总量）时
// 不产出负数 input_tokens，退回整段 prompt。
func TestMessagesUsageFieldsInconsistentUpstream(t *testing.T) {
	got := messagesUsageFields(map[string]any{
		"prompt_tokens":               10.0,
		"completion_tokens":           1.0,
		"cache_read_input_tokens":     20.0,
		"cache_creation_input_tokens": 5.0,
	})
	if v := got["input_tokens"].(int64); v != 10 {
		t.Fatalf("input_tokens=%d want 10 (fallback to prompt, no negative)", v)
	}
	if v := got["cache_read_input_tokens"].(int64); v != 20 {
		t.Fatalf("cache_read=%d want 20 (pass through upstream value)", v)
	}
}

// TestMessagesUsageFieldsPromptCacheWriteAlias：上游用 prompt_cache_write_tokens
// 命名写入维度时同样识别。
func TestMessagesUsageFieldsPromptCacheWriteAlias(t *testing.T) {
	got := messagesUsageFields(map[string]any{
		"prompt_tokens":             50.0,
		"completion_tokens":         2.0,
		"prompt_cache_hit_tokens":   30.0,
		"prompt_cache_write_tokens": 8.0,
	})
	if v := got["input_tokens"].(int64); v != 12 {
		t.Fatalf("input_tokens=%d want 12 (50-30-8)", v)
	}
	if v := got["cache_read_input_tokens"].(int64); v != 30 {
		t.Fatalf("cache_read=%d want 30", v)
	}
	if v := got["cache_creation_input_tokens"].(int64); v != 8 {
		t.Fatalf("cache_creation=%d want 8", v)
	}
}
