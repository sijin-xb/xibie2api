package server

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/apikey"
	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// round2_fixes_test.go 覆盖第二轮审查修复的缺陷（Responses 适配 / 鉴权 / 分类）。

// #1 Responses 流式 custom 工具：arguments 分片先非法 JSON（返回原文，长）后拼成
// 合法 JSON（input 值更短）→ 不得切片越界 panic。
func TestResponsesCustomToolNoPanicOnShorterInput(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"apply_patch","arguments":"{\"input\":\"ab"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"c\"}"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	// 不得 panic（此前 slice bounds out of range）。
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
		`{"model":"m","max_tokens":100,"stream":true,"tools":[{"type":"custom","name":"apply_patch","description":"edit","input_schema":{"type":"object"}}],"input":"hi"}`)))
	if rec.Code != 200 {
		t.Fatalf("code=%d", rec.Code)
	}
}

// #2 全部 api_keys 禁用 → 仍要求鉴权（拒绝一切），不得变成无鉴权开放访问。
func TestAllKeysDisabledStillRequiresAuth(t *testing.T) {
	no := false
	tbl, err := apikey.NewTable([]apikey.Entry{
		{Key: "k1", Enabled: &no},
		{Key: "k2", Enabled: &no},
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !tbl.Required() {
		t.Fatal("configuring api_keys must require auth even when all disabled")
	}
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer totally-wrong")
	if _, ok := tbl.Authenticate(req); ok {
		t.Fatal("all-disabled config must reject every key, not allow anonymous access")
	}
}

// #3 body 任意位置出现 "12153" 不得判成 session_dead（避免永久误杀账号）。
func TestBareCode12153NotSessionDead(t *testing.T) {
	up, _ := staticRegressionUpstream(400, `{"code":400,"msg":"bad request","requestId":"req-12153-abc"}`, false)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))

	st, _ := p.Status("u1")
	if st.Disabled {
		t.Fatalf("account disabled by bare '12153' substring (wire=%d)", rec.Code)
	}
}

// #5 Aggregate 多帧 usage 必须按 key 合并，不得整表覆盖丢字段。
func TestAggregateUsageMergedAcrossFrames(t *testing.T) {
	const sse = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_cache_hit_tokens":8}}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":15}}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, false)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)))

	obj := mustDecodeJSON(t, rec.Body.Bytes())
	usage, _ := obj["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 10 {
		t.Fatalf("prompt_tokens lost by whole-map overwrite: %v", usage)
	}
}

// #6 Responses 流式 error 帧后不得再输出正文。
func TestResponsesNoContentAfterError(t *testing.T) {
	const sse = `data: {"error":{"message":"boom","type":"upstream_error"}}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"LEAKED"}}]}

data: [DONE]

`
	up, _ := staticRegressionUpstream(200, sse, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"m","max_tokens":100,"stream":true,"input":"hi"}`)))

	if strings.Contains(rec.Body.String(), "LEAKED") {
		t.Fatal("content emitted after error frame")
	}
}

// #7 非流式渲染失败（Anthropic 路径工具参数非法 JSON → 502）不得记账号成功。
func TestNonStreamRenderFailureNotMarkedSuccess(t *testing.T) {
	up, _ := staticRegressionUpstream(200, msgBadToolArgsSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != 502 {
		t.Fatalf("expected 502 render failure, got %d", rec.Code)
	}
	st, _ := p.Status("u1")
	if st.SuccessCount > 0 {
		t.Fatalf("account marked success despite 502 render failure (success=%d)", st.SuccessCount)
	}
}
