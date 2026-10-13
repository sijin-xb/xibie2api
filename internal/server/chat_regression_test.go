// chat_regression_test.go —— /v1/chat/completions 的回归安全网（Stage 0）。
//
// 目的：在「抽取共享协议核心」重构前，把当前 HTTP 边界上的可观测行为逐字节钉住，
// 使重构可被证明为行为保持。本文件只做**黑盒断言**：一律经 NewHandler + ServeHTTP
// 走完整管线（读体 → 别名 → realm 解析 → 提示词改写 → 选号 → 出站 → 聚合/透传 →
// 记账），不直调内部函数。出站请求体由 roundTripFunc 捕获后断言。
//
// 复用 handler_test.go 既有夹具：newFakeUpstream 之外的假上游在本文件用
// newRegressionUpstream（额外记录出站 Authorization + body）；SSE 夹具复用 sseOK、
// markupFixture、markupUpstreamSSE、declaredToolsBody；池用 testPoolWith（确定性
// RNG + credits=1000）。
//
// 说明（离线可验证性）：
//   - 网关**始终**以 stream:true 打上游（upstream.PrepareBodyOptWithEffortsAndDefault
//     强制），因此「非流式客户端请求」对应的上游响应仍是 SSE，由 handler 聚合成
//     chat.completion。不存在「上游回非流式 JSON」的形态，测试据此对齐真实管线。
//   - 工具调用增量合并分两处：非流式走 Aggregate（网关合并后下发完整 tool_calls）；
//     流式走 StreamHint（逐帧透传、仅按 index 收敛 name，合并由客户端完成）。两者
//     都逐字节钉住。
//   - DSML 标记修复合成的 tool_call id 含 time.Now().UnixNano()（upstream.markupCallID），
//     非确定；相关用例把 id 规范化为 call_TS_N 后再逐字节比较，并单独断言 id 形态。
package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// regressionCall 记录一次出站请求的可观测面（Authorization + 出站 body）。
type regressionCall struct {
	auth string
	body []byte
}

// newRegressionUpstream 构造一个记录出站调用的假上游。respond 按调用序号（1-based）、
// Authorization 与出站 body 决定响应；isStream 只影响 Content-Type。
func newRegressionUpstream(respond func(n int, authz string, body []byte) (status int, respBody string, isStream bool)) (*upstream.Client, func() []regressionCall) {
	var mu sync.Mutex
	var calls []regressionCall
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			b, _ := io.ReadAll(r.Body)
			authz := r.Header.Get("Authorization")
			mu.Lock()
			calls = append(calls, regressionCall{auth: authz, body: b})
			n := len(calls)
			mu.Unlock()
			status, body, isStream := respond(n, authz, b)
			ct := "application/json"
			if isStream {
				ct = "text/event-stream"
			}
			return &http.Response{
				StatusCode: status,
				Header:     http.Header{"Content-Type": []string{ct}},
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	get := func() []regressionCall {
		mu.Lock()
		defer mu.Unlock()
		out := make([]regressionCall, len(calls))
		copy(out, calls)
		return out
	}
	return up, get
}

// staticRegressionUpstream 恒返回同一响应。
func staticRegressionUpstream(status int, body string, isStream bool) (*upstream.Client, func() []regressionCall) {
	return newRegressionUpstream(func(int, string, []byte) (int, string, bool) { return status, body, isStream })
}

func assertExactBody(t *testing.T, got, want string) {
	t.Helper()
	if got != want {
		t.Fatalf("response body mismatch\n got: %q\nwant: %q", got, want)
	}
}

var regressionCallIDRe = regexp.MustCompile(`call_\d+_\d+`)

// normalizeRegressionCallIDs 把 DSML 修复合成的非确定 id（call_<nano>_<idx>）归一，
// 以便对其余字节做精确比较。
func normalizeRegressionCallIDs(s string) string {
	return regressionCallIDRe.ReplaceAllString(s, "call_TS_N")
}

func mustDecodeJSON(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("not json: %v body=%s", err, b)
	}
	return v
}

// ---------------------------------------------------------------------------
// 固定 SSE 夹具（全部为真实换行的 SSE 帧；raw string 让 JSON 转义原样可读）。
// ---------------------------------------------------------------------------

// regNonStreamSSE：两段正文 + stop + usage，用于非流式聚合的逐字节断言。
const regNonStreamSSE = `data: {"id":"chatcmpl-fixed","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"}}]}

data: {"id":"chatcmpl-fixed","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"content":"，世界"}}]}

data: {"id":"chatcmpl-fixed","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}}

data: [DONE]

`

// regFragmentedToolSSE：tool_calls 的 arguments 被拆到多个 chunk（增量合并输入）。
const regFragmentedToolSSE = `data: {"id":"chatcmpl-tc","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_frag","type":"function","function":{"name":"exec_command"}}]}}]}

data: {"id":"chatcmpl-tc","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":"}}]}}]}

data: {"id":"chatcmpl-tc","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls -la\"}"}}]}}]}

data: {"id":"chatcmpl-tc","object":"chat.completion.chunk","created":1753600001,"model":"glm-5.2","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":9,"total_tokens":12}}

data: [DONE]

`

// regCreditSSE：带 usage.credit 的流，用于记账断言。
const regCreditSSE = `data: {"id":"chatcmpl-credit","object":"chat.completion.chunk","created":1753600000,"model":"glm-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10,"credit":1.25}}

data: [DONE]

`

// regReasoningSSE：推理模型先流 reasoning_content（思维链），再流 content（正文）。
// 用于验证 Anthropic /v1/messages 渲染把思维链映射为 thinking 块（而非丢弃）。
const regReasoningSSE = `data: {"id":"chatcmpl-reason","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"先算"}}]}

data: {"id":"chatcmpl-reason","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"reasoning_content":"17乘23"}}]}

data: {"id":"chatcmpl-reason","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"content":"391"}}]}

data: {"id":"chatcmpl-reason","object":"chat.completion.chunk","created":1753600002,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}

data: [DONE]

`

// ---------------------------------------------------------------------------
// 1. 非流式请求：上游 SSE 聚合为 chat.completion，响应体逐字节。
// ---------------------------------------------------------------------------

// TestChatRegressionNonStreamAggregatedBodyExact 锁定：stream:false 客户端请求 →
// 网关把上游 SSE 聚合为 chat.completion，响应体逐字节等于期望（content 拼接、model
// 回显、usage 透传、finish_reason）。任何聚合字段/取值/键序变化都会失败。
func TestChatRegressionNonStreamAggregatedBodyExact(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regNonStreamSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":false,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	const want = `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"你好，世界","role":"assistant"}}],"created":1753600000,"id":"chatcmpl-fixed","model":"glm-5.2","object":"chat.completion","usage":{"completion_tokens":5,"prompt_tokens":7,"total_tokens":12}}`
	assertExactBody(t, rec.Body.String(), want)
}

// ---------------------------------------------------------------------------
// 2. 流式请求：上游 SSE 逐帧透传（规范化后），响应体逐字节 + Content-Type。
// ---------------------------------------------------------------------------

// TestChatRegressionStreamPassthroughBodyExact 锁定：stream:true → 逐帧规范化透传，
// 整段响应字节（含终止 data: [DONE]）精确匹配；Content-Type 为 text/event-stream。
func TestChatRegressionStreamPassthroughBodyExact(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type=%q want text/event-stream", ct)
	}
	const want = `data: {"choices":[{"delta":{"content":"你好","role":"assistant"},"finish_reason":null,"index":0}],"created":1753600000,"id":"chatcmpl-1","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":1753600000,"id":"chatcmpl-1","model":"glm-5.2","object":"chat.completion.chunk","usage":{"completion_tokens":1,"prompt_tokens":1,"total_tokens":2}}

data: [DONE]

`
	assertExactBody(t, rec.Body.String(), want)
}

// ---------------------------------------------------------------------------
// 3. 工具调用增量合并（非流式聚合 + 流式透传契约）
// ---------------------------------------------------------------------------

// TestChatRegressionFragmentedToolCallsNonStreamExact 锁定：上游把 tool_calls 的
// arguments 拆到多个 chunk 时，非流式聚合按 index 合并为完整 arguments JSON，
// 响应体逐字节（含 finish_reason=tool_calls、index 字段）。
func TestChatRegressionFragmentedToolCallsNonStreamExact(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regFragmentedToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	const want = `{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"cmd\":\"ls -la\"}","name":"exec_command"},"id":"call_frag","index":0,"type":"function"}]}}],"created":1753600001,"id":"chatcmpl-tc","model":"glm-5.2","object":"chat.completion","usage":{"completion_tokens":9,"prompt_tokens":3,"total_tokens":12}}`
	assertExactBody(t, rec.Body.String(), want)
}

// TestChatRegressionFragmentedToolCallsStreamExact 锁定流式契约：网关**不**跨帧合并
// tool_calls，而是逐帧透传（首片保留 function.name，同 index 的后续分片删除 name 键）；
// 合并由客户端按 index 完成。整段响应字节精确匹配。
func TestChatRegressionFragmentedToolCallsStreamExact(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regFragmentedToolSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	const want = `data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"function":{"name":"exec_command"},"id":"call_frag","index":0,"type":"function"}]},"finish_reason":null,"index":0}],"created":1753600001,"id":"chatcmpl-tc","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"cmd\":"},"index":0}]},"finish_reason":null,"index":0}],"created":1753600001,"id":"chatcmpl-tc","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\"ls -la\"}"},"index":0}]},"finish_reason":null,"index":0}],"created":1753600001,"id":"chatcmpl-tc","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],"created":1753600001,"id":"chatcmpl-tc","model":"glm-5.2","object":"chat.completion.chunk","usage":{"completion_tokens":9,"prompt_tokens":3,"total_tokens":12}}

data: [DONE]

`
	assertExactBody(t, rec.Body.String(), want)
}

// TestChatRegressionDSMLRepairNonStreamExact 补充既有 TestChatNonStreamRepairsMarkupToolCalls
// （后者只断言 name/cmd/finish_reason）：本用例把**修复后的完整响应体**逐字节钉住
// （含 arguments 精确 JSON `{"cmd":"ls -la"}`、正文摘除标记后的精确文本、finish_reason
// 收敛为 tool_calls）。id 为 time-based 合成值，先规范化。
func TestChatRegressionDSMLRepairNonStreamExact(t *testing.T) {
	content := markupFixture("先读一遍控制器。\n\n", "ls -la")
	up, _ := staticRegressionUpstream(200, markupUpstreamSSE(content), true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	reqBody := `{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],` + declaredToolsBody + `}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	const want = `{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"content":"先读一遍控制器。\n\n","role":"assistant","tool_calls":[{"function":{"arguments":"{\"cmd\":\"ls -la\"}","name":"exec_command"},"id":"call_TS_N","type":"function"}]}}],"created":1,"id":"resp-1","model":"glm-5.2","object":"chat.completion"}`
	assertExactBody(t, normalizeRegressionCallIDs(rec.Body.String()), want)
	// id 形态守卫：合成 id 必须匹配 call_<digits>_<digits>。
	if !regressionCallIDRe.MatchString(rec.Body.String()) {
		t.Fatalf("synthesized tool_call id missing/ill-formed: %s", rec.Body)
	}
}

// TestChatRegressionDSMLRepairStreamExact 补充既有 TestChatStreamRepairsMarkupToolCalls
// （后者只统计帧）：本用例把修复后的**整段流字节**逐字节钉住——正文帧（标记已摘除）+
// tool_calls 帧（精确 arguments JSON）+ 收尾 finish_reason=tool_calls 帧 + [DONE]。
func TestChatRegressionDSMLRepairStreamExact(t *testing.T) {
	content := markupFixture("先读一遍控制器。\n\n", "ls -la")
	up, _ := staticRegressionUpstream(200, markupUpstreamSSE(content), true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	reqBody := `{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}],` + declaredToolsBody + `}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	const want = `data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: {"choices":[{"delta":{"content":"先读一遍控制器。\n\n"},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"cmd\":\"ls -la\"}","name":"exec_command"},"id":"call_TS_N","index":0,"type":"function"}]},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],"created":1,"id":"resp-1","model":"glm-5.2","object":"chat.completion.chunk","usage":null}

data: [DONE]

`
	assertExactBody(t, normalizeRegressionCallIDs(rec.Body.String()), want)
}

// ---------------------------------------------------------------------------
// 4. 用量 / 账号记账（精确增量）
// ---------------------------------------------------------------------------

// TestChatRegressionUsageAccounting 锁定成功请求的账号台账增量：流式与非流式各一次
// 成功请求后，p.Status(uid) 的 token 计数与 credits 精确反映 usage（credit=1.25 →
// credits 1000→999，四舍五入扣减）。断言精确增量而非"发生过调用"。
func TestChatRegressionUsageAccounting(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"nonstream", `{"model":"glm-5.2","messages":[]}`},
		{"stream", `{"model":"glm-5.2","stream":true,"messages":[]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			up, _ := staticRegressionUpstream(200, regCreditSSE, true)
			p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
			h := NewHandler(Config{Pool: p, Upstream: up})

			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(tc.body)))
			if rec.Code != http.StatusOK {
				t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
			}
			st, ok := p.Status("u1")
			if !ok {
				t.Fatal("account status missing")
			}
			tu := st.TokenUsage
			if tu.RequestCount != 1 || tu.UsageCount != 1 ||
				tu.PromptTokens != 4 || tu.CompletionTokens != 6 || tu.TotalTokens != 10 {
				t.Fatalf("token usage=%+v want req=1 usage=1 prompt=4 completion=6 total=10", tu)
			}
			if tu.LastModel != "glm-5.2" {
				t.Errorf("last_model=%q want glm-5.2", tu.LastModel)
			}
			if st.Credits != 999 {
				t.Errorf("credits=%d want 999 (1000 - round(1.25))", st.Credits)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 5. 错误响应 + 上游异常（响应体逐字节 + 账号状态）
// ---------------------------------------------------------------------------

// TestChatRegressionUpstream4xxBodyExactNoPenalty 锁定：上游 4xx 客户端错误 →
// 网关 503 且 body 精确（no_healthy_account + 上游原文透传，无 gateway_hint），
// 账号不被冷却/禁用/累计 errTotal（NoteFailures 只喂连败计数，未达阈无副作用）。
func TestChatRegressionUpstream4xxBodyExactNoPenalty(t *testing.T) {
	up, _ := staticRegressionUpstream(400, `{"code":400,"msg":"bad request"}`, false)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code=%d want 503 body=%s", rec.Code, rec.Body)
	}
	const want = `{"error":{"code":"no_healthy_account","message":"{\"code\":400,\"msg\":\"bad request\"}","type":"api_error"}}`
	assertExactBody(t, rec.Body.String(), want)

	st, _ := p.Status("u1")
	if st.Cooling || st.Disabled || st.ErrTotal != 0 {
		t.Fatalf("4xx must not penalize account: %+v", st)
	}
	if st.ConsecutiveFails != 1 {
		t.Errorf("consecutive_fails=%d want 1 (ErrClient feeds consecutive-failure counter, below degrade threshold)", st.ConsecutiveFails)
	}
}

// TestChatRegressionUpstream5xxBodyExactPenalizes 锁定 5xx 的两面：
//   - 单号：网关 503，body 精确（no_healthy_account + 上游原文，无 hint），账号熔断冷却、
//     errTotal 累计（MUST penalize）；
//   - 双号：5xx 号被罚后轮转到健康号成功（MUST rotate）。
func TestChatRegressionUpstream5xxBodyExactPenalizes(t *testing.T) {
	t.Run("single_account_body_and_state", func(t *testing.T) {
		up, _ := staticRegressionUpstream(500, `{"code":500}`, false)
		p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
		p.SetBreaker(1, time.Hour, time.Hour) // 阈值 1：一次 5xx 即熔断（冷却可观测）
		h := NewHandler(Config{Pool: p, Upstream: up})

		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
			strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("code=%d want 503 body=%s", rec.Code, rec.Body)
		}
		const want = `{"error":{"code":"no_healthy_account","message":"{\"code\":500}","type":"api_error"}}`
		assertExactBody(t, rec.Body.String(), want)

		st, _ := p.Status("u1")
		if !st.Cooling {
			t.Fatalf("5xx must trip breaker (cooling): %+v", st)
		}
		if st.ErrTotal != 1 {
			t.Errorf("err_total=%d want 1", st.ErrTotal)
		}
	})

	t.Run("two_accounts_rotate", func(t *testing.T) {
		up, calls := newRegressionUpstream(func(n int, authz string, _ []byte) (int, string, bool) {
			if authz == "Bearer at-bad" {
				return 500, `{"code":500}`, false
			}
			return 200, sseOK, true
		})
		p := testPoolWith(
			&auth.Auth{UID: "bad", AccessToken: "at-bad", ExpiresAt: 9999999999},
			&auth.Auth{UID: "good", AccessToken: "at-good", ExpiresAt: 9999999999},
		)
		p.SetCredits("bad", 2000, 0) // 确定性源 r=0 + 高积分 → bad 先被选中
		p.SetCredits("good", 1000, 0)
		p.SetBreaker(1, time.Hour, time.Hour)
		h := NewHandler(Config{Pool: p, Upstream: up, SoftCooldown: time.Minute})

		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
			strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("code=%d body=%s (want rotate to good)", rec.Code, rec.Body)
		}
		got := calls()
		if len(got) != 2 || got[0].auth != "Bearer at-bad" || got[1].auth != "Bearer at-good" {
			t.Fatalf("upstream auth sequence=%v want [at-bad, at-good]", got)
		}
		stBad, _ := p.Status("bad")
		if !stBad.Cooling {
			t.Errorf("bad should be cooling after 5xx: %+v", stBad)
		}
		stGood, _ := p.Status("good")
		if stGood.Cooling || stGood.Disabled {
			t.Errorf("good should stay healthy: %+v", stGood)
		}
	})
}

// TestChatRegressionTransportErrorBodyExactNoPenalty 锁定传输层错误：网关 503，
// body 精确（本地 no_healthy_account 文案 + no-healthy gateway_hint——无上游原文），
// 账号不被冷却/禁用/累计 errTotal（连败计数 +1，未达阈）。
func TestChatRegressionTransportErrorBodyExactNoPenalty(t *testing.T) {
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return nil, io.ErrUnexpectedEOF
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code=%d want 503 body=%s", rec.Code, rec.Body)
	}
	const want = `{"error":{"code":"no_healthy_account","gateway_hint":"no healthy account available in pool; check /status or retry later","message":"all accounts are temporarily unavailable, please retry later","type":"api_error"}}`
	assertExactBody(t, rec.Body.String(), want)

	st, _ := p.Status("u1")
	if st.Cooling || st.Disabled || st.ErrTotal != 0 {
		t.Fatalf("transport error must not penalize account: %+v", st)
	}
}

// ---------------------------------------------------------------------------
// 6. 请求取消
// ---------------------------------------------------------------------------

// TestChatRegressionRequestCancellation 锁定客户端在请求已取消时的行为：
//   - handler 立即终止（不挂起、不空转 MaxRotate 次）；
//   - 返回 503 + 本地 no_healthy_account 文案（与传输层抖动同出口，非"没号可用"以外
//     的形态）；
//   - 账号不被冷却/禁用、errTotal 不累计（连败计数 +1，未达阈；请求计数 +1，因为
//     recordAttempt 对失败尝试也计数）。
//
// 观察到的行为（本仓库当前实现）：r.Context() 已取消 → http 传输失败 → isUpstreamTimeout
// 因 clientGone=true 判否 → 走"网络抖动"分支（NoteFailures + fail + rotateBackoff，
// 后者因 ctx 取消立即返回 false 终止轮转）→ 末端 503。ctx-aware 假上游在取消时返回
// r.Context().Err()，确保该路径被真实触发。
func TestChatRegressionRequestCancellation(t *testing.T) {
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if err := r.Context().Err(); err != nil {
				return nil, err
			}
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(sseOK)),
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[]}`))
	ctx, cancel := context.WithCancel(req.Context())
	cancel()
	req = req.WithContext(ctx)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req) // 若挂起，测试超时即失败

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code=%d want 503 body=%s", rec.Code, rec.Body)
	}
	const want = `{"error":{"code":"no_healthy_account","gateway_hint":"no healthy account available in pool; check /status or retry later","message":"all accounts are temporarily unavailable, please retry later","type":"api_error"}}`
	assertExactBody(t, rec.Body.String(), want)

	st, _ := p.Status("u1")
	if st.Cooling || st.Disabled || st.ErrTotal != 0 {
		t.Fatalf("cancelled request must not penalize account (cooling/disabled/errTotal): %+v", st)
	}
	if st.TokenUsage.RequestCount != 1 {
		t.Errorf("request_count=%d want 1 (failed attempts are still counted)", st.TokenUsage.RequestCount)
	}
}

// ---------------------------------------------------------------------------
// 7. 账号选择 / 硬积分轮转
// ---------------------------------------------------------------------------

// TestChatRegressionAccountSelectionAndHardCreditRotation 锁定选号与轮转：
//   - 2 号池中，确定性 RNG（r=0）+ bad 高积分 → 首次选中 bad（Authorization 见 at-bad）；
//   - bad 返回 402 硬积分 → 立即轮转到 good 成功（Authorization 见 at-good）；
//   - bad 进入硬积分冷却（至次日 04:00），good 保持健康。
func TestChatRegressionAccountSelectionAndHardCreditRotation(t *testing.T) {
	up, calls := newRegressionUpstream(func(n int, authz string, _ []byte) (int, string, bool) {
		if authz == "Bearer at-bad" {
			return 402, `{"code":1,"msg":"余额不足"}`, false
		}
		return 200, sseOK, true
	})
	p := testPoolWith(
		&auth.Auth{UID: "bad", AccessToken: "at-bad", ExpiresAt: 9999999999},
		&auth.Auth{UID: "good", AccessToken: "at-good", ExpiresAt: 9999999999},
	)
	p.SetCredits("bad", 2000, 0)
	p.SetCredits("good", 1000, 0)
	h := NewHandler(Config{Pool: p, Upstream: up, SoftCooldown: time.Minute})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s (want 200 after hard-credit rotation)", rec.Code, rec.Body)
	}
	got := calls()
	if len(got) != 2 {
		t.Fatalf("upstream calls=%d want 2: %v", len(got), got)
	}
	if got[0].auth != "Bearer at-bad" {
		t.Fatalf("first pick=%q want Bearer at-bad (deterministic RNG + higher credits)", got[0].auth)
	}
	if got[1].auth != "Bearer at-good" {
		t.Fatalf("rotation target=%q want Bearer at-good", got[1].auth)
	}
	stBad, _ := p.Status("bad")
	if !stBad.Cooling || stBad.Until.Hour() != 4 {
		t.Fatalf("hard-credit account should cool until next-day 04:00: %+v", stBad)
	}
	stGood, _ := p.Status("good")
	if stGood.Cooling || stGood.Disabled {
		t.Fatalf("good should stay healthy: %+v", stGood)
	}
}

// ---------------------------------------------------------------------------
// 8. 模型别名（端到端）
// ---------------------------------------------------------------------------

// TestChatRegressionModelAliasOutbound 锁定模型别名端到端：命中别名 → 出站 body 的
// model 重写为解析后的裸名（claude-sonnet-4-5 → cn:glm-5.2 → glm-5.2）；未命中 → 原样
// 透传。并证明选号用的是解析后的裸模型名（对裸名做模型级冷却即选不到号 → 503）。
func TestChatRegressionModelAliasOutbound(t *testing.T) {
	tbl, err := NewAliasTable([]ModelAlias{{From: "claude-sonnet-4-5", To: "cn:glm-5.2"}})
	if err != nil {
		t.Fatalf("build alias table: %v", err)
	}

	up, calls := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, ModelAliases: tbl})

	// 命中别名：出站 model 必须是被解析后的裸名 glm-5.2，别名原名零残留。
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("aliased code=%d body=%s", rec.Code, rec.Body)
	}
	out := mustDecodeJSON(t, calls()[0].body)
	if out["model"] != "glm-5.2" {
		t.Fatalf("outbound model=%v want glm-5.2 (alias resolved to bare name)", out["model"])
	}
	if strings.Contains(string(calls()[0].body), "claude-sonnet-4-5") {
		t.Fatalf("alias source name must not leak upstream: %s", calls()[0].body)
	}

	// 未命中别名：原样透传。
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"unmapped-model","messages":[{"role":"user","content":"hi"}]}`)))
	if rec2.Code != http.StatusOK {
		t.Fatalf("unmapped code=%d body=%s", rec2.Code, rec2.Body)
	}
	out2 := mustDecodeJSON(t, calls()[1].body)
	if out2["model"] != "unmapped-model" {
		t.Fatalf("unmapped model=%v want unmapped-model (passthrough)", out2["model"])
	}

	// 选号路径使用解析后的裸名：对 glm-5.2 做模型级冷却 → 选号无候选且被判定为
	// 模型级阻塞 → 400 model_unavailable（而非 503 no_healthy_account）。
	// 若选号误用别名原名，u1 不会被冷却、请求仍会成功，本断言即失败。
	p.BlockModelBackoff("u1", "glm-5.2", upstream.ModelBlockReason)
	rec3 := httptest.NewRecorder()
	h.ServeHTTP(rec3, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`)))
	if rec3.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400 model_unavailable (resolved model glm-5.2 is model-cooled on the only account), body=%s", rec3.Code, rec3.Body)
	}
	if !strings.Contains(rec3.Body.String(), "model_unavailable") {
		t.Fatalf("body=%s want model_unavailable code", rec3.Body)
	}
}

// ---------------------------------------------------------------------------
// 9. system/提示词改写 + 脱敏管线（出站 body）
// ---------------------------------------------------------------------------

// TestChatRegressionPromptOverrideAndSanitizeOutbound 锁定出站前的提示词管线：
//   - custom + PromptText：客户端 system 消息被替换为自有提示词，原 system 零残留；
//   - 对照（custom 但 PromptText 空）：system 原样透传；
//   - 两条路径的 user 消息出站字节完全一致（提示词改写层不动 user 消息）；
//   - 脱敏层实际生效：user 消息里的**未改写指纹源串**被 sanitizeRewrites 改写成目标串。
//
// 脱敏源/目标串必须以字节级工具核对 upstream/sanitize.go 的 sanitizeRewrites——普通
// 终端渲染会折叠部分字符串，须用 xxd 等确认：
//   - 身份句：源 "…official CLI for Claude" → 目标 "…official CLI tool for Claude"（插入 "tool"）；
//   - 分支句：源 "Main branch (…)" → 目标 "Default branch (…)"（Main→Default）。
//
// 本用例喂入的是**未改写的源串**（关键：绝不能喂已含 "tool" 的目标串，否则不匹配源
// 模式、脱敏不触发），断言出站 body 含目标串、不含源串。
//
// 与 handler_test.go 的 TestCustomModeFingerprintSanitizePreserved 同层（都经 handler
// 端到端）；本用例更精简：只钉住「脱敏改写确实发生」与 prompt 覆盖两点，不重复其矩阵。
func TestChatRegressionPromptOverrideAndSanitizeOutbound(t *testing.T) {
	// 未改写的指纹源串（身份句缺 "tool"；分支句为 "Main branch ("）。
	const fingerprintSrc = "You are Claude Code, Anthropic's official CLI for Claude. Main branch (you will usually use this for PRs)"
	// sanitizeRewrites 改写后的目标形态。
	const fingerprintDst = "You are Claude Code, Anthropic's official CLI tool for Claude. Default branch (you will usually use this for PRs)"

	// 自检：源串必须真的不同于目标串、且含未改写形态；否则断言会因"源==目标"而虚假通过
	// （终端渲染会把这两串显示得几乎一样，务必以本自检 + 出站断言双重保证）。
	if fingerprintSrc == fingerprintDst {
		t.Fatal("fingerprintSrc must differ from fingerprintDst (fixture would be vacuous)")
	}
	for _, pre := range []string{"official CLI for Claude", "Main branch ("} {
		if !strings.Contains(fingerprintSrc, pre) {
			t.Fatalf("fingerprintSrc missing pre-rewrite form %q", pre)
		}
	}

	// 同一份请求体：system 原文 + user 未改写指纹串。
	reqBody := `{"model":"glm-5.2","messages":[{"role":"system","content":"ORIGINAL_SYS"},{"role":"user","content":"` + fingerprintSrc + `"}]}`

	run := func(promptText string) []byte {
		up, calls := staticRegressionUpstream(200, sseOK, true)
		up.SanitizeFingerprints.Store(true)
		p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
		h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: "custom", PromptText: promptText})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody)))
		if rec.Code != http.StatusOK {
			t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
		}
		return calls()[0].body
	}

	overrideBody := run("<system>")
	noOverrideBody := run("")
	override := mustDecodeJSON(t, overrideBody)
	noOverride := mustDecodeJSON(t, noOverrideBody)

	sysOf := func(v map[string]any) string {
		msgs := v["messages"].([]any)
		return msgs[0].(map[string]any)["content"].(string)
	}
	userOf := func(v map[string]any) string {
		msgs := v["messages"].([]any)
		return msgs[1].(map[string]any)["content"].(string)
	}

	// 1) prompt 覆盖：custom+PromptText 替换 system 且原 system 零残留；空 PromptText 透传。
	if got := sysOf(override); got != "<system>" {
		t.Fatalf("override system content=%q want <system>", got)
	}
	if strings.Contains(string(overrideBody), "ORIGINAL_SYS") {
		t.Fatalf("original system must be removed when overridden: %s", overrideBody)
	}
	if got := sysOf(noOverride); got != "ORIGINAL_SYS" {
		t.Fatalf("no-override system content=%q want ORIGINAL_SYS (passthrough)", got)
	}

	// 2) 脱敏层实际生效：出站 body 含改写目标串、不含未改写源串（行为检测，非"HTTP 200"）。
	if !strings.Contains(string(overrideBody), fingerprintDst) {
		t.Fatalf("outbound body missing sanitized fingerprint %q: %s", fingerprintDst, overrideBody)
	}
	for _, src := range []string{
		"official CLI for Claude", // 身份句源片段（目标串含 "tool"，不含此片段）
		"Main branch (",           // 分支句源片段
	} {
		if strings.Contains(string(overrideBody), src) {
			t.Fatalf("outbound body still contains unsanitized source %q: %s", src, overrideBody)
		}
	}

	// 3) 提示词改写层不动 user 消息：两条路径 user 出站字节一致，且等于脱敏目标串。
	if a, b := userOf(override), userOf(noOverride); a != b {
		t.Fatalf("user message diverged between prompt modes: override=%q noOverride=%q", a, b)
	}
	if got := userOf(override); got != fingerprintDst {
		t.Fatalf("user fingerprint content=%q want sanitized %q", got, fingerprintDst)
	}
}
