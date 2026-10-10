package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/prompt"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// captureUpstream 返回一个把每次出站请求体追加到 bodies 的 upstream.Client，
// 响应由 behavior(authz) 决定。用于对**捕获到的出站 body**做行为断言（而非只看
// HTTP 状态码）。transport 在 handler 同 goroutine 内同步调用，无并发写。
func captureUpstream(bodies *[][]byte, behavior func(authz string) (status int, body string, isStream bool)) *upstream.Client {
	return &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			raw, _ := io.ReadAll(r.Body)
			*bodies = append(*bodies, raw)
			status, body, isStream := behavior(r.Header.Get("Authorization"))
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
}

// outboundMessages 解析出站 body 的 messages 数组为对象切片，失败即 Fatal。
func outboundMessages(t *testing.T, body []byte) []map[string]any {
	t.Helper()
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		t.Fatalf("outbound body not JSON: %v body=%s", err, body)
	}
	raw, ok := obj["messages"].([]any)
	if !ok {
		t.Fatalf("outbound messages not an array: %v (body=%s)", obj["messages"], body)
	}
	out := make([]map[string]any, 0, len(raw))
	for _, m := range raw {
		mm, ok := m.(map[string]any)
		if !ok {
			t.Fatalf("message not an object: %v", m)
		}
		out = append(out, mm)
	}
	return out
}

// countSystemContent 统计 role=="system" 且 content==want 的消息条数。
func countSystemContent(msgs []map[string]any, want string) int {
	n := 0
	for _, m := range msgs {
		if m["role"] == "system" && m["content"] == want {
			n++
		}
	}
	return n
}

// TestPromptCustomModeOutboundReplacesClientSystem custom 模式：出站 body 恰好一条
// system，内容为网关 PromptText；客户端 system 原文零残留；user 消息逐字不变。
func TestPromptCustomModeOutboundReplacesClientSystem(t *testing.T) {
	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: "custom", PromptText: "GATEWAY_SYS"})

	body := `{"model":"glm-5.2","stream":true,"messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) != 1 {
		t.Fatalf("upstream attempts=%d want 1", len(bodies))
	}
	out := bodies[0]
	msgs := outboundMessages(t, out)

	sysCount := 0
	for _, m := range msgs {
		if m["role"] == "system" {
			sysCount++
			if m["content"] != "GATEWAY_SYS" {
				t.Errorf("system content=%v want GATEWAY_SYS (all=%v)", m["content"], msgs)
			}
		}
	}
	if sysCount != 1 {
		t.Errorf("want exactly 1 system message, got %d (all=%v)", sysCount, msgs)
	}
	if strings.Contains(string(out), "CLIENT_SYS") {
		t.Errorf("client system must be gone from outbound body: %s", out)
	}
	userCount := 0
	for _, m := range msgs {
		if m["role"] == "user" {
			userCount++
			if m["content"] != "hi" {
				t.Errorf("user content=%v want hi", m["content"])
			}
		}
	}
	if userCount != 1 {
		t.Errorf("want exactly 1 user message, got %d (all=%v)", userCount, msgs)
	}
}

// TestPromptAppendModeOutboundPreservesClientSystem append 模式：出站同时含客户端
// system（逐字不变）与网关 system，恰好两条 system，顺序为
// [system(CLIENT_SYS), system(GATEWAY_SYS), user]。
func TestPromptAppendModeOutboundPreservesClientSystem(t *testing.T) {
	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: "append", PromptText: "GATEWAY_SYS"})

	body := `{"model":"glm-5.2","stream":true,"messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) != 1 {
		t.Fatalf("upstream attempts=%d want 1", len(bodies))
	}
	out := bodies[0]
	msgs := outboundMessages(t, out)

	if n := countSystemContent(msgs, "CLIENT_SYS"); n != 1 {
		t.Errorf("CLIENT_SYS count=%d want 1 (all=%v)", n, msgs)
	}
	if n := countSystemContent(msgs, "GATEWAY_SYS"); n != 1 {
		t.Errorf("GATEWAY_SYS count=%d want 1 (all=%v)", n, msgs)
	}
	if len(msgs) != 3 {
		t.Fatalf("messages=%v want 3 (system, system, user)", msgs)
	}
	if msgs[0]["role"] != "system" || msgs[0]["content"] != "CLIENT_SYS" {
		t.Errorf("msgs[0]=%v want system/CLIENT_SYS", msgs[0])
	}
	if msgs[1]["role"] != "system" || msgs[1]["content"] != "GATEWAY_SYS" {
		t.Errorf("msgs[1]=%v want system/GATEWAY_SYS", msgs[1])
	}
	if msgs[2]["role"] != "user" || msgs[2]["content"] != "hi" {
		t.Errorf("msgs[2]=%v want user/hi", msgs[2])
	}
}

// TestPromptPassthroughModeOutboundPreservesClientSystem passthrough 模式：出站
// system 内容为客户端原文，且不含网关提示词。
func TestPromptPassthroughModeOutboundPreservesClientSystem(t *testing.T) {
	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: "passthrough"})

	body := `{"model":"glm-5.2","stream":true,"messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) != 1 {
		t.Fatalf("upstream attempts=%d want 1", len(bodies))
	}
	out := bodies[0]
	msgs := outboundMessages(t, out)
	if n := countSystemContent(msgs, "CLIENT_SYS"); n != 1 {
		t.Errorf("CLIENT_SYS count=%d want 1 (all=%v)", n, msgs)
	}
	if strings.Contains(string(out), "GATEWAY_SYS") {
		t.Errorf("passthrough must not inject gateway prompt: %s", out)
	}
}

// TestPromptFileContentReachesOutbound 端到端：文件内容经 prompt.Load 进入
// PromptText，再在 custom 模式下逐字出现在出站 body。
func TestPromptFileContentReachesOutbound(t *testing.T) {
	const marker = "MARKER_9f3a2b_custom_prompt"
	dir := t.TempDir()
	fp := filepath.Join(dir, "p.md")
	if err := os.WriteFile(fp, []byte(marker), 0o600); err != nil {
		t.Fatal(err)
	}
	text, err := prompt.Load("custom", fp)
	if err != nil {
		t.Fatal(err)
	}

	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: "custom", PromptText: text})

	body := `{"model":"glm-5.2","stream":true,"messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) != 1 {
		t.Fatalf("upstream attempts=%d want 1", len(bodies))
	}
	if !strings.Contains(string(bodies[0]), marker) {
		t.Errorf("outbound body must contain prompt file marker %q: %s", marker, bodies[0])
	}
	if n := countSystemContent(outboundMessages(t, bodies[0]), marker); n != 1 {
		t.Errorf("marker system count=%d want 1", n)
	}
}

// TestPromptNoDuplicateAcrossRotation 轮转（首号 500 → 换号 200）时，注入的网关
// system 在每个出站 body 中恰好出现一次：提示词改写只在轮转循环前执行一次，
// 重试/换号不叠加。
func TestPromptNoDuplicateAcrossRotation(t *testing.T) {
	var bodies [][]byte
	calls := 0
	up := captureUpstream(&bodies, func(string) (int, string, bool) {
		calls++
		if calls == 1 {
			return 500, `{"code":500,"msg":"server error"}`, false
		}
		return 200, sseOK, true
	})
	p := testPoolWith(
		&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999},
		&auth.Auth{UID: "u2", AccessToken: "at2", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: "custom", PromptText: "GATEWAY_SYS"})

	body := `{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) < 2 {
		t.Fatalf("want >1 upstream attempt (rotation), got %d", len(bodies))
	}
	for i, b := range bodies {
		if n := countSystemContent(outboundMessages(t, b), "GATEWAY_SYS"); n != 1 {
			t.Errorf("attempt %d: GATEWAY_SYS count=%d want exactly 1 (body=%s)", i, n, b)
		}
	}
}

// TestPromptDegradedRetryNoDuplicate passthrough 模式首请求 400（内容拦截）→ 降级
// 重试用 Degraded 提示词。第二个出站 body 的 system 恰为 prompt.Degraded，且
// 恰好一条 system（不会把客户端 system 与 Degraded 叠加）。
func TestPromptDegradedRetryNoDuplicate(t *testing.T) {
	var bodies [][]byte
	calls := 0
	up := captureUpstream(&bodies, func(string) (int, string, bool) {
		calls++
		if calls == 1 {
			return 400, `{"code":11-128,"msg":"blocked by security policy"}`, false
		}
		return 200, sseOK, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, PromptMode: "passthrough"})

	body := `{"model":"glm-5.2","stream":true,"messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s (want 200 after degraded retry)", rec.Code, rec.Body)
	}
	if len(bodies) != 2 {
		t.Fatalf("upstream attempts=%d want 2 (400 + degraded retry)", len(bodies))
	}
	msgs := outboundMessages(t, bodies[1])
	sysCount := 0
	for _, m := range msgs {
		if m["role"] == "system" {
			sysCount++
			if m["content"] != prompt.Degraded {
				t.Errorf("retry system content=%v want prompt.Degraded", m["content"])
			}
		}
	}
	if sysCount != 1 {
		t.Errorf("retry want exactly 1 system message (Degraded), got %d (all=%v)", sysCount, msgs)
	}
}

// TestPromptEmptyModeDefaultsToPassthrough 空 PromptMode 必须落到 passthrough 语义，
// 与配置层 Default() 的实际默认一致。
//
// 关键：PromptText 故意设为哨兵值 "GATEWAY_SYS"。若空模式被误当作 custom，出站 system
// 会被替换为哨兵（客户端 system 丢失）；只有真正走 passthrough 才会保留客户端 system
// 且哨兵零出现。故本用例能区分两种语义，而不是"两种都无改写"的假通过。
func TestPromptEmptyModeDefaultsToPassthrough(t *testing.T) {
	var bodies [][]byte
	up := captureUpstream(&bodies, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	// PromptMode 留空；PromptText 非空以暴露"误判为 custom"的情形。
	h := NewHandler(Config{Pool: p, Upstream: up, PromptText: "GATEWAY_SYS"})

	body := `{"model":"glm-5.2","stream":true,"messages":[{"role":"system","content":"CLIENT_SYS"},{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if len(bodies) != 1 {
		t.Fatalf("upstream attempts=%d want 1", len(bodies))
	}
	msgs := outboundMessages(t, bodies[0])

	// 客户端 system 原样保留，且哨兵（custom 提示词）绝不出现。
	if got := countSystemContent(msgs, "CLIENT_SYS"); got != 1 {
		t.Errorf("client system preserved count=%d want 1 (msgs=%v)", got, msgs)
	}
	if got := countSystemContent(msgs, "GATEWAY_SYS"); got != 0 {
		t.Errorf("empty mode must NOT inject custom prompt, got %d (msgs=%v)", got, msgs)
	}
	// user 消息不受影响。
	if len(msgs) != 2 || msgs[1]["role"] != "user" || msgs[1]["content"] != "hi" {
		t.Errorf("unexpected messages: %v", msgs)
	}
}
