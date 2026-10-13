package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/livecfg"
	"github.com/linguo2625469/workbuddy2api-panel/internal/reqlog"
	"github.com/linguo2625469/workbuddy2api-panel/internal/usage"
)

func getMetrics(h *Handler, key string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("GET", "/metrics", nil)
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	return rec
}

// TestMetricsDisabledReturns404 锁定默认关闭：未启用时该路径表现为"不存在"，
// 而不是 501——端口扫描器不该从状态码区分出这是一个真实端点。
func TestMetricsDisabledReturns404(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	if rec := getMetrics(h, ""); rec.Code != http.StatusNotFound {
		t.Fatalf("未启用应 404，got %d body=%s", rec.Code, rec.Body)
	}
}

// TestMetricsEnabledRendersPoolAndUsage 锁定启用后的内容：build_info 带版本、
// 账号池状态、realm 可服务性，以及 usage 累计量。
func TestMetricsEnabledRendersPoolAndUsage(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	rec := usage.New("")
	rec.Add(time.Now(), "cn", "u1", "glm-5.2", usage.Delta{
		PromptTokens: 100, HasPromptTokens: true,
		CompletionTokens: 20, HasCompletion: true,
		TotalTokens: 120, HasTotal: true,
	}, true)
	h := NewHandler(Config{
		Pool: p, Upstream: up, Usage: rec, RequestLog: reqlog.New(reqlog.Config{}),
		Version: "9.9.9",
		Live:    livecfg.New(livecfg.Snapshot{MetricsEnabled: true}),
	})

	out := getMetrics(h, "")
	if out.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", out.Code, out.Body)
	}
	if ct := out.Header().Get("Content-Type"); !strings.Contains(ct, "text/plain") {
		t.Fatalf("Content-Type=%q 应为 Prometheus 文本格式", ct)
	}
	body := out.Body.String()
	for _, want := range []string{
		`workbuddy_build_info{version="9.9.9"} 1`,
		`workbuddy_accounts{state="total"} 1`,
		`workbuddy_account_credits{uid="u1"`,
		`workbuddy_account_healthy{uid="u1"} 1`,
		`workbuddy_realm_servable{realm="cn"}`,
		`workbuddy_usage_tokens_total{kind="total"} 120`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("缺少 %q\n---\n%s", want, body)
		}
	}
}

// TestMetricsRequiresAuth 锁定鉴权口径与 /status 一致：配了密钥就必须带。
func TestMetricsRequiresAuth(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{
		Pool: p, Upstream: up,
		Live: livecfg.New(livecfg.Snapshot{APIKey: "k1", MetricsEnabled: true}),
	})

	if rec := getMetrics(h, ""); rec.Code != http.StatusUnauthorized {
		t.Fatalf("缺密钥应 401，got %d", rec.Code)
	}
	if rec := getMetrics(h, "wrong"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("错误密钥应 401，got %d", rec.Code)
	}
	if rec := getMetrics(h, "k1"); rec.Code != http.StatusOK {
		t.Fatalf("正确密钥应 200，got %d body=%s", rec.Code, rec.Body)
	}
}
