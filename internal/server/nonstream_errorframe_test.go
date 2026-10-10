// nonstream_errorframe_test.go —— 非流式 error 帧、上游读错误、hasImagePart 的回归。
//
// P1：非流式路径此前无 error 帧检测，上游「200 + 仅 error 帧」被 Aggregate 聚合成功，
// 掩蔽成空 chat.completion（假成功）。本文件对 chat / responses / messages 三协议各
// 钉一个「仅 error 帧 → 错误响应（非 200+空）」用例，并保留「正常流仍聚合成功」的对照。
//
// P3：runTurn 流式分支区分「客户端写失败」与「上游读错误」——后者不得记成功/绑粘性。
//
// P4：hasImagePart 对「字符串 content + 数组 image_url」混排必须判真。
package server

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/session"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// errOnlyRateLimitSSE：上游 200 但只回一帧 error（6004 限流），没有任何正文帧。
const errOnlyRateLimitSSE = "data: {\"error\":{\"message\":\"rate limited\",\"code\":6004}}\n\ndata: [DONE]\n\n"

// ---------------------------------------------------------------------------
// P1：非流式 error 帧 → 错误响应，而非 200+空成功
// ---------------------------------------------------------------------------

func TestNonStreamChatErrorFrameOnly(t *testing.T) {
	up, _ := staticRegressionUpstream(200, errOnlyRateLimitSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d body=%s (want 429; error-only stream must NOT be a 200+empty success)", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	errObj, _ := obj["error"].(map[string]any)
	if errObj == nil || errObj["code"] != "rate_limit_exceeded" {
		t.Fatalf("error envelope=%v", obj)
	}
	if st, _ := p.Status("u1"); st.SuccessCount != 0 {
		t.Fatalf("error-frame path must NOT NoteSuccess: %+v", st)
	}
}

func TestNonStreamResponsesErrorFrameOnly(t *testing.T) {
	up, _ := staticRegressionUpstream(200, errOnlyRateLimitSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","input":"hi"}`)))

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d body=%s (want 429)", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	errObj, _ := obj["error"].(map[string]any)
	if errObj == nil || errObj["code"] != "rate_limit_exceeded" {
		t.Fatalf("error envelope=%v", obj)
	}
}

func TestNonStreamMessagesErrorFrameOnly(t *testing.T) {
	up, _ := staticRegressionUpstream(200, errOnlyRateLimitSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"glm-5.2","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d body=%s (want 429)", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["type"] != "error" {
		t.Fatalf("envelope=%v", obj)
	}
	errObj, _ := obj["error"].(map[string]any)
	if errObj == nil || errObj["type"] != "rate_limit_error" {
		t.Fatalf("error=%v", errObj)
	}
}

// TestNonStreamChatNormalStillSuccess 对照：正常 SSE 流仍聚合为 chat.completion 200。
func TestNonStreamChatNormalStillSuccess(t *testing.T) {
	up, _ := staticRegressionUpstream(200, regNonStreamSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	obj := mustDecodeJSON(t, rec.Body.Bytes())
	if obj["object"] != "chat.completion" {
		t.Fatalf("object=%v", obj["object"])
	}
	msg := obj["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "你好，世界" {
		t.Fatalf("content=%q", msg["content"])
	}
	if st, _ := p.Status("u1"); st.SuccessCount == 0 {
		t.Fatalf("normal non-stream success must NoteSuccess: %+v", st)
	}
}

// ---------------------------------------------------------------------------
// P3：上游读错误 ≠ 客户端断连
// ---------------------------------------------------------------------------

// TestChatStreamUpstreamReadErrorNotMarkedSuccess 上游流中途读错误 → 账号不得被记成功、
// 不得绑粘性（此前读错误与客户端写失败共用同一 error 值，被误判为断连而记成功）。
func TestChatStreamUpstreamReadErrorNotMarkedSuccess(t *testing.T) {
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body: io.NopCloser(&errAfterReader{
					data: []byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n"),
					err:  errors.New("upstream read failed"),
				}),
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	st := newBindStore()
	sess := session.New(session.Config{
		TTL:       time.Minute,
		Store:     st,
		Available: func() []string { return []string{"u1"} },
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, Session: sess, SoftCooldown: time.Minute})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}],"metadata":{"conversation_id":"conv-1"}}`)))

	body := rec.Body.String()
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("upstream read error must not emit a success [DONE] terminal: %q", body)
	}
	if st2, _ := p.Status("u1"); st2.SuccessCount != 0 {
		t.Fatalf("upstream read error must NOT be marked as account success: %+v", st2)
	}
	if uid, ok := st.lastUID("conv-1"); ok {
		t.Fatalf("upstream read error must NOT bind stickiness, got %s", uid)
	}
}

// ---------------------------------------------------------------------------
// P4：hasImagePart 混排
// ---------------------------------------------------------------------------

func TestHasImagePartMixedContent(t *testing.T) {
	cases := []struct {
		name string
		body string
		want bool
	}{
		{
			"string content + array image part",
			`{"messages":[{"role":"system","content":"sys"},{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"x"}}]}]}`,
			true,
		},
		{
			"array only image part",
			`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}`,
			true,
		},
		{
			"no image part",
			`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			false,
		},
		{
			"string content only",
			`{"messages":[{"role":"user","content":"hi"}]}`,
			false,
		},
	}
	for _, c := range cases {
		if got := hasImagePart([]byte(c.body)); got != c.want {
			t.Errorf("%s: hasImagePart=%v want %v", c.name, got, c.want)
		}
	}
}
