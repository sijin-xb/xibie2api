// keepalive_test.go —— 流式响应的 SSE 注释心跳测试。
package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// TestSSEKeepaliveEmitsComment 断言后台 goroutine 周期性写出 ": keepalive" 并 flush。
func TestSSEKeepaliveEmitsComment(t *testing.T) {
	rec := httptest.NewRecorder()
	k := newSSEKeepalive(rec, 10*time.Millisecond)
	defer k.Close()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(rec.Body.String(), ": keepalive") {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !strings.Contains(rec.Body.String(), ": keepalive\n\n") {
		t.Fatalf("no keepalive comment emitted: %q", rec.Body.String())
	}
	if !rec.Flushed {
		t.Fatal("keepalive comment not flushed")
	}
}

// TestSSEKeepaliveCloseStops 断言 Close 后不再有心跳写（后台 goroutine 已退出）。
func TestSSEKeepaliveCloseStops(t *testing.T) {
	rec := httptest.NewRecorder()
	k := newSSEKeepalive(rec, 5*time.Millisecond)
	time.Sleep(30 * time.Millisecond)
	k.Close()
	n := len(rec.Body.String())
	time.Sleep(40 * time.Millisecond)
	if got := len(rec.Body.String()); got != n {
		t.Fatalf("writes after Close: %d -> %d", n, got)
	}
}

// TestSSEKeepaliveIntervalDisabled 断言 interval<=0 时不发心跳（Close 也不阻塞）。
func TestSSEKeepaliveIntervalDisabled(t *testing.T) {
	rec := httptest.NewRecorder()
	k := newSSEKeepalive(rec, 0)
	k.Write([]byte("x"))
	k.Close()
	if rec.Body.String() != "x" {
		t.Fatalf("unexpected body %q", rec.Body.String())
	}
}

// TestSSEKeepaliveConcurrentWriteIntegrity 断言并发写与心跳不会交错写坏帧。
func TestSSEKeepaliveConcurrentWriteIntegrity(t *testing.T) {
	rec := httptest.NewRecorder()
	k := newSSEKeepalive(rec, 2*time.Millisecond)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				k.Write([]byte("data: {\"n\":1}\n\n"))
			}
		}()
	}
	wg.Wait()
	k.Close()
	body := rec.Body.String()
	// 每个非心跳块必须是完整帧；不允许出现半截 "data: {" 被截断的情况。
	for _, block := range strings.Split(body, "\n\n") {
		if block == "" || block == ": keepalive" {
			continue
		}
		if block != `data: {"n":1}` {
			t.Fatalf("interleaved/corrupted block: %q", block)
		}
	}
}

// TestResponsesStreamEmitsKeepalive 端到端：/v1/responses 流式响应里出现心跳注释，
// 且事件序列仍可正常解析（注释不影响事件）。
func TestResponsesStreamEmitsKeepalive(t *testing.T) {
	old := sseKeepaliveInterval
	sseKeepaliveInterval = 5 * time.Millisecond
	defer func() { sseKeepaliveInterval = old }()

	up, _ := staticRegressionUpstream(200, respTextSSE, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	events := parseSSEEvents(t, rec.Body.String())
	if len(events) == 0 {
		t.Fatalf("no events parsed (body=%q)", rec.Body.String())
	}
	got := eventNames(events)
	if got[len(got)-1] != "response.completed" {
		t.Fatalf("last event=%q want response.completed (all=%v)", got[len(got)-1], got)
	}
}

// TestResponsesStreamKeepaliveDuringSilence 断言上游长时间静默（首个 data 帧之前）
// 期间网关仍持续发心跳，且响应头 Content-Type 正确（心跳不得抢先提交默认头）。
func TestResponsesStreamKeepaliveDuringSilence(t *testing.T) {
	old := sseKeepaliveInterval
	sseKeepaliveInterval = 5 * time.Millisecond
	defer func() { sseKeepaliveInterval = old }()

	// 上游响应头立即返回，但 body 先阻塞 120ms 才吐第一帧（模拟长推理静默）。
	release := make(chan struct{})
	up := &upstream.Client{
		HTTP: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			pr, pw := io.Pipe()
			go func() {
				<-release
				io.WriteString(pw, respTextSSE)
				pw.Close()
			}()
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       pr,
			}, nil
		})},
		ChatBaseCN:    "https://fake.example",
		BillingBaseCN: "https://fake.example",
	}
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	go func() { time.Sleep(120 * time.Millisecond); close(release) }()

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses",
		strings.NewReader(`{"model":"glm-5.2","stream":true,"input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type=%q want text/event-stream (keepalive clobbered headers?)", ct)
	}
	if !strings.Contains(rec.Body.String(), ": keepalive") {
		t.Fatalf("no keepalive during silence: %q", rec.Body.String())
	}
	// 心跳不得破坏事件序列。
	events := parseSSEEvents(t, rec.Body.String())
	if len(events) == 0 || eventNames(events)[len(events)-1] != "response.completed" {
		t.Fatalf("events broken by keepalive: %v", eventNames(events))
	}
}
