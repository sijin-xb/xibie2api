package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// recordingRenderer 是 turnRenderer 的测试替身：记录哪个渲染入口被调用，并写出
// 可辨识的标记正文——以此证明 runTurn 的线上输出确实经由注入的 renderer 产生，
// 而非内联的 OpenAI 写出函数（边界是「活」的，不是死抽象）。
type recordingRenderer struct {
	streamCalled  bool
	nonStreamResp map[string]any
	errStatus     int
	errCode       string
	errMsg        string
	errHint       string
}

func (r *recordingRenderer) renderStream(w http.ResponseWriter, _ io.Reader,
	_ func() upstream.HintContext, _ *upstream.MarkupRepair, _ func(string)) error {
	r.streamCalled = true
	return nil
}

func (r *recordingRenderer) renderNonStream(w http.ResponseWriter, resp map[string]any) bool {
	r.nonStreamResp = resp
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("CUSTOM_OK"))
	return true
}

func (r *recordingRenderer) renderError(w http.ResponseWriter, status int, code, msg, hint string) {
	r.errStatus, r.errCode, r.errMsg, r.errHint = status, code, msg, hint
	w.WriteHeader(status)
	_, _ = w.Write([]byte("CUSTOM_ERROR"))
}

func newTurnParamsForTest(body string, stream bool, r turnRenderer) *turnParams {
	return &turnParams{
		stream:        stream,
		rawModel:      "glm-5.2",
		bareModel:     "glm-5.2",
		realm:         "",
		clientIP:      "",
		body:          []byte(body),
		declaredTools: map[string]bool{},
		chatMeta:      upstream.ChatMeta{},
		renderer:      r,
	}
}

func TestRunTurnNonStreamUsesInjectedRenderer(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	r := &recordingRenderer{}
	st := newChatStat(time.Now(), []byte(`{"model":"glm-5.2","messages":[]}`), false)
	defer st.done()
	rec := httptest.NewRecorder()
	h.runTurn(context.Background(), rec, st, newTurnParamsForTest(`{"model":"glm-5.2","messages":[]}`, false, r))

	if r.streamCalled {
		t.Fatal("non-stream turn must not call renderStream")
	}
	if r.nonStreamResp == nil {
		t.Fatal("injected renderer.renderNonStream was not called")
	}
	if _, ok := r.nonStreamResp["choices"]; !ok {
		t.Fatalf("aggregated response missing choices: %v", r.nonStreamResp)
	}
	if got := rec.Body.String(); got != "CUSTOM_OK" {
		t.Fatalf("injected renderer did not own the wire output: body=%q", got)
	}
}

func TestRunTurnTerminalErrorUsesInjectedRenderer(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 500, `{"error":"boom"}`, false })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, MaxRotate: 1})

	r := &recordingRenderer{}
	st := newChatStat(time.Now(), []byte(`{"model":"glm-5.2","messages":[]}`), false)
	defer st.done()
	rec := httptest.NewRecorder()
	h.runTurn(context.Background(), rec, st, newTurnParamsForTest(`{"model":"glm-5.2","messages":[]}`, false, r))

	if r.errStatus == 0 {
		t.Fatal("injected renderer.renderError was not called")
	}
	if got := rec.Body.String(); got != "CUSTOM_ERROR" {
		t.Fatalf("injected renderer did not own the error wire output: body=%q", got)
	}
	if rec.Code != r.errStatus {
		t.Fatalf("status mismatch: recorder=%d renderer=%d", rec.Code, r.errStatus)
	}
}

func TestRunTurnStreamUsesInjectedRenderer(t *testing.T) {
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true })
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up})

	r := &recordingRenderer{}
	st := newChatStat(time.Now(), []byte(`{"model":"glm-5.2","messages":[]}`), true)
	defer st.done()
	rec := httptest.NewRecorder()
	h.runTurn(context.Background(), rec, st, newTurnParamsForTest(`{"model":"glm-5.2","messages":[]}`, true, r))

	if !r.streamCalled {
		t.Fatal("stream turn must call renderStream")
	}
	if r.nonStreamResp != nil || r.errStatus != 0 {
		t.Fatal("stream turn must not call renderNonStream/renderError")
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("injected renderStream wrote nothing, so wire body must be empty: %q", rec.Body.String())
	}
}
