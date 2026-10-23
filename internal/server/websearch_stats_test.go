package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

func TestWebSearchStatsEndpointTracksCacheAndEngine(t *testing.T) {
	const querySSE = "data: {\"choices\":[{\"delta\":{\"content\":\"Go 1.24 release notes\"}}]}\n\ndata: [DONE]\n\n"
	up, calls := newRegressionUpstream(func(n int, _ string, _ []byte) (int, string, bool) {
		if n == 1 {
			return http.StatusOK, querySSE, true
		}
		return http.StatusOK, regNonStreamSSE, true
	})
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	client, queries := newSearxngCapture(t, "https://go.dev/doc/go1.24")
	h := NewHandler(Config{
		Pool: p, Upstream: up, WebSearch: client,
		WebSearchQueryMode: "model", APIKey: "panel-key",
	})

	body := `{"model":"glm-5.2","max_tokens":100,"tools":[{"type":"web_search_20260209","name":"web_search"}],"messages":[{"role":"user","content":"Go 1.24 有哪些新特性？"}]}`
	for i := 0; i < 2; i++ {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer panel-key")
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("search request %d: code=%d body=%s", i+1, rec.Code, rec.Body)
		}
	}
	if len(*queries) != 1 || (*queries)[0] != "Go 1.24 release notes" {
		t.Fatalf("engine queries=%v, want one generated query", *queries)
	}
	if len(calls()) != 3 {
		t.Fatalf("upstream calls=%d, want query generation + two answers", len(calls()))
	}

	req := httptest.NewRequest("GET", "/panel/api/search_stats", nil)
	req.Header.Set("Authorization", "Bearer panel-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("stats code=%d body=%s", rec.Code, rec.Body)
	}
	var got webSearchStatsSnapshot
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.Enabled || got.Backend != "searxng" || got.QueryMode != "model" {
		t.Fatalf("identity=%+v", got)
	}
	if got.Queries != 2 || got.Cache.Hits != 1 || got.Cache.Misses != 1 || got.Cache.ActiveEntries != 1 {
		t.Fatalf("cache stats=%+v queries=%d", got.Cache, got.Queries)
	}
	if len(got.Engines) != 1 {
		t.Fatalf("engines=%+v", got.Engines)
	}
	engine := got.Engines[0]
	if engine.Backend != "searxng" || engine.Status != "healthy" ||
		engine.Attempts != 1 || engine.Successes != 1 || engine.Failures != 0 {
		t.Fatalf("engine health=%+v", engine)
	}
	if len(got.Recent) != 2 || !got.Recent[0].CacheHit || got.Recent[1].CacheHit {
		t.Fatalf("recent must be newest-first with one cache hit: %+v", got.Recent)
	}
}

func TestWebSearchStatsEndpointAuthAndDisabledState(t *testing.T) {
	h := NewHandler(Config{APIKey: "panel-key"})

	unauth := httptest.NewRecorder()
	h.ServeHTTP(unauth, httptest.NewRequest("GET", "/panel/api/search_stats", nil))
	if unauth.Code != http.StatusUnauthorized {
		t.Fatalf("without key code=%d body=%s", unauth.Code, unauth.Body)
	}

	req := httptest.NewRequest("GET", "/panel/api/search_stats", nil)
	req.Header.Set("Authorization", "Bearer panel-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("with key code=%d body=%s", rec.Code, rec.Body)
	}
	var got webSearchStatsSnapshot
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Enabled || got.Queries != 0 || len(got.Engines) != 0 || len(got.Recent) != 0 {
		t.Fatalf("disabled stats=%+v", got)
	}
}

func TestWebSearchTelemetryHealthTransitions(t *testing.T) {
	tel := newWebSearchTelemetry(2)
	for i := 0; i < 3; i++ {
		tel.recordAttempt("searxng", "q", 0, time.Millisecond, errors.New("backend unavailable"))
	}
	engines, recent := tel.snapshot("searxng", true)
	if len(engines) != 1 || engines[0].Status != "down" || engines[0].ConsecutiveFailures != 3 {
		t.Fatalf("engine health=%+v", engines)
	}
	if len(recent) != 2 || recent[0].Time < recent[1].Time {
		t.Fatalf("bounded recent events must stay newest-first: %+v", recent)
	}
}
