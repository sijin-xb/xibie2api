package metrics

import (
	"strings"
	"testing"
)

func TestWriteSearchMetrics(t *testing.T) {
	snap := Snapshot{
		SearchEnabled:      true,
		SearchQueries:      7,
		SearchCacheHits:    3,
		SearchCacheMisses:  4,
		SearchCacheExpired: 1,
		SearchCacheEvicted: 0,
		SearchActiveKeys:   2,
		SearchEngines: []SearchEngineMetric{{
			Backend: "searxng", Attempts: 4, Successes: 4, Failures: 0,
			ConsecutiveFailures: 0, AvgLatencySeconds: 1.5, Healthy: true,
		}},
	}
	var sb strings.Builder
	if err := Write(&sb, snap); err != nil {
		t.Fatalf("Write: %v", err)
	}
	out := sb.String()
	for _, want := range []string{
		`workbuddy_search_cache_events_total{kind="hit"} 3`,
		`workbuddy_search_cache_events_total{kind="miss"} 4`,
		`workbuddy_search_cache_events_total{kind="expired"} 1`,
		`workbuddy_search_cache_active_keys 2`,
		`workbuddy_search_engine_attempts_total{backend="searxng"} 4`,
		`workbuddy_search_engine_latency_seconds_avg{backend="searxng"} 1.5`,
		`workbuddy_search_engine_healthy{backend="searxng"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in output:\n%s", want, out)
		}
	}
	if n := strings.Count(out, "# HELP workbuddy_search_cache_events_total"); n != 1 {
		t.Fatalf("duplicate HELP lines for cache events: %d", n)
	}
}
