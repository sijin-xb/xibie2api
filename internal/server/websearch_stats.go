package server

import (
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// webSearchCacheSnapshot 是 TTL 缓存的只读观测口径。所有计数均从进程启动时
// 开始累计；重启后清零。
type webSearchCacheSnapshot struct {
	TTLSeconds    int64   `json:"ttl_seconds"`
	MaxKeys       int     `json:"max_keys"`
	ActiveEntries int     `json:"active_entries"`
	Hits          int64   `json:"hits"`
	Misses        int64   `json:"misses"`
	Expired       int64   `json:"expired"`
	Evicted       int64   `json:"evicted"`
	HitRate       float64 `json:"hit_rate"`
}

// webSearchEngineHealth 是按检索后端聚合的健康度。缓存命中不会增加 attempts：
// 它没有访问搜索引擎，只用于判断缓存对上游请求的削减程度。
type webSearchEngineHealth struct {
	Backend             string  `json:"backend"`
	Status              string  `json:"status"`
	Attempts            int64   `json:"attempts"`
	Successes           int64   `json:"successes"`
	Failures            int64   `json:"failures"`
	EmptyResults        int64   `json:"empty_results"`
	ConsecutiveFailures int     `json:"consecutive_failures"`
	SuccessRate         float64 `json:"success_rate"`
	AverageLatencyMs    int64   `json:"average_latency_ms"`
	LastLatencyMs       int64   `json:"last_latency_ms"`
	LastAttemptAt       string  `json:"last_attempt_at,omitempty"`
	LastSuccessAt       string  `json:"last_success_at,omitempty"`
	LastFailureAt       string  `json:"last_failure_at,omitempty"`
	LastError           string  `json:"last_error,omitempty"`
}

// webSearchLogEvent 是面向面板的最近检索事件。query/error 都截断并压成单行，
// 避免客户端可控文本撑大常驻内存或破坏表格展示。
type webSearchLogEvent struct {
	Time      string `json:"time"`
	Backend   string `json:"backend"`
	Query     string `json:"query"`
	CacheHit  bool   `json:"cache_hit"`
	Success   bool   `json:"success"`
	Results   int    `json:"results"`
	LatencyMs int64  `json:"latency_ms"`
	Error     string `json:"error,omitempty"`
}

type webSearchStatsSnapshot struct {
	Enabled   bool                    `json:"enabled"`
	Backend   string                  `json:"backend,omitempty"`
	QueryMode string                  `json:"query_mode"`
	Queries   int64                   `json:"queries"`
	Cache     webSearchCacheSnapshot  `json:"cache"`
	Engines   []webSearchEngineHealth `json:"engines"`
	Recent    []webSearchLogEvent     `json:"recent"`
}

type webSearchEngineState struct {
	attempts            int64
	successes           int64
	failures            int64
	emptyResults        int64
	consecutiveFailures int
	totalLatencyMs      int64
	lastLatencyMs       int64
	lastAttemptAt       time.Time
	lastSuccessAt       time.Time
	lastFailureAt       time.Time
	lastError           string
}

// webSearchTelemetry 保存检索日志聚合与有界最近事件。缓存自身负责命中/未命中
// 计数；这里只记录每次有效查询对应的一个事件，以及真正打到引擎的尝试。
type webSearchTelemetry struct {
	mu       sync.Mutex
	capacity int
	engines  map[string]*webSearchEngineState
	recent   []webSearchLogEvent
}

func newWebSearchTelemetry(capacity int) *webSearchTelemetry {
	if capacity <= 0 {
		capacity = 30
	}
	return &webSearchTelemetry{
		capacity: capacity,
		engines:  make(map[string]*webSearchEngineState),
		recent:   make([]webSearchLogEvent, 0, capacity),
	}
}

func (t *webSearchTelemetry) recordCacheHit(backend, query string, hit webSearchCacheEntry) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.appendLocked(webSearchLogEvent{
		Time:     time.Now().Format(time.RFC3339Nano),
		Backend:  backend,
		Query:    searchDisplayText(query, 300),
		CacheHit: true,
		Success:  hit.err == "",
		Results:  len(hit.results),
		Error:    searchDisplayText(hit.err, 500),
	})
}

func (t *webSearchTelemetry) recordAttempt(backend, query string, results int, latency time.Duration, err error) {
	if t == nil {
		return
	}
	now := time.Now()
	latencyMs := latency.Milliseconds()
	if latency > 0 && latencyMs == 0 {
		latencyMs = 1
	}
	errText := ""
	if err != nil {
		errText = err.Error()
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.engines[backend]
	if st == nil {
		st = &webSearchEngineState{}
		t.engines[backend] = st
	}
	st.attempts++
	st.lastAttemptAt = now
	st.lastLatencyMs = latencyMs
	st.totalLatencyMs += latencyMs
	if err != nil {
		st.failures++
		st.consecutiveFailures++
		st.lastFailureAt = now
		st.lastError = searchDisplayText(errText, 500)
	} else {
		st.successes++
		st.consecutiveFailures = 0
		st.lastSuccessAt = now
		if results == 0 {
			st.emptyResults++
		}
	}
	t.appendLocked(webSearchLogEvent{
		Time:      now.Format(time.RFC3339Nano),
		Backend:   backend,
		Query:     searchDisplayText(query, 300),
		Success:   err == nil,
		Results:   results,
		LatencyMs: latencyMs,
		Error:     searchDisplayText(errText, 500),
	})
}

func (t *webSearchTelemetry) appendLocked(event webSearchLogEvent) {
	if len(t.recent) >= t.capacity {
		copy(t.recent, t.recent[1:])
		t.recent[len(t.recent)-1] = event
		return
	}
	t.recent = append(t.recent, event)
}

func (t *webSearchTelemetry) snapshot(activeBackend string, enabled bool) ([]webSearchEngineHealth, []webSearchLogEvent) {
	if t == nil {
		return []webSearchEngineHealth{}, []webSearchLogEvent{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	names := make([]string, 0, len(t.engines)+1)
	seen := make(map[string]bool, len(t.engines)+1)
	if activeBackend != "" {
		names = append(names, activeBackend)
		seen[activeBackend] = true
	}
	for name := range t.engines {
		if !seen[name] {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	if activeBackend != "" {
		for i, name := range names {
			if name == activeBackend {
				copy(names[1:i+1], names[:i])
				names[0] = name
				break
			}
		}
	}

	engines := make([]webSearchEngineHealth, 0, len(names))
	for _, name := range names {
		st := t.engines[name]
		h := webSearchEngineHealth{Backend: name, Status: "unknown"}
		if st != nil {
			h.Attempts = st.attempts
			h.Successes = st.successes
			h.Failures = st.failures
			h.EmptyResults = st.emptyResults
			h.ConsecutiveFailures = st.consecutiveFailures
			h.LastLatencyMs = st.lastLatencyMs
			if st.attempts > 0 {
				h.SuccessRate = float64(st.successes) / float64(st.attempts) * 100
				h.AverageLatencyMs = st.totalLatencyMs / st.attempts
			}
			h.LastAttemptAt = formatSearchTime(st.lastAttemptAt)
			h.LastSuccessAt = formatSearchTime(st.lastSuccessAt)
			h.LastFailureAt = formatSearchTime(st.lastFailureAt)
			h.LastError = st.lastError
			switch {
			case !enabled:
				h.Status = "disabled"
			case st.consecutiveFailures >= 3:
				h.Status = "down"
			case st.consecutiveFailures > 0:
				h.Status = "degraded"
			case st.attempts > 0:
				h.Status = "healthy"
			}
		} else if !enabled {
			h.Status = "disabled"
		}
		engines = append(engines, h)
	}

	recent := make([]webSearchLogEvent, len(t.recent))
	copy(recent, t.recent)
	// 面板按新到旧展示；内部环形追加保持 O(1) 写入。
	for i, j := 0, len(recent)-1; i < j; i, j = i+1, j-1 {
		recent[i], recent[j] = recent[j], recent[i]
	}
	return engines, recent
}

func formatSearchTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.Format(time.RFC3339Nano)
}

func searchDisplayText(s string, limit int) string {
	s = strings.Join(strings.Fields(s), " ")
	if limit <= 0 || len(s) <= limit {
		return s
	}
	return strings.ToValidUTF8(s[:limit], "") + "…"
}

func (h *Handler) webSearchStatsSnapshot() webSearchStatsSnapshot {
	c := h.cfg.WebSearch
	enabled := c != nil && c.Enabled()
	backend := ""
	if c != nil {
		backend = c.Backend()
	}
	cache := h.searchCache.snapshot(time.Now())
	engines, recent := h.searchTelemetry.snapshot(backend, enabled)
	queryMode := h.cfg.WebSearchQueryMode
	if queryMode == "" {
		queryMode = "raw"
	}
	return webSearchStatsSnapshot{
		Enabled:   enabled,
		Backend:   backend,
		QueryMode: queryMode,
		Queries:   cache.Hits + cache.Misses,
		Cache:     cache,
		Engines:   engines,
		Recent:    recent,
	}
}

func (h *Handler) webSearchStatsHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	writeJSON(w, http.StatusOK, h.webSearchStatsSnapshot())
}
