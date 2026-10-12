package alert

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// sink 捕获 webhook 收到的请求（原始字节 + 头），供签名与状态机断言。
type sink struct {
	mu   sync.Mutex
	body [][]byte
	hdr  []http.Header
	code int // 返回给调用方的状态码；0 = 200
}

func (s *sink) server(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.body = append(s.body, b)
		s.hdr = append(s.hdr, r.Header.Clone())
		code := s.code
		s.mu.Unlock()
		if code == 0 {
			code = http.StatusOK
		}
		w.WriteHeader(code)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (s *sink) events(t *testing.T) []Event {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Event, 0, len(s.body))
	for _, b := range s.body {
		var e Event
		if err := json.Unmarshal(b, &e); err != nil {
			t.Fatalf("请求体不是合法 JSON: %v (%s)", err, b)
		}
		out = append(out, e)
	}
	return out
}

func (s *sink) headers() []http.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]http.Header(nil), s.hdr...)
}

func (s *sink) raw() [][]byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([][]byte(nil), s.body...)
}

// newTestMonitor 构造一个时钟可控的监控器；healthy 由闭包读取，便于在测试中改。
func newTestMonitor(t *testing.T, cfg Config, healthy *int) (*Monitor, *time.Time) {
	t.Helper()
	total := 5
	now := time.Date(2026, 10, 12, 12, 0, 0, 0, time.UTC)
	clock := &now
	m := New("workbuddy2api", "test", func() Config { return cfg },
		func() (int, int) { return total, *healthy })
	m.now = func() time.Time { return *clock }
	return m, clock
}

// TestAlertThenCooldownThenRecovery 覆盖完整状态机：跌破即告警、冷却内不重复、
// 冷却后重复提醒、恢复后推 recovery 且状态复位。
func TestAlertThenCooldownThenRecovery(t *testing.T) {
	var s sink
	srv := s.server(t)
	healthy := 2
	m, clock := newTestMonitor(t, Config{
		Enabled: true, WebhookURL: srv.URL, Secret: "s3cr3t",
		MinHealthy: 1, Cooldown: 30 * time.Minute, Interval: time.Minute,
	}, &healthy)
	ctx := context.Background()

	// 1) 健康数达标：不告警。
	m.Check(ctx)
	if got := len(s.raw()); got != 0 {
		t.Fatalf("未跌破阈值不应推送，got %d 次", got)
	}

	// 2) 跌破（0 < 1）：立即告警。
	healthy = 0
	*clock = clock.Add(2 * time.Minute)
	m.Check(ctx)
	evs := s.events(t)
	if len(evs) != 1 || evs[0].Event != "alert" {
		t.Fatalf("应推出一条 alert，got %+v", evs)
	}
	if evs[0].Healthy != 0 || evs[0].Total != 5 || evs[0].Threshold != 1 {
		t.Fatalf("告警内容错误: %+v", evs[0])
	}
	if evs[0].Service != "workbuddy2api" || evs[0].Version != "test" {
		t.Fatalf("service/version 缺失: %+v", evs[0])
	}

	// 3) 冷却内重复检查：不重复推送。
	*clock = clock.Add(5 * time.Minute)
	m.Check(ctx)
	if got := len(s.raw()); got != 1 {
		t.Fatalf("冷却内不应重复推送，got %d 次", got)
	}

	// 4) 冷却到点：重复提醒。
	*clock = clock.Add(31 * time.Minute)
	m.Check(ctx)
	evs = s.events(t)
	if len(evs) != 2 || evs[1].Event != "alert" {
		t.Fatalf("冷却后应重复提醒，got %+v", evs)
	}

	// 5) 恢复：推 recovery 并复位。
	healthy = 3
	*clock = clock.Add(2 * time.Minute)
	m.Check(ctx)
	evs = s.events(t)
	if len(evs) != 3 || evs[2].Event != "recovery" {
		t.Fatalf("恢复应推 recovery，got %+v", evs)
	}
	// 复位后再跌破应重新告警（而不是因为 firing 残留而沉默）。
	healthy = 0
	*clock = clock.Add(2 * time.Minute)
	m.Check(ctx)
	evs = s.events(t)
	if len(evs) != 4 || evs[3].Event != "alert" {
		t.Fatalf("恢复后再次跌破应重新告警，got %+v", evs)
	}
}

// TestSignatureAndTimestampHeaders 锁定签名口径：接收方按
// hex(hmac_sha256(secret, raw_body)) 校验，且 X-Timestamp 与请求体 timestamp 一致。
func TestSignatureAndTimestampHeaders(t *testing.T) {
	var s sink
	srv := s.server(t)
	healthy := 0
	m, _ := newTestMonitor(t, Config{
		Enabled: true, WebhookURL: srv.URL, Secret: "top-secret",
		MinHealthy: 1, Cooldown: time.Hour, Interval: time.Minute,
	}, &healthy)

	m.Check(context.Background())

	hdrs := s.headers()
	raw := s.raw()
	if len(hdrs) != 1 || len(raw) != 1 {
		t.Fatalf("应恰好一次推送，got %d/%d", len(hdrs), len(raw))
	}
	want := "sha256=" + Signature("top-secret", raw[0])
	if got := hdrs[0].Get("X-Signature"); got != want {
		t.Fatalf("X-Signature=%q want %q", got, want)
	}
	if ct := hdrs[0].Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type=%q", ct)
	}
	var e Event
	if err := json.Unmarshal(raw[0], &e); err != nil {
		t.Fatal(err)
	}
	if got := hdrs[0].Get("X-Timestamp"); got != itoa(e.Timestamp) {
		t.Fatalf("X-Timestamp=%q 与请求体 timestamp=%d 不一致", got, e.Timestamp)
	}
}

// TestNoSecretNoSignature 锁定：未配密钥时不发签名头（内网简化用法），但时间戳仍在。
func TestNoSecretNoSignature(t *testing.T) {
	var s sink
	srv := s.server(t)
	healthy := 0
	m, _ := newTestMonitor(t, Config{
		Enabled: true, WebhookURL: srv.URL, MinHealthy: 1,
		Cooldown: time.Hour, Interval: time.Minute,
	}, &healthy)
	m.Check(context.Background())
	hdrs := s.headers()
	if len(hdrs) != 1 {
		t.Fatalf("应推送一次，got %d", len(hdrs))
	}
	if got := hdrs[0].Get("X-Signature"); got != "" {
		t.Fatalf("未配密钥不应有签名头，got %q", got)
	}
	if hdrs[0].Get("X-Timestamp") == "" {
		t.Fatal("X-Timestamp 应始终存在")
	}
}

// TestDisabledOrUnconfigured 锁定关闭/未配置时不推送，且状态复位。
func TestDisabledOrUnconfigured(t *testing.T) {
	var s sink
	srv := s.server(t)
	healthy := 0
	// enabled=false
	m, clock := newTestMonitor(t, Config{Enabled: false, WebhookURL: srv.URL, MinHealthy: 1, Interval: time.Minute}, &healthy)
	m.Check(context.Background())
	// enabled=true 但 URL 为空
	m2, _ := newTestMonitor(t, Config{Enabled: true, WebhookURL: "", MinHealthy: 1, Interval: time.Minute}, &healthy)
	m2.Check(context.Background())
	if got := len(s.raw()); got != 0 {
		t.Fatalf("关闭/未配置不应推送，got %d", got)
	}
	// 关闭状态下曾 firing 的，重新启用应重新告警（状态已复位）。
	m.firing = true
	m.Check(context.Background()) // enabled=false → 复位
	*clock = clock.Add(time.Minute)
	cfg := Config{Enabled: true, WebhookURL: srv.URL, MinHealthy: 1, Interval: time.Minute}
	m.cfg = func() Config { return cfg }
	m.Check(context.Background())
	if got := len(s.raw()); got != 1 {
		t.Fatalf("重新启用应重新告警，got %d", got)
	}
}

// TestIntervalGating 锁定 interval 门控：间隔内的 Check 不评估（不推送也不改状态）。
func TestIntervalGating(t *testing.T) {
	var s sink
	srv := s.server(t)
	healthy := 0
	m, clock := newTestMonitor(t, Config{
		Enabled: true, WebhookURL: srv.URL, MinHealthy: 1,
		Cooldown: time.Hour, Interval: 10 * time.Minute,
	}, &healthy)
	m.Check(context.Background())
	if len(s.raw()) != 1 {
		t.Fatal("首次应告警")
	}
	// 恢复但仍在 interval 窗口内：不应被评估到，因此不发 recovery。
	healthy = 5
	*clock = clock.Add(time.Minute)
	m.Check(context.Background())
	if got := len(s.raw()); got != 1 {
		t.Fatalf("interval 内不应评估，got %d 次推送", got)
	}
	// 越过 interval：评估到恢复，发 recovery。
	*clock = clock.Add(10 * time.Minute)
	m.Check(context.Background())
	evs := s.events(t)
	if len(evs) != 2 || evs[1].Event != "recovery" {
		t.Fatalf("越过 interval 应评估到恢复，got %+v", evs)
	}
}

// TestPostFailureKeepsStateAndRetries 锁定：推送失败**不切换状态**，下一轮重试。
// 告警被静默吞掉是告警系统最不能接受的失败方式。
func TestPostFailureKeepsStateAndRetries(t *testing.T) {
	var s sink
	srv := s.server(t)
	healthy := 0
	m, clock := newTestMonitor(t, Config{
		Enabled: true, WebhookURL: srv.URL, MinHealthy: 1,
		Cooldown: time.Hour, Interval: time.Minute,
	}, &healthy)

	s.mu.Lock()
	s.code = http.StatusInternalServerError
	s.mu.Unlock()
	m.Check(context.Background())
	if m.firing {
		t.Fatal("推送失败不应进入 firing 状态")
	}

	// 下一轮（越过 interval）服务端恢复 200 → 重试成功并进入 firing。
	s.mu.Lock()
	s.code = 0
	s.mu.Unlock()
	*clock = clock.Add(2 * time.Minute)
	m.Check(context.Background())
	if !m.firing {
		t.Fatal("重试成功后应进入 firing 状态")
	}
	evs := s.events(t)
	if len(evs) != 2 || evs[0].Event != "alert" || evs[1].Event != "alert" {
		t.Fatalf("应推送两次 alert（首次失败后重试），got %+v", evs)
	}
}

// TestRunStopsOnContextCancel 锁定 Run 随 ctx 结束退出（不泄漏协程）。
func TestRunStopsOnContextCancel(t *testing.T) {
	var s sink
	srv := s.server(t)
	healthy := 5
	m, _ := newTestMonitor(t, Config{
		Enabled: true, WebhookURL: srv.URL, MinHealthy: 1,
		Cooldown: time.Hour, Interval: time.Minute,
	}, &healthy)
	m.tick = 5 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { m.Run(ctx); close(done) }()
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Run 未随 ctx 取消退出")
	}
}

func itoa(n int64) string {
	b, _ := json.Marshal(n)
	return string(b)
}
