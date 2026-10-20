package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// 上游 14003 实测原文（无重置墙钟、无 Retry-After，仅 retry_policy）。
const body14003 = `{"code":14003,"msg":"too many requests","requestId":"abc","retry_policy":{"retry":true},"displayMsg":{"en":"Too many requests. Please retry later.","zh":"请求过于频繁，请稍后重试"}}`

// TestIsSharedDimensionRateLimit 判定边界：仅「非 6004 + 无重置墙钟 + 无 Retry-After」
// 的账号级 429 参与共享维度判定。
func TestIsSharedDimensionRateLimit(t *testing.T) {
	h := NewHandler(Config{})
	cases := []struct {
		name string
		body string
		uerr *upstream.Error
		want bool
	}{
		{"14003 无恢复时刻", body14003, nil, true},
		{"14003 + Retry-After 头", body14003, &upstream.Error{RetryAfter: 30 * time.Second}, false},
		{"6004 模型级", `{"code":6004,"msg":"model usage limit exceeded"}`, nil, false},
		{"带重置墙钟", `{"code":1,"msg":"将在 2026-10-13 22:00:00 重置"}`, nil, false},
		{"空 body 兜底", "", nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := h.isSharedDimensionRateLimit(tc.body, tc.uerr); got != tc.want {
				t.Errorf("isSharedDimensionRateLimit(%q)=%v want %v", tc.body, got, tc.want)
			}
		})
	}
}

// TestRateIPGateActivatesOnTwoUIDs 单号反复命中永不激活；两个不同号在窗内命中即激活，
// 且激活时对窗内各 uid 调用 revoke 撤销误记冷却。
func TestRateIPGateActivatesOnTwoUIDs(t *testing.T) {
	var g rateIPGate
	revoked := map[string]int{}
	revoke := func(uid string) bool { revoked[uid]++; return true }

	if g.note("u1", revoke) {
		t.Fatal("单号首次命中不应激活")
	}
	if g.note("u1", revoke) {
		t.Fatal("同号重复命中不累计不同号数，不应激活")
	}
	if g.active() {
		t.Fatal("尚未激活")
	}
	if !g.note("u2", revoke) {
		t.Fatal("第二个不同号命中应激活")
	}
	if !g.active() {
		t.Fatal("激活后 active() 应为 true")
	}
	if revoked["u1"] != 1 || revoked["u2"] != 1 {
		t.Errorf("激活时应撤销窗内各号: revoked=%v", revoked)
	}
	// 激活期内新命中不续期、不重复撤销。
	if !g.note("u3", revoke) {
		t.Fatal("激活期内命中应返回 true")
	}
	if revoked["u3"] != 0 {
		t.Errorf("激活期内不应再撤销: revoked=%v", revoked)
	}
}

// TestRateIPGateWindowExpiry 激活到期后需全新命中重新判定。
func TestRateIPGateWindowExpiry(t *testing.T) {
	old := rateIPWindow
	rateIPWindow = 20 * time.Millisecond
	defer func() { rateIPWindow = old }()

	var g rateIPGate
	g.note("u1", nil)
	g.note("u2", nil) // 激活
	time.Sleep(30 * time.Millisecond)
	if g.active() {
		t.Fatal("窗口过后应解除")
	}
	if g.note("u3", nil) {
		t.Fatal("解除后单个新命中不应立即激活")
	}
}

// TestChat14003SharedDimensionDoesNotCoolPool 端到端：两个账号接连命中 14003（无恢复
// 时刻）→ 判定共享维度 → 轮转 fail-fast 终止，且两个账号都不被逐号软冷却（回溯撤销）。
func TestChat14003SharedDimensionDoesNotCoolPool(t *testing.T) {
	oldBackoff := rotateBackoffBase
	rotateBackoffBase = 0 // 跳过轮转退避，加速测试
	defer func() { rotateBackoffBase = oldBackoff }()

	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		return http.StatusTooManyRequests, body14003, false
	})
	p := testPoolWith(
		&auth.Auth{UID: "a", AccessToken: "at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "b", AccessToken: "at-b", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{Pool: p, Upstream: up, SoftCooldown: 600 * time.Second})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	// 上游原文透传优先（body 非空）：仅校验轮转被终止 + 两号均未被逐号软冷却。
	for _, uid := range []string{"a", "b"} {
		st, _ := p.Status(uid)
		if st.Cooling {
			t.Errorf("账号 %s 不应被逐号软冷却（共享维度误记应被撤销）: %+v", uid, st)
		}
	}
}

// TestChat14003SharedDimensionWording 上游 body 为空时（无原文可透传），末端错误文案
// 标识共享维度出口限流。
func TestChat14003SharedDimensionWording(t *testing.T) {
	oldBackoff := rotateBackoffBase
	rotateBackoffBase = 0
	defer func() { rotateBackoffBase = oldBackoff }()

	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		return http.StatusTooManyRequests, "", false
	})
	p := testPoolWith(
		&auth.Auth{UID: "a", AccessToken: "at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "b", AccessToken: "at-b", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{Pool: p, Upstream: up, SoftCooldown: 600 * time.Second})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "shared egress") {
		t.Errorf("空 body 末端文案应标识共享维度: %s", rec.Body)
	}
}

// TestChatSingleAccount14003StillCools 单账号池：只一个号命中 14003，不构成共享维度
// 证据，按原语义进入账号级软冷却（回归既有行为，不被新逻辑误伤）。
func TestChatSingleAccount14003StillCools(t *testing.T) {
	oldBackoff := rotateBackoffBase
	rotateBackoffBase = 0
	defer func() { rotateBackoffBase = oldBackoff }()

	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		return http.StatusTooManyRequests, body14003, false
	})
	p := testPoolWith(&auth.Auth{UID: "solo", AccessToken: "at-solo", ExpiresAt: 9999999999})
	h := NewHandler(Config{Pool: p, Upstream: up, SoftCooldown: 600 * time.Second})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))

	st, _ := p.Status("solo")
	if !st.Cooling || st.CoolKind != "soft_rate" {
		t.Fatalf("单号命中应保持账号级软冷却: %+v", st)
	}
}

// TestRateIPGateDisable 关闭开关后 note 恒 false、active 恒 false——轮转永不因
// 共享维度终止（429 回落到账号级软冷却 + 继续换号）。
func TestRateIPGateDisable(t *testing.T) {
	g := rateIPGate{disabled: true}
	for _, uid := range []string{"u1", "u2", "u3"} {
		if g.note(uid, nil) {
			t.Fatalf("disabled 时 %s 命中不应激活", uid)
		}
	}
	if g.active() {
		t.Fatal("disabled 时 active() 应为 false")
	}
}

// TestRateIPGateCustomThreshold 阈值可调：阈值 3 时两个号命中不激活，第三个才激活。
func TestRateIPGateCustomThreshold(t *testing.T) {
	g := rateIPGate{threshold: 3}
	if g.note("u1", nil) {
		t.Fatal("第 1 个号不应激活")
	}
	if g.note("u2", nil) {
		t.Fatal("阈值 3 时第 2 个号不应激活")
	}
	if !g.note("u3", nil) {
		t.Fatal("第 3 个号应激活")
	}
}

// TestRateIPGateCustomWindow 窗口可调：自定义短窗到期后自然解除。
func TestRateIPGateCustomWindow(t *testing.T) {
	g := rateIPGate{window: 20 * time.Millisecond}
	g.note("u1", nil)
	g.note("u2", nil)
	if !g.active() {
		t.Fatal("两个号命中应激活")
	}
	time.Sleep(30 * time.Millisecond)
	if g.active() {
		t.Fatal("自定义窗口到期后应解除")
	}
}

// TestChat429ContinuesRotatingWhenDisabled 关闭共享维度门后，两个号都 429 时
// 轮转应把 MaxRotate 次机会用完（而非首次判定就 break），最终仍回 429。
func TestChat429ContinuesRotatingWhenDisabled(t *testing.T) {
	oldBackoff := rotateBackoffBase
	rotateBackoffBase = 0
	defer func() { rotateBackoffBase = oldBackoff }()

	var hits int
	up := newFakeUpstream(t, func(authz string) (int, string, bool) {
		hits++
		return http.StatusTooManyRequests, body14003, false
	})
	p := testPoolWith(
		&auth.Auth{UID: "a", AccessToken: "at-a", ExpiresAt: 9999999999},
		&auth.Auth{UID: "b", AccessToken: "at-b", ExpiresAt: 9999999999},
	)
	h := NewHandler(Config{
		Pool: p, Upstream: up, SoftCooldown: 600 * time.Second,
		RateIPDisable: true, MaxRotate: 3,
	})

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"glm-5.2","messages":[]}`)))

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	if hits < 2 {
		t.Fatalf("关闭共享维度后应继续换号，实际上游只被调用 %d 次", hits)
	}
}
