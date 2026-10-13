package pool

import (
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// TestUndoSoftRateCooldownRemovesMisapplied 正常路径：撤销一次账号级软冷却，
// 清 until/coolKind/reason 并回退 softStreak。
func TestUndoSoftRateCooldownRemovesMisapplied(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.CooldownSoftRate("u1", time.Hour, time.Time{}, "429 rate limit")
	st, _ := p.Status("u1")
	if !st.Cooling || st.SoftStreak != 1 {
		t.Fatalf("前置：应进入软冷却: %+v", st)
	}
	if !p.UndoSoftRateCooldown("u1") {
		t.Fatal("应撤销成功")
	}
	st, _ = p.Status("u1")
	if st.Cooling || st.Reason != "" {
		t.Errorf("撤销后不应冷却: %+v", st)
	}
	if st.SoftStreak != 0 {
		t.Errorf("softStreak 应回退到 0: %d", st.SoftStreak)
	}
}

// TestUndoSoftRateCooldownGuards 守卫：非软冷却/已过期/禁用/存在真实模型级冷却时
// 均不撤销。
func TestUndoSoftRateCooldownGuards(t *testing.T) {
	// 硬冷却（CoolHard）不动。
	p := New("")
	p.Add(&auth.Auth{UID: "hard"})
	p.Cooldown("hard", CoolHard, time.Hour, "余额不足")
	if p.UndoSoftRateCooldown("hard") {
		t.Error("硬冷却不应被撤销")
	}

	// 禁用号不动。
	p.Add(&auth.Auth{UID: "dis"})
	p.CooldownSoftRate("dis", time.Hour, time.Time{}, "429 rate limit")
	p.Disable("dis", "test")
	if p.UndoSoftRateCooldown("dis") {
		t.Error("禁用号不应被撤销")
	}

	// 存在真实模型级冷却（6004）时，账号级软冷却不属共享维度误记，保留。
	// 走公开 API 构造不出并存态：CooldownSoftRate 会清路由模型冷却，故直接注入。
	p.Add(&auth.Auth{UID: "mod"})
	p.CooldownSoftRate("mod", time.Hour, time.Time{}, "429 rate limit")
	p.mu.Lock()
	p.byUID["mod"].modelCooldowns = map[string]modelCooldown{
		"glm": {Until: time.Now().Add(time.Hour), Reason: "6004 model rate limit"},
	}
	p.mu.Unlock()
	if p.UndoSoftRateCooldown("mod") {
		t.Error("存在真实模型级冷却时不应撤销")
	}

	// 不存在的 uid。
	if p.UndoSoftRateCooldown("nope") {
		t.Error("不存在 uid 不应撤销")
	}
}
