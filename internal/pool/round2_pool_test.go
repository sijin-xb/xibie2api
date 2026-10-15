package pool

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// round2_pool_test.go 覆盖第二轮审查修复的账号池缺陷。

// #4 SetNickname 与 Status 并发：不得有数据竞争（-race 下必须干净）。
func TestNicknameConcurrentWithStatusNoRace(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 3000; i++ {
			p.SetNickname("u1", fmt.Sprintf("n%d", i))
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 3000; i++ {
			_, _ = p.Status("u1")
		}
	}()
	wg.Wait()
}

// #8 有效硬冷却（余额耗尽）不得被后到的软冷却覆盖。
func TestSoftDoesNotOverrideActiveHardCooldown(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	p.CooldownUntilTomorrow4AM("u1", "402 hard")
	p.CooldownSoftRate("u1", 5*time.Minute, time.Now().Add(5*time.Minute), "429 soft")
	st, _ := p.Status("u1")
	if st.CoolKind != "hard_credit" {
		t.Fatalf("hard cooldown overridden by soft: kind=%q", st.CoolKind)
	}
}

// #9 面板删除+重建条目后，在途计数必须跨重建保留（旧请求确实还在途），
// 否则旧请求的 Release 会扣到新条目上 → 在途少计 → 超发。
func TestStaleReleaseDoesNotOverAdmit(t *testing.T) {
	p := New("")
	p.SetMaxInFlight(1)
	p.Add(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	if !p.Acquire("u1") {
		t.Fatal("first acquire failed")
	}
	// 模拟面板删除 + 重新导入（同 uid 重建条目）
	p.Remove("u1")
	p.Add(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	// 旧请求 A 仍在途：重建后的条目必须仍显示占满，不得再接纳新请求
	if p.Acquire("u1") {
		t.Fatal("rebuilt entry lost in-flight count -> over-admission (limit=1)")
	}
	// 旧请求 A 结束，释放名额
	p.Release("u1")
	if !p.Acquire("u1") {
		t.Fatal("slot should be free after the stale request released it")
	}
}

// #11 Revive 必须清连败降权（degradeUntil）。
func TestReviveClearsDegrade(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	for i := 0; i < 6; i++ {
		p.NoteFailures("u1")
	}
	p.mu.RLock()
	before := p.byUID["u1"].degradeUntil
	p.mu.RUnlock()
	if before.IsZero() {
		t.Skip("degrade not triggered with current threshold")
	}
	p.Revive("u1")
	p.mu.RLock()
	after := p.byUID["u1"].degradeUntil
	p.mu.RUnlock()
	if !after.IsZero() {
		t.Fatalf("Revive left degradeUntil set: %s", after.Format(time.RFC3339))
	}
}

// #12 UndoSoftRateCooldown 不得清掉无关成因（WAF 403）的软冷却。
func TestUndoDoesNotClearUnrelatedCooldown(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	p.CooldownSoftRate("u1", time.Hour, time.Time{}, "waf 403 block")
	if p.UndoSoftRateCooldown("u1") {
		t.Fatal("UndoSoftRateCooldown cleared an unrelated WAF cooldown")
	}
	if st, _ := p.Status("u1"); !st.Cooling {
		t.Fatal("WAF cooldown should survive UndoSoftRateCooldown")
	}
	// 共享维度 429 冷却则可撤销。
	p.CooldownSoftRate("u1", time.Hour, time.Time{}, "429 rate limit")
	if !p.UndoSoftRateCooldown("u1") {
		t.Fatal("shared-dimension 429 cooldown should be revocable")
	}
}

// #14 BlockModelBackoff(11102) 不得覆盖既有真实 6004 条目。
func TestModelBackoffDoesNotOverwrite6004(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	p.CooldownSoftForModel("u1", 5*time.Minute, time.Now().Add(20*time.Minute), "glm-5.2", "6004 model rate limit")
	p.mu.RLock()
	before := p.byUID["u1"].modelCooldowns["glm-5.2"]
	p.mu.RUnlock()
	p.BlockModelBackoff("u1", "glm-5.2", "11102 model unavailable")
	p.mu.RLock()
	after := p.byUID["u1"].modelCooldowns["glm-5.2"]
	p.mu.RUnlock()
	if after.Reason != before.Reason {
		t.Fatalf("6004 entry overwritten by 11102: reason %q -> %q", before.Reason, after.Reason)
	}
	if !after.Until.Equal(before.Until) {
		t.Fatalf("6004 until changed by 11102: %s -> %s", before.Until.Format(time.RFC3339), after.Until.Format(time.RFC3339))
	}
}
