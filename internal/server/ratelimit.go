// ratelimit.go 上游 429「共享维度」限流的 fail-fast 状态机。
//
// 背景（实测 2026-10-13）：global 域 429 body =
// {"code":14003,"msg":"too many requests","retry_policy":{"retry":true}}，
// 无「将在 … 重置」文案、无 Retry-After 头。实测该 429 打在**当前被选中的账号**
// 头上，但该账号前 60s 内往往只发了 0–1 次请求（网关总吞吐仅 ~15 req/min）——
// 限流维度不是账号，而是网关共享的出口 IP / 设备指纹。既有 applyErrorPolicy 把
// 每次此类 429 记为账号级软冷却（soft_rate 600s 起），于是全池账号被逐个罚 10 分钟，
// 表现为「莫名全池限流冷却」。
//
// 判定（对称 wafip.go）：rateIPWindow 内 ≥ rateIPThreshold 个**不同** UID 命中此类
// 429 → 判定共享维度限流，激活至 now+window；激活期内轮转遇此类 429 直接终止
// （不再把请求放大 MaxRotate 倍打同一出口），并跳过账号级软冷却（不误罚号）。
// 只有「无权威恢复时刻的账号级 429」参与判定：带 6004 模型级、带 Retry-After 头、
// 带「将在 … 重置」墙钟的限流各有权威恢复时刻，不属共享维度（见 isSharedDimensionRateLimit）。
//
// 归属层：放 server（Handler 局部）而非 pool——与 wafip.go 同因：IP 级状态唯一
// 消费者是 runTurn 轮转循环（是否继续轮转），pool 是账号级记账层，跨 UID 语义
// 不属于任何账号。进程内状态、重启清零（窗口 60s，重建成本极低）。
package server

import (
	"log"
	"sync"
	"time"
)

// rateIPWindow 共享维度判定滑动窗 + 激活时长（对称 wafIPWindow）。var 仅供测试
// 注入短窗（生产恒 60s）。
var rateIPWindow = 60 * time.Second

// rateIPThreshold 判定阈值：窗内不同 UID 数达到该值激活。取 2——「多号」的最小
// 定义：单号反复命中永不触发（账号级偶发归软冷却管），两个不同号在窗内接连命中
// （同一出口 IP/设备指纹）已是共享维度证据。
const rateIPThreshold = 2

// rateIPGate 共享维度限流状态机（Handler 内嵌，零值可用）。
type rateIPGate struct {
	mu    sync.Mutex
	hits  map[string]time.Time // uid → 最近一次命中时刻（判定窗内，惰性剪枝）
	until time.Time            // 激活截止；零值 = 未激活
	// disabled 关闭该门：note 恒返回 false，轮转永不因共享维度 fail-fast 终止。
	// 单号池下该门本就永不激活（需 ≥2 个不同 UID），关闭只在多号池有意义。
	disabled bool
	// threshold 激活阈值，0 = 用 rateIPThreshold 常量。
	threshold int
	// window 判定窗 + 激活时长，0 = 用 rateIPWindow 变量。
	window time.Duration
}

// note 记一次某账号的共享维度 429，返回记账后是否处于激活态（调用方据此 fail-fast
// 终止轮转，并跳过账号级软冷却）。revoke 用于激活时**回溯撤销**判定窗内各账号已被
// 误记的账号级软冷却（首个命中号在激活前已按软冷却记账 600s，若不撤销仍会被罚满）；
// 传 nil 表示不撤销（测试/无需撤销场景）。
//   - 已激活（now < until）：不续期、不记账（窗口期内不重置——保守自然解除）→ true；
//   - 未激活：记 hits[uid]=now（同号重复命中覆盖不累计，判定口径是「不同号数」），
//     剪掉窗外的过期命中；不同 UID 数达 rateIPThreshold → 激活到 now+rateIPWindow
//     （打一条 WARN 供观测），对窗内各 uid 调 revoke 撤销误记冷却，清空判定窗
//     （解除后需全新命中重新判定，不叠旧账）。
func (g *rateIPGate) note(uid string, revoke func(uid string) bool) bool {
	if g.disabled {
		return false
	}
	win := g.window
	if win <= 0 {
		win = rateIPWindow
	}
	threshold := g.threshold
	if threshold <= 0 {
		threshold = rateIPThreshold
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	if now.Before(g.until) {
		return true // 激活期内新命中：不续期（自然解除语义）
	}
	if g.hits == nil {
		g.hits = map[string]time.Time{}
	}
	g.hits[uid] = now
	for u, t := range g.hits {
		if now.Sub(t) > win {
			delete(g.hits, u)
		}
	}
	if len(g.hits) >= threshold {
		g.until = now.Add(win)
		revoked := 0
		if revoke != nil {
			for u := range g.hits {
				if revoke(u) {
					revoked++
				}
			}
		}
		log.Printf("WARN: [server] upstream 429 shared-dimension throttle: %d accounts hit within %s, rotate fail-fast until %s (revoked %d misapplied soft cooldown(s))",
			len(g.hits), win, g.until.Format(time.RFC3339), revoked)
		g.hits = map[string]time.Time{}
		return true
	}
	return false
}

// active 报告共享维度限流是否激活（末端错误文案区分用）。
func (g *rateIPGate) active() bool {
	if g.disabled {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return time.Now().Before(g.until)
}
