// Package alert 账号池可用性阈值告警：健康账号数跌破阈值时向 webhook 推一条
// 通知，恢复到阈值以上时再推一条恢复通知。
//
// 为什么需要它：网关最典型的故障形态是「账号全被冷却/禁用，但进程还活着」——
// /healthz 会返回 503、/v1/* 返回 503 no_healthy_account，可如果没有外部告警，
// 用户只有在客户端报错时才发现。这个包把「池子里还有几个能用的号」变成一个可
// 被外部系统订阅的信号。
//
// 签名与重放：请求体是 JSON，用共享密钥做 HMAC-SHA256，放进 `X-Signature:
// sha256=<hex>`；同时带 `X-Timestamp`（unix 秒），并与请求体里的 `timestamp`
// 字段同值——接收方应校验两者一致且时间新鲜，防止旧报文被重放。接收方算法：
//
//	hex(hmac_sha256(secret, raw_body)) == header 去掉 "sha256=" 前缀的值
//
// 本包的 Signature 就是给接收方对照用的同一实现。
//
// 状态机只有两个状态（firing / 正常），并且**只在成功送达后**才切换状态与冷却：
// 推送失败保持原状态，下一次检查立即重试——否则一次网络抖动会让告警被"吞掉"，
// 而告警被吞掉正是告警系统最不能接受的失败方式。
package alert

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// defaultTick Run 的基础节拍。实际是否评估由配置里的 interval 门控（可热改），
// 因此节拍取一个比常见 interval 更细的固定值，热改 interval 无需重启。
const defaultTick = 15 * time.Second

// postTimeout 单次 webhook 推送的超时。
const postTimeout = 10 * time.Second

// Config 告警配置（由调用方从 livecfg 快照提供，因此天然热生效）。
type Config struct {
	Enabled    bool          // 总开关；false 时状态机复位（重新启用会重新告警一次）
	WebhookURL string        // 目标地址（http/https）；为空视为未配置
	Secret     string        // HMAC 共享密钥；为空则不签名（仅建议内网使用）
	MinHealthy int           // 健康账号数下限；healthy < MinHealthy 即告警
	Cooldown   time.Duration // 持续故障期间的重复提醒间隔
	Interval   time.Duration // 两次评估之间的最小间隔
}

// Event 推送给 webhook 的请求体。
type Event struct {
	Event     string `json:"event"` // "alert"（故障）| "recovery"（恢复）
	Service   string `json:"service"`
	Version   string `json:"version,omitempty"`
	Healthy   int    `json:"healthy"`
	Total     int    `json:"total"`
	Threshold int    `json:"threshold"`
	Timestamp int64  `json:"timestamp"` // unix 秒，与 X-Timestamp 头同值
	At        string `json:"at"`        // RFC3339（本地时区），便于人读
}

// Monitor 告警状态机。
//
// cfg/count 都是闭包：cfg 每次评估时重新读取（热生效），count 直接问池子要
// 当前口径的数字，避免这里再维护一份会漂移的副本。
type Monitor struct {
	service string
	version string
	cfg     func() Config
	count   func() (total, healthy int)

	// 以下三个字段在测试里被替换（同包测试可直接赋值），生产走默认实现。
	now  func() time.Time
	post func(ctx context.Context, url, secret string, body []byte, ts int64) error
	tick time.Duration

	mu        sync.Mutex
	firing    bool      // 当前是否处于"已告警未恢复"状态
	lastSent  time.Time // 最近一次**成功送达**的时间（冷却基准）
	lastCheck time.Time // 最近一次评估时间（interval 门控基准）
}

// New 构建监控器。service/version 用于请求体与日志，便于多实例区分。
func New(service, version string, cfg func() Config, count func() (total, healthy int)) *Monitor {
	return &Monitor{
		service: service,
		version: version,
		cfg:     cfg,
		count:   count,
		now:     time.Now,
		post:    defaultPost,
		tick:    defaultTick,
	}
}

// Run 按固定节拍评估，直到 ctx 结束。启动时立即评估一次（进程刚起来就发现池子
// 是空的，不该等到第一个节拍）。
func (m *Monitor) Run(ctx context.Context) {
	t := time.NewTicker(m.tick)
	defer t.Stop()
	m.Check(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			m.Check(ctx)
		}
	}
}

// Check 执行一次评估。可由测试直接调用（不必等待节拍）。
func (m *Monitor) Check(ctx context.Context) {
	cfg := m.cfg()

	m.mu.Lock()
	defer m.mu.Unlock()

	// 关闭或未配置：复位状态机。这样"关掉再打开"会重新告警一次，而不是沿用
	// 关之前的 firing 状态导致恢复通知永不发出。
	if !cfg.Enabled || strings.TrimSpace(cfg.WebhookURL) == "" {
		m.firing = false
		m.lastCheck = time.Time{}
		return
	}
	now := m.now()
	if !m.lastCheck.IsZero() && now.Sub(m.lastCheck) < cfg.Interval {
		return
	}
	m.lastCheck = now

	total, healthy := m.count()
	below := healthy < cfg.MinHealthy

	switch {
	case !below:
		// 正常：若之前在告警，推一条恢复通知。
		if m.firing {
			if err := m.send(ctx, cfg, "recovery", total, healthy, now); err == nil {
				m.firing = false
				m.lastSent = now
			}
		}
	case !m.firing:
		// 首次跌破：立即告警。
		if err := m.send(ctx, cfg, "alert", total, healthy, now); err == nil {
			m.firing = true
			m.lastSent = now
		}
	case now.Sub(m.lastSent) >= cfg.Cooldown:
		// 仍在故障且已过冷却：重复提醒。长时间故障只提醒一次等于没提醒——
		// 值班的人换班后就再也收不到信号了。
		if err := m.send(ctx, cfg, "alert", total, healthy, now); err == nil {
			m.lastSent = now
		}
	}
}

func (m *Monitor) send(ctx context.Context, cfg Config, event string, total, healthy int, now time.Time) error {
	ev := Event{
		Event:     event,
		Service:   m.service,
		Version:   m.version,
		Healthy:   healthy,
		Total:     total,
		Threshold: cfg.MinHealthy,
		Timestamp: now.Unix(),
		At:        now.Format(time.RFC3339),
	}
	body, err := json.Marshal(ev)
	if err != nil {
		log.Printf("[alert] 序列化告警失败: %v", err)
		return err
	}
	if err := m.post(ctx, cfg.WebhookURL, cfg.Secret, body, now.Unix()); err != nil {
		// 不打 URL 之外的内容（secret 与 body 都不进日志）。
		log.Printf("[alert] %s 推送失败: %v（下轮重试）", event, err)
		return err
	}
	log.Printf("[alert] 已推送 %s: healthy=%d/%d 阈值=%d", event, healthy, total, cfg.MinHealthy)
	return nil
}

// Signature 计算请求体的 HMAC-SHA256 签名（十六进制，不带前缀）。
// 导出是为了让接收方能对照同一实现，也便于测试锁住签名口径。
func Signature(secret string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}

// defaultPost 默认推送实现：POST JSON，带 X-Timestamp 与（有密钥时）X-Signature。
func defaultPost(ctx context.Context, url, secret string, body []byte, ts int64) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "workbuddy2api-alert")
	req.Header.Set("X-Timestamp", strconv.FormatInt(ts, 10))
	if secret != "" {
		req.Header.Set("X-Signature", "sha256="+Signature(secret, body))
	}
	client := &http.Client{Timeout: postTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	// 读掉并丢弃响应体（限长），保证连接可复用；内容只用于诊断状态码。
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("webhook 返回 HTTP %d", resp.StatusCode)
	}
	return nil
}
