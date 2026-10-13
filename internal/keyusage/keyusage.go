// Package keyusage 逐 API Key 的用量台账。
//
// 为什么不复用 internal/usage：那个包的桶维度是 (时间片, realm, uid, model, rate)，
// 再加一个 key 维度会让桶数乘上钥匙把数——而它的桶是"账号数 × 模型数 × 天数"量级，
// 乘法放大会把 maxBuckets 顶穿。逐 Key 台账的诉求小得多：只回答"哪把钥匙用了多少"，
// 不需要按模型/账号下钻，因此按 (指纹, 本地日) 分桶，桶数 = 钥匙数 × 天数，天然很小。
//
// 落盘 data/key-usage.json，原子替换 + 30s 防抖，重启不丢。
// **只存指纹（sha256 前 8 字节）与备注名，永不存密钥原文**。
package keyusage

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/usage"
)

// flushInterval 防抖落盘间隔（与 usage 包同值，保持运维直觉一致）。
const flushInterval = 30 * time.Second

// dayLayout 分片键的日期格式（本地时区，与用户直觉一致）。
const dayLayout = "2006-01-02"

// fileVersion 是 key-usage.json 的当前格式版本。
const fileVersion = 1

// bucket 一把钥匙在某一天（本地日）的累计量。
type bucket struct {
	FP   string  `json:"k"`           // 密钥指纹
	Name string  `json:"n,omitempty"` // 备注名（配置改名后以最近一次为准）
	Day  string  `json:"d"`           // 本地日 YYYY-MM-DD
	Req  int64   `json:"q"`           // 请求数（含失败）
	Err  int64   `json:"e"`           // 失败数
	PT   int64   `json:"p"`           // prompt tokens
	CT   int64   `json:"c"`           // completion tokens
	TT   int64   `json:"t"`           // total tokens
	CR   float64 `json:"r,omitempty"` // 积分累计（仅明确存在的观测）
	CRN  int64   `json:"rn,omitempty"`
	Last int64   `json:"l"` // 最近使用（unix 秒）
}

type file struct {
	Version int      `json:"version"`
	Saved   string   `json:"saved"`
	Buckets []bucket `json:"buckets"`
}

// Ledger 并发安全的逐 Key 台账。
type Ledger struct {
	mu      sync.Mutex
	path    string
	buckets map[string]*bucket // key: fp|day
	dirty   bool

	stopOnce sync.Once
	stop     chan struct{}
	done     chan struct{}
}

// New 创建台账。path 为空时禁用落盘（纯内存，测试用）。
func New(path string) *Ledger {
	l := &Ledger{
		path:    path,
		buckets: make(map[string]*bucket),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}
	if path != "" {
		if err := l.load(); err != nil {
			log.Printf("[keyusage] 读取 %s 失败（从零开始）: %v", path, err)
		}
	}
	return l
}

// Start 启动后台防抖落盘。Stop 前一直运行。
func (l *Ledger) Start() {
	go func() {
		defer close(l.done)
		t := time.NewTicker(flushInterval)
		defer t.Stop()
		for {
			select {
			case <-l.stop:
				l.flush(true)
				return
			case <-t.C:
				l.flush(false)
			}
		}
	}()
}

// Stop 停止后台循环并做最后一次落盘。
func (l *Ledger) Stop() {
	l.stopOnce.Do(func() { close(l.stop) })
	<-l.done
}

// Add 记录一次请求尝试。fp 为空时直接返回（未启用多 Key，不产生台账）。
//
// ok=false 表示该次尝试失败；失败也计入请求数，理由与 usage 包相同——重试放大
// 只有把失败计进来才看得见。
func (l *Ledger) Add(now time.Time, fp, name string, d usage.Delta, ok bool) {
	if l == nil || fp == "" {
		return
	}
	day := now.Format(dayLayout)
	key := fp + "|" + day

	l.mu.Lock()
	defer l.mu.Unlock()

	b := l.buckets[key]
	if b == nil {
		b = &bucket{FP: fp, Name: name, Day: day}
		l.buckets[key] = b
	}
	if name != "" {
		b.Name = name
	}
	b.Req++
	if !ok {
		b.Err++
	}
	if d.HasPromptTokens {
		b.PT += d.PromptTokens
	}
	if d.HasCompletion {
		b.CT += d.CompletionTokens
	}
	if d.HasTotal {
		b.TT += d.TotalTokens
	} else if d.HasPromptTokens || d.HasCompletion {
		b.TT += d.PromptTokens + d.CompletionTokens
	}
	if d.HasCredit {
		b.CR += d.Credit
		b.CRN++
	}
	if ts := now.Unix(); ts > b.Last {
		b.Last = ts
	}
	l.dirty = true
}

// ---------------------------------------------------------------- 持久化 ----

func (l *Ledger) load() error {
	raw, err := os.ReadFile(l.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	var f file
	if err := json.Unmarshal(raw, &f); err != nil {
		return err
	}
	for i := range f.Buckets {
		b := f.Buckets[i]
		l.buckets[b.FP+"|"+b.Day] = &b
	}
	log.Printf("[keyusage] 已恢复 %d 个逐 Key 台账桶（%s）", len(l.buckets), l.path)
	return nil
}

func (l *Ledger) flush(force bool) {
	if l == nil || l.path == "" {
		return
	}
	l.mu.Lock()
	if !l.dirty && !force {
		l.mu.Unlock()
		return
	}
	snap := file{Version: fileVersion, Saved: time.Now().Format(time.RFC3339), Buckets: make([]bucket, 0, len(l.buckets))}
	for _, b := range l.buckets {
		snap.Buckets = append(snap.Buckets, *b)
	}
	l.dirty = false
	l.mu.Unlock()

	raw, err := json.Marshal(snap)
	if err != nil {
		log.Printf("[keyusage] 序列化失败: %v", err)
		return
	}
	if err := os.MkdirAll(filepath.Dir(l.path), 0o755); err != nil {
		log.Printf("[keyusage] 建目录失败: %v", err)
		return
	}
	tmp := l.path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		log.Printf("[keyusage] 写临时文件失败: %v", err)
		return
	}
	if err := os.Rename(tmp, l.path); err != nil {
		log.Printf("[keyusage] 原子替换失败: %v", err)
	}
}

// Save 立即落盘（面板刷新或关闭前调用）。
func (l *Ledger) Save() { l.flush(true) }

// ---------------------------------------------------------------- 聚合 ----

// Row 一把钥匙在所选窗口内的用量汇总。
type Row struct {
	Fingerprint      string  `json:"fingerprint"`
	Name             string  `json:"name,omitempty"`
	Requests         int64   `json:"requests"`
	Errors           int64   `json:"errors"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	TotalTokens      int64   `json:"total_tokens"`
	Credits          float64 `json:"credits"`
	CreditSamples    int64   `json:"credit_samples"`
	ActiveDays       int     `json:"active_days"`
	LastUsed         string  `json:"last_used,omitempty"` // RFC3339；从未使用为空
}

// Snapshot 一次拉取的台账视图。
type Snapshot struct {
	Days      int    `json:"days"` // 生效窗口天数；0 = 全部历史
	Rows      []Row  `json:"rows"`
	Buckets   int    `json:"buckets"`
	FileBytes int64  `json:"file_bytes"`
	Since     string `json:"since,omitempty"`
	Generated string `json:"generated"`
}

// Snapshot 聚合窗口内的台账。days<=0 表示全部历史；否则窗口 = 含今天在内的最近
// days 个本地日（days=1 即"今天"）。窗口按桶的日标签判定，与 usage 包的日桶口径一致。
//
// nicks 提供"配置里配了但从未使用"的钥匙也要占一行所需的展示名；传 nil 时只出
// 有数据的行（纯内存测试场景）。函数只读台账，不校验指纹是否仍在配置中——那是
// 面板层与配置对照的职责。
func (l *Ledger) Snapshot(days int, configured map[string]string) Snapshot {
	snap := Snapshot{Days: days, Generated: time.Now().Format(time.RFC3339)}
	if l == nil {
		return snap
	}
	var cutoff string
	if days > 0 {
		cutoff = time.Now().AddDate(0, 0, -(days - 1)).Format(dayLayout)
	}

	l.mu.Lock()
	bs := make([]bucket, 0, len(l.buckets))
	for _, b := range l.buckets {
		bs = append(bs, *b)
	}
	l.mu.Unlock()

	agg := map[string]*Row{}
	activeDays := map[string]map[string]bool{}
	since := ""
	for i := range bs {
		b := &bs[i]
		if since == "" || b.Day < since {
			since = b.Day
		}
		if cutoff != "" && b.Day < cutoff {
			continue
		}
		snap.Buckets++
		r := agg[b.FP]
		if r == nil {
			r = &Row{Fingerprint: b.FP, Name: b.Name}
			agg[b.FP] = r
			activeDays[b.FP] = map[string]bool{}
		}
		if b.Name != "" {
			r.Name = b.Name
		}
		r.Requests += b.Req
		r.Errors += b.Err
		r.PromptTokens += b.PT
		r.CompletionTokens += b.CT
		r.TotalTokens += b.TT
		r.Credits += b.CR
		r.CreditSamples += b.CRN
		activeDays[b.FP][b.Day] = true
		if b.Last > 0 {
			ts := time.Unix(b.Last, 0).Format(time.RFC3339)
			if r.LastUsed == "" || ts > r.LastUsed {
				r.LastUsed = ts
			}
		}
	}

	// 配置里配了但窗口内没用过的钥匙也占一行（用 0 值行），否则"某把钥匙是不是
	// 没人用"这个问题在面板上无法回答。
	for fp, name := range configured {
		if _, ok := agg[fp]; !ok {
			agg[fp] = &Row{Fingerprint: fp, Name: name}
		}
	}

	snap.Rows = make([]Row, 0, len(agg))
	for _, r := range agg {
		r.ActiveDays = len(activeDays[r.Fingerprint])
		snap.Rows = append(snap.Rows, *r)
	}
	// 按总量降序；同量按指纹升序，保证输出稳定（前端 diff 不抖）。
	sort.Slice(snap.Rows, func(i, j int) bool {
		if snap.Rows[i].TotalTokens != snap.Rows[j].TotalTokens {
			return snap.Rows[i].TotalTokens > snap.Rows[j].TotalTokens
		}
		if snap.Rows[i].Requests != snap.Rows[j].Requests {
			return snap.Rows[i].Requests > snap.Rows[j].Requests
		}
		return snap.Rows[i].Fingerprint < snap.Rows[j].Fingerprint
	})
	snap.Since = since
	if l.path != "" {
		if fi, err := os.Stat(l.path); err == nil {
			snap.FileBytes = fi.Size()
		}
	}
	return snap
}
