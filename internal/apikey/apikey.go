// Package apikey 多 API Key 的配置模型与匹配表。
//
// 背景：早期只有顶层单个 api_key，一把钥匙通行全部模型、两个域。多把钥匙的诉求是
// 把「谁能调什么」收窄到钥匙粒度：手机端只给便宜模型、Codex 专用钥匙只绑 global 域、
// 自动化脚本的钥匙不给贵模型——越权请求必须在网关本地就被挡掉，不送上游、不消耗额度。
//
// 本包只做两件事：
//   - 把配置里的 api_keys 编译成一张匹配表（含校验，fail fast）；
//   - 对入站请求做常量时间的密钥匹配，返回身份（备注名 / 指纹 / 模型白名单 / 域绑定）。
//
// 不做的事：不读请求体、不决定 HTTP 状态码、不写响应。模型白名单的判定入口是
// Identity.Allows，由 server 层在解析出模型名之后调用——那里才知道请求到底要哪个模型。
//
// 兼容纪律：**未配置 api_keys 时行为与引入前逐字节一致**（回落顶层单 api_key 语义），
// 老配置零变化。这是本包所有分支的首要约束。
package apikey

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// bearerPrefix 认证方案前缀（与 internal/httpauth 同口径，大小写敏感）。
const bearerPrefix = "Bearer "

// realmCN / realmGlobal 合法域取值；空串表示"跟随请求"。
const (
	RealmCN     = "cn"
	RealmGlobal = "global"
)

// Entry 配置里的一条 API Key（config.json 的 api_keys 元素）。
//
// JSON 字段名与面板/文档一致；Enabled 用指针语义表达"缺省 true"会让调用方
// 处处判空，故这里用普通 bool 并在 Compile 前由 Default 统一补默认值（见 Normalize）。
type Entry struct {
	Key     string   `json:"key"`               // 密钥原文（必填；启用中的条目为空即启动报错）
	Name    string   `json:"name,omitempty"`    // 备注名，用于日志/面板/报错文案
	Models  []string `json:"models,omitempty"`  // 允许的模型，支持 * / ? 通配；空 = 不限制
	Realm   string   `json:"realm,omitempty"`   // 出口绑定 cn / global；空 = 跟随请求
	Enabled *bool    `json:"enabled,omitempty"` // 缺省 true；false 时该条目不参与匹配
}

// IsEnabled 缺省 true（未写 enabled 视为启用）。
func (e Entry) IsEnabled() bool { return e.Enabled == nil || *e.Enabled }

// Identity 一次成功鉴权得到的调用方身份。零值 = "未启用鉴权"，模型/域均不限制。
type Identity struct {
	Name        string // 备注名（未配置或走顶层单 key 时为空）
	Fingerprint string // 密钥指纹（sha256 前 8 字节 hex）；顶层单 key 也为空
	Realm       string // 出口绑定（空 = 跟随请求）
	models      []string
	configured  bool // 是否来自某条显式配置（顶层单 key / 无鉴权均为 false）
}

// Configured 报告本次身份是否来自一条显式配置的 API Key。
func (id Identity) Configured() bool { return id.configured }

// compiled 一条已校验、已预算摘要的配置。
type compiled struct {
	entry  Entry
	digest []byte
	models []string // 规范化后的模型模式（小写；含 ':' 的按带前缀名匹配）
}

// Table 编译后的匹配表。并发只读（热更新靠整体替换，见 livecfg）。
type Table struct {
	entries  []compiled
	legacy   []byte // 顶层 api_key 的摘要；nil = 未配置
	required bool   // 是否需要鉴权（配了任意一把钥匙）
}

// Fingerprint 返回密钥指纹：sha256 前 8 字节的 hex。
//
// 台账与面板只用指纹，**不落密钥原文**——台账文件可能被备份/同步/贴进 issue，
// 存原文等于把密钥扩散到它不该去的地方。
func Fingerprint(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:8])
}

func digest(s string) []byte {
	sum := sha256.Sum256([]byte(s))
	return sum[:]
}

// NewTable 校验并编译匹配表。legacyKey 是顶层单 api_key（可为空）。
//
// 校验 fail fast：密钥为空（但条目启用）、密钥重复、realm 非法、密钥含空白或 '*'
// 都在这里报错。理由与别名表一致——"配了但永远匹配不上"的哑配置比启动失败更难查。
func NewTable(entries []Entry, legacyKey string) (*Table, error) {
	t := &Table{}
	if legacyKey != "" {
		t.legacy = digest(legacyKey)
		t.required = true
	}
	seen := map[string]bool{}
	for i, e := range entries {
		// 禁用条目只校验 realm 与 models 形态：它不参与匹配，因此允许把 key 清空
		// （"先禁用、回头再换密钥"是正常操作），但仍然拦住 realm 拼错这类哑配置。
		if !e.IsEnabled() {
			if err := validateRealmAndModels(e); err != nil {
				return nil, fmt.Errorf("api_keys[%d]: %w", i, err)
			}
			continue
		}
		if err := validateKey(e.Key); err != nil {
			return nil, fmt.Errorf("api_keys[%d]: %w", i, err)
		}
		if err := validateRealmAndModels(e); err != nil {
			return nil, fmt.Errorf("api_keys[%d]: %w", i, err)
		}
		if seen[e.Key] {
			return nil, fmt.Errorf("api_keys[%d]: duplicate key %s", i, Fingerprint(e.Key))
		}
		seen[e.Key] = true
		c := compiled{entry: e, digest: digest(e.Key)}
		for _, m := range e.Models {
			c.models = append(c.models, strings.ToLower(strings.TrimSpace(m)))
		}
		t.entries = append(t.entries, c)
		t.required = true
	}
	return t, nil
}

func validateKey(key string) error {
	if strings.TrimSpace(key) == "" {
		return errors.New("key is empty")
	}
	if strings.ContainsAny(key, " \t\r\n") {
		return errors.New("key must not contain whitespace")
	}
	if strings.Contains(key, "*") {
		return errors.New("key must not contain '*'")
	}
	return nil
}

func validateRealmAndModels(e Entry) error {
	switch e.Realm {
	case "", RealmCN, RealmGlobal:
	default:
		return fmt.Errorf("realm must be %q, %q or empty, got %q", RealmCN, RealmGlobal, e.Realm)
	}
	for _, m := range e.Models {
		if strings.TrimSpace(m) == "" {
			return errors.New("models must not contain empty patterns")
		}
	}
	return nil
}

// Required 报告是否启用了鉴权（配了任意一把钥匙或顶层 api_key）。
func (t *Table) Required() bool { return t != nil && t.required }

// Entries 返回全部**启用中**的配置条目（面板展示"配了但没用过的钥匙"用）。
func (t *Table) Entries() []Entry {
	if t == nil {
		return nil
	}
	out := make([]Entry, 0, len(t.entries))
	for _, c := range t.entries {
		out = append(out, c.entry)
	}
	return out
}

// LookupFingerprint 按指纹反查配置条目（台账行与配置对照用）。
func (t *Table) LookupFingerprint(fp string) (Entry, bool) {
	if t == nil {
		return Entry{}, false
	}
	for _, c := range t.entries {
		if Fingerprint(c.entry.Key) == fp {
			return c.entry, true
		}
	}
	return Entry{}, false
}

// Authenticate 校验入站请求的 Bearer 密钥并返回身份。**这是 /v1/* 的严格口径**。
//
// 语义：
//   - 未配置任何密钥（required=false）→ 恒放行，返回零值身份（不鉴权模式）；
//   - 配了 api_keys → **只有 api_keys 里的密钥可用**，顶层 api_key 不再放行 /v1/*。
//     理由：本功能的目的就是把"谁能调什么"收窄到钥匙粒度，若顶层万能钥匙仍可通行，
//     收窄就是装饰性的。顶层 api_key 退居面板/管理接口的后备密钥（面板鉴权不走本表）。
//   - 只配了顶层 api_key（无 api_keys）→ 回落既有单密钥语义，行为与引入前逐字节一致。
//
// 常量时间纪律：**不提前返回**。所有条目的摘要都比较一遍，命中索引按位或累积，
// 最后一次性选出结果。提前 return 会让"第一把就命中"与"最后一把才命中"的耗时不同，
// 从而泄露条目在表中的位置。
func (t *Table) Authenticate(r *http.Request) (Identity, bool) {
	if !t.Required() {
		return Identity{}, true
	}
	tok := ""
	if authz := r.Header.Get("Authorization"); strings.HasPrefix(authz, bearerPrefix) {
		tok = authz[len(bearerPrefix):]
	} else if k := r.Header.Get("x-api-key"); k != "" {
		// Anthropic 客户端（Claude Code / ZCode 的 anthropic provider）用 x-api-key。
		tok = k
	}
	d := digest(tok)

	// 无 api_keys：既有单密钥路径（含"缺头也走一次摘要比较以保持耗时形状"的原语义）。
	if len(t.entries) == 0 {
		if subtle.ConstantTimeCompare(d, t.legacy) == 1 {
			return Identity{}, true
		}
		return Identity{}, false
	}

	// 命中索引用"常量时间选择"累积：eq 是 0/1，hit = eq*i + (1-eq)*hit 无分支，
	// 因此耗时与命中位置无关（写成 if eq == 1 { hit = i } 会留下一个只在命中那次
	// 才走的分支，侧信道可据此定位有效密钥在表中的序号）。
	hit := 0
	matched := 0
	for i := range t.entries {
		eq := subtle.ConstantTimeCompare(d, t.entries[i].digest)
		matched |= eq
		hit = eq*i + (1-eq)*hit
	}
	if matched != 1 {
		return Identity{}, false
	}
	c := t.entries[hit]
	return Identity{
		Name:        c.entry.Name,
		Fingerprint: Fingerprint(c.entry.Key),
		Realm:       c.entry.Realm,
		models:      c.models,
		configured:  true,
	}, true
}

// Allows 判定本身份是否可以调用给定模型。
//
// 两个入参是同一模型名的两种形态：effectiveModel 是别名替换后、realm 前缀仍在的
// 客户端可见名；bareModel 是剥掉前缀后的上游裸名。匹配规则按模式是否带 ':' 分流——
// 带 ':' 的模式（如 "global:glm-*"）比带前缀名，不带 ':' 的模式（如 "glm-*"）比裸名。
// 这样两种写法都自然可用，且不会出现"写了 glm-* 却因为客户端带了 global: 前缀而失配"。
//
// 白名单为空 = 不限制（未配置 api_keys、走顶层单 key、或该条目没写 models）。
func (id Identity) Allows(effectiveModel, bareModel string) bool {
	if len(id.models) == 0 {
		return true
	}
	eff := strings.ToLower(effectiveModel)
	bare := strings.ToLower(bareModel)
	for _, pat := range id.models {
		target := bare
		if strings.Contains(pat, ":") {
			target = eff
		}
		if globMatch(pat, target) {
			return true
		}
	}
	return false
}

// RealmOverride 返回本身份绑定的域；空串 = 跟随请求（不覆盖）。
func (id Identity) RealmOverride() string { return id.Realm }

// globMatch 匹配只含 '*'（任意长度，含空）与 '?'（恰好一个字符）的模式。
// 模式与目标都应为小写（调用方已规范化）。迭代实现，回溯点只保留最近一个 '*'，
// 因此最坏 O(n*m)、常规接近线性——白名单模式都是短串，无需更复杂的算法。
func globMatch(pat, s string) bool {
	var (
		p, i        int
		star        = -1
		starResumed int
	)
	for i < len(s) {
		switch {
		case p < len(pat) && (pat[p] == '?' || pat[p] == s[i]):
			p++
			i++
		case p < len(pat) && pat[p] == '*':
			star = p
			starResumed = i
			p++
		case star >= 0:
			// 回溯：让上一个 '*' 多吃一个字符再试。
			p = star + 1
			starResumed++
			i = starResumed
		default:
			return false
		}
	}
	for p < len(pat) && pat[p] == '*' {
		p++
	}
	return p == len(pat)
}
