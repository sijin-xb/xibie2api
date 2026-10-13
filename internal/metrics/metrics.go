// Package metrics 把网关指标渲染成 Prometheus 文本格式（exposition format 0.0.4）。
//
// 为什么手写而不引 prometheus/client_golang：本仓库的依赖纪律是「能不加就不加」
// （前端 go:embed 单文件、零外部构建依赖；go.mod 里只有 redis 与 x/sys 两个间接
// 依赖）。指标只有十几个 gauge/counter，文本格式真正需要遵守的规范面（名字字符集、
// 标签值转义、HELP/TYPE 行、每族一条）小到可以手工正确实现并用测试锁住，不值得为此
// 引入一整棵依赖树。
//
// 本包是**纯渲染层**：不读 pool/reqlog/usage，只吃调用方填好的 Snapshot。这样
// 「哪些账号算 healthy、哪些请求算失败」的口径留在 server 层，与 /status、面板
// 完全同源，不会出现「面板和 /metrics 数字对不上」这类最难查的问题。
package metrics

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// AccountMetric 单个账号的观测值。标签只用 uid——昵称可能含引号/换行/非 ASCII，
// 放进标签值虽合法但会让仪表盘查询难写，且改名会造成时间序列 churn。
type AccountMetric struct {
	UID     string
	Credits int64
	Healthy bool // 未禁用、未冷却（与 /status 的 healthy 同口径）
	Realm   string
}

// Snapshot 一次抓取的全部指标值。零值合法（渲染出全 0，不 panic）。
type Snapshot struct {
	Version string
	Uptime  time.Duration

	AccountsTotal    int
	AccountsHealthy  int
	AccountsCooling  int
	AccountsDisabled int
	AccountsInFlight int // 因在途上限而暂时不可选号的账号数
	RealmServable    map[string]bool
	Accounts         []AccountMetric

	RequestsInFlight  int64
	RequestsSucceeded int64
	RequestsFailed    int64
	RequestAvgSeconds float64

	TokensPrompt     int64
	TokensCompletion int64
	TokensTotal      int64
	Credits          float64
	CacheHitTokens   int64
	CacheMissTokens  int64

	// 工具调用链路观测（加法式，仅观测，不影响行为）。
	// 配对管线：客户端坏历史的自愈动作计数（见 upstream/tool_pairing.go）。
	PairingMergedAdjacent int64
	PairingRepacked       int64
	PairingDroppedCalls   int64
	PairingDroppedResults int64
	// 原生工具调用标记还原：Seen=识别到的标记块数，Converted=还原出的调用数，
	// Rejected=识别到但未还原的块数（见 upstream/dsml.go）。
	NativeRepairSeen      int64
	NativeRepairConverted int64
	NativeRepairRejected  int64
}

// Write 把快照写成 Prometheus 文本格式。任何错误都来自底层 Writer。
// Write 把快照写成 Prometheus 文本格式。任何错误都来自底层 Writer。
//
// 输出按**指标族**组织：每个名字只出一条 HELP/TYPE，随后是它的全部样本（见 emitter）。
func Write(w io.Writer, s Snapshot) error {
	e := newEmitter()

	e.gauge("workbuddy_build_info", "Build information; value is always 1.",
		[]label{{"version", s.Version}}, 1)
	e.gauge("workbuddy_uptime_seconds", "Seconds since the gateway started.",
		nil, s.Uptime.Seconds())

	e.gauge("workbuddy_accounts", "Account pool size by state.",
		[]label{{"state", "total"}}, float64(s.AccountsTotal))
	e.gauge("workbuddy_accounts", "Account pool size by state.",
		[]label{{"state", "healthy"}}, float64(s.AccountsHealthy))
	e.gauge("workbuddy_accounts", "Account pool size by state.",
		[]label{{"state", "cooling"}}, float64(s.AccountsCooling))
	e.gauge("workbuddy_accounts", "Account pool size by state.",
		[]label{{"state", "disabled"}}, float64(s.AccountsDisabled))
	e.gauge("workbuddy_accounts", "Account pool size by state.",
		[]label{{"state", "in_flight_full"}}, float64(s.AccountsInFlight))

	// realm 可服务性：0/1，与 /healthz 的 realm_servable 同口径（双域部署时
	// 单独告警某一域不可用）。固定顺序输出，保证同一进程两次抓取逐字节可比。
	for _, realm := range []string{"cn", "global"} {
		v := 0.0
		if s.RealmServable[realm] {
			v = 1
		}
		e.gauge("workbuddy_realm_servable", "1 when the realm can serve requests now.",
			[]label{{"realm", realm}}, v)
	}

	// 逐账号指标：按 uid 排序输出（稳定，便于 diff 与测试）。
	accts := append([]AccountMetric(nil), s.Accounts...)
	sort.Slice(accts, func(i, j int) bool { return accts[i].UID < accts[j].UID })
	for _, a := range accts {
		e.gauge("workbuddy_account_credits", "Remaining credits of one account.",
			[]label{{"uid", a.UID}, {"realm", a.Realm}}, float64(a.Credits))
	}
	for _, a := range accts {
		v := 0.0
		if a.Healthy {
			v = 1
		}
		e.gauge("workbuddy_account_healthy", "1 when the account can serve requests now.",
			[]label{{"uid", a.UID}}, v)
	}

	e.gauge("workbuddy_requests_in_flight", "Requests currently being served.",
		nil, float64(s.RequestsInFlight))
	e.counter("workbuddy_requests_total", "Completed requests by outcome.",
		[]label{{"outcome", "succeeded"}}, float64(s.RequestsSucceeded))
	e.counter("workbuddy_requests_total", "Completed requests by outcome.",
		[]label{{"outcome", "failed"}}, float64(s.RequestsFailed))
	e.gauge("workbuddy_request_duration_seconds_avg",
		"Average end-to-end request duration in seconds.", nil, s.RequestAvgSeconds)

	e.counter("workbuddy_usage_tokens_total", "Tokens observed from upstream usage by kind.",
		[]label{{"kind", "prompt"}}, float64(s.TokensPrompt))
	e.counter("workbuddy_usage_tokens_total", "Tokens observed from upstream usage by kind.",
		[]label{{"kind", "completion"}}, float64(s.TokensCompletion))
	e.counter("workbuddy_usage_tokens_total", "Tokens observed from upstream usage by kind.",
		[]label{{"kind", "total"}}, float64(s.TokensTotal))
	e.counter("workbuddy_usage_credits_total", "Credits deducted, as reported by upstream usage.",
		nil, s.Credits)
	e.counter("workbuddy_cache_tokens_total", "Prefix-cache tokens by result.",
		[]label{{"result", "hit"}}, float64(s.CacheHitTokens))
	e.counter("workbuddy_cache_tokens_total", "Prefix-cache tokens by result.",
		[]label{{"result", "miss"}}, float64(s.CacheMissTokens))

	// 工具调用配对管线事件（客户端坏历史自愈）：direction 固定 4 值，基数可控。
	// 无 model 标签——模型维度基数不可控（上游模型名任意），细节留在 WARN 日志里。
	e.counter("workbuddy_tool_pairing_events_total",
		"Tool-call pairing pipeline events by direction (client history repair).",
		[]label{{"direction", "merged_adjacent"}}, float64(s.PairingMergedAdjacent))
	e.counter("workbuddy_tool_pairing_events_total",
		"Tool-call pairing pipeline events by direction (client history repair).",
		[]label{{"direction", "repacked"}}, float64(s.PairingRepacked))
	e.counter("workbuddy_tool_pairing_events_total",
		"Tool-call pairing pipeline events by direction (client history repair).",
		[]label{{"direction", "dropped_tool_call"}}, float64(s.PairingDroppedCalls))
	e.counter("workbuddy_tool_pairing_events_total",
		"Tool-call pairing pipeline events by direction (client history repair).",
		[]label{{"direction", "dropped_tool_result"}}, float64(s.PairingDroppedResults))

	// 原生工具调用标记还原（DSML 文本 → 结构化 tool_calls）：result 固定 3 值。
	// rejected 是「识别到标记却没还原」的残留风险面，单独看这条。
	e.counter("workbuddy_native_tool_repair_total",
		"Native tool-call markup repair outcomes (DSML text -> structured tool_calls).",
		[]label{{"result", "seen"}}, float64(s.NativeRepairSeen))
	e.counter("workbuddy_native_tool_repair_total",
		"Native tool-call markup repair outcomes (DSML text -> structured tool_calls).",
		[]label{{"result", "converted"}}, float64(s.NativeRepairConverted))
	e.counter("workbuddy_native_tool_repair_total",
		"Native tool-call markup repair outcomes (DSML text -> structured tool_calls).",
		[]label{{"result", "rejected"}}, float64(s.NativeRepairRejected))

	_, err := io.WriteString(w, e.render())
	return err
}

type label struct{ name, value string }

type sample struct {
	labels []label
	value  float64
}

// family 一个指标族的全部样本。同名样本必须聚在同一族里，才能只出一条 HELP/TYPE。
type family struct {
	name, help, typ string
	samples         []sample
}

// emitter 按首次出现顺序收集指标族，最后一次性渲染。
//
// 为什么必须成族输出：规范要求「同一指标名只允许一条 HELP 行」，严格解析器
// （expfmt 的 TextParser）遇到重复 HELP 会直接报错，整次抓取失败。早期实现按
// 样本各带一条 HELP/TYPE，用真实抓取才暴露出来。
type emitter struct {
	order []string
	fams  map[string]*family
}

func newEmitter() *emitter { return &emitter{fams: map[string]*family{}} }

func (e *emitter) gauge(name, help string, labels []label, v float64) {
	e.add(name, help, "gauge", labels, v)
}

func (e *emitter) counter(name, help string, labels []label, v float64) {
	e.add(name, help, "counter", labels, v)
}

func (e *emitter) add(name, help, typ string, labels []label, v float64) {
	f := e.fams[name]
	if f == nil {
		f = &family{name: name, help: help, typ: typ}
		e.fams[name] = f
		e.order = append(e.order, name)
	}
	f.samples = append(f.samples, sample{labels: labels, value: v})
}

func (e *emitter) render() string {
	var b strings.Builder
	for _, name := range e.order {
		f := e.fams[name]
		b.WriteString("# HELP ")
		b.WriteString(f.name)
		b.WriteByte(' ')
		b.WriteString(escapeHelp(f.help))
		b.WriteByte('\n')
		b.WriteString("# TYPE ")
		b.WriteString(f.name)
		b.WriteByte(' ')
		b.WriteString(f.typ)
		b.WriteByte('\n')
		for _, s := range f.samples {
			b.WriteString(f.name)
			if len(s.labels) > 0 {
				b.WriteByte('{')
				for i, l := range s.labels {
					if i > 0 {
						b.WriteByte(',')
					}
					b.WriteString(l.name)
					b.WriteString(`="`)
					b.WriteString(escapeLabel(l.value))
					b.WriteByte('"')
				}
				b.WriteByte('}')
			}
			b.WriteByte(' ')
			b.WriteString(formatFloat(s.value))
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// formatFloat 优先用整数表示：计数器都是整数量级（token/请求数），用 'g' 会把
// 12345678 打成 "1.2345678e+07"，虽合法但难读。非整数（积分、平均耗时）回退 'g'
// 的最短往返表示。
func formatFloat(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

// escapeLabel 转义标签值：反斜杠、双引号、换行（规范要求的三项）。
func escapeLabel(s string) string {
	if !strings.ContainsAny(s, "\\\"\n") {
		return s
	}
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return r.Replace(s)
}

// escapeHelp 转义 HELP 文本：反斜杠与换行（双引号在 HELP 里无需转义）。
func escapeHelp(s string) string {
	if !strings.ContainsAny(s, "\\\n") {
		return s
	}
	return strings.NewReplacer(`\`, `\\`, "\n", `\n`).Replace(s)
}

// ContentType 是 Prometheus 文本格式的 Content-Type（含版本协商参数）。
const ContentType = "text/plain; version=0.0.4; charset=utf-8"

// Describe 返回一行人类可读摘要（启动日志用）。
func (s Snapshot) Describe() string {
	return fmt.Sprintf("%d accounts (%d healthy), %d requests served, %d failed",
		s.AccountsTotal, s.AccountsHealthy, s.RequestsSucceeded, s.RequestsFailed)
}
