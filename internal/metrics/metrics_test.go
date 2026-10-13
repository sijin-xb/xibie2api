package metrics

import (
	"strings"
	"testing"
	"time"
)

func render(t *testing.T, s Snapshot) string {
	t.Helper()
	var b strings.Builder
	if err := Write(&b, s); err != nil {
		t.Fatalf("Write: %v", err)
	}
	return b.String()
}

// 断言某个指标族存在且带指定样本行。
func hasLine(t *testing.T, out, want string) {
	t.Helper()
	if !strings.Contains(out, want) {
		t.Errorf("缺少样本行 %q\n---\n%s", want, out)
	}
}

func TestWriteCoreSeries(t *testing.T) {
	out := render(t, Snapshot{
		Version:           "1.13.0",
		Uptime:            90 * time.Second,
		AccountsTotal:     3,
		AccountsHealthy:   2,
		AccountsCooling:   1,
		AccountsDisabled:  0,
		AccountsInFlight:  1,
		RealmServable:     map[string]bool{"cn": true, "global": false},
		RequestsInFlight:  5,
		RequestsSucceeded: 12345678,
		RequestsFailed:    3,
		RequestAvgSeconds: 1.5,
		TokensPrompt:      1000,
		TokensCompletion:  250,
		TokensTotal:       1250,
		Credits:           7.25,
		CacheHitTokens:    800,
		CacheMissTokens:   200,
		Accounts: []AccountMetric{
			{UID: "u-b", Credits: 42, Healthy: true, Realm: "cn"},
			{UID: "u-a", Credits: 0, Healthy: false, Realm: "global"},
		},
	})

	hasLine(t, out, `workbuddy_build_info{version="1.13.0"} 1`)
	hasLine(t, out, "workbuddy_uptime_seconds 90")
	hasLine(t, out, `workbuddy_accounts{state="total"} 3`)
	hasLine(t, out, `workbuddy_accounts{state="healthy"} 2`)
	hasLine(t, out, `workbuddy_accounts{state="cooling"} 1`)
	hasLine(t, out, `workbuddy_accounts{state="in_flight_full"} 1`)
	hasLine(t, out, `workbuddy_realm_servable{realm="cn"} 1`)
	hasLine(t, out, `workbuddy_realm_servable{realm="global"} 0`)
	hasLine(t, out, `workbuddy_account_credits{uid="u-a",realm="global"} 0`)
	hasLine(t, out, `workbuddy_account_credits{uid="u-b",realm="cn"} 42`)
	hasLine(t, out, `workbuddy_account_healthy{uid="u-a"} 0`)
	hasLine(t, out, `workbuddy_account_healthy{uid="u-b"} 1`)
	hasLine(t, out, "workbuddy_requests_in_flight 5")
	hasLine(t, out, `workbuddy_requests_total{outcome="succeeded"} 12345678`)
	hasLine(t, out, `workbuddy_requests_total{outcome="failed"} 3`)
	hasLine(t, out, "workbuddy_request_duration_seconds_avg 1.5")
	hasLine(t, out, `workbuddy_usage_tokens_total{kind="prompt"} 1000`)
	hasLine(t, out, `workbuddy_usage_tokens_total{kind="completion"} 250`)
	hasLine(t, out, `workbuddy_usage_tokens_total{kind="total"} 1250`)
	hasLine(t, out, "workbuddy_usage_credits_total 7.25")
	hasLine(t, out, `workbuddy_cache_tokens_total{result="hit"} 800`)
	hasLine(t, out, `workbuddy_cache_tokens_total{result="miss"} 200`)

	// 计数器必须用整数表示，不能打成 1.2345678e+07。
	if strings.Contains(out, "e+07") {
		t.Errorf("大整数被写成科学计数法:\n%s", out)
	}
	// 每族样本都带 HELP/TYPE 行（解析器接受重复，这里锁住格式完整性）。
	if !strings.Contains(out, "# TYPE workbuddy_build_info gauge") {
		t.Error("缺 TYPE 行")
	}
	if !strings.Contains(out, "# TYPE workbuddy_requests_total counter") {
		t.Error("缺 counter TYPE 行")
	}
}

// TestAccountSeriesSortedByUID 锁定输出顺序稳定：同一进程两次抓取应逐字节可比，
// 否则 diff 工具与测试都会抖。
func TestAccountSeriesSortedByUID(t *testing.T) {
	s := Snapshot{Accounts: []AccountMetric{
		{UID: "zz"}, {UID: "aa"}, {UID: "mm"},
	}}
	out := render(t, s)
	iAA := strings.Index(out, `uid="aa"`)
	iMM := strings.Index(out, `uid="mm"`)
	iZZ := strings.Index(out, `uid="zz"`)
	if !(iAA < iMM && iMM < iZZ) {
		t.Fatalf("账号序列未按 uid 升序: aa=%d mm=%d zz=%d", iAA, iMM, iZZ)
	}
}

// TestLabelEscaping 锁定标签值转义：uid 若含引号/反斜杠/换行，未转义会破坏文本
// 格式（甚至注入额外标签）。
func TestLabelEscaping(t *testing.T) {
	out := render(t, Snapshot{Accounts: []AccountMetric{
		{UID: "a\"b\\c\nd", Credits: 1},
	}})
	hasLine(t, out, `workbuddy_account_credits{uid="a\"b\\c\nd",realm=""} 1`)
}

// TestZeroSnapshot 空快照必须可渲染（不能 panic，也不能缺族），否则首次抓取
// （池子还没加载完）会让 Prometheus 报错。
func TestZeroSnapshot(t *testing.T) {
	out := render(t, Snapshot{})
	hasLine(t, out, "# TYPE workbuddy_accounts gauge") // 零值也要出族，否则首次抓取会报错
	if !strings.HasSuffix(out, "\n") {
		t.Error("输出应以换行结尾")
	}
}

func TestDescribe(t *testing.T) {
	s := Snapshot{AccountsTotal: 3, AccountsHealthy: 2, RequestsSucceeded: 10, RequestsFailed: 1}
	if got := s.Describe(); !strings.Contains(got, "3 accounts") || !strings.Contains(got, "2 healthy") {
		t.Fatalf("Describe()=%q", got)
	}
}

// TestHelpTypeOncePerFamily 是回归闸门：规范要求同一指标名只允许一条 HELP 行，
// 严格解析器（expfmt TextParser）遇到重复 HELP 会直接报错、整次抓取失败。
// 早期实现按样本各带一条 HELP/TYPE，正是被这条测试挡下的问题。
func TestHelpTypeOncePerFamily(t *testing.T) {
	out := render(t, Snapshot{
		AccountsTotal: 1, AccountsHealthy: 1,
		RealmServable: map[string]bool{"cn": true, "global": true},
		Accounts: []AccountMetric{
			{UID: "u1", Credits: 1, Healthy: true},
			{UID: "u2", Credits: 2, Healthy: true},
		},
	})
	helps := map[string]int{}
	types := map[string]int{}
	for _, line := range strings.Split(out, "\n") {
		if name, ok := strings.CutPrefix(line, "# HELP "); ok {
			helps[strings.Fields(name)[0]]++
		}
		if name, ok := strings.CutPrefix(line, "# TYPE "); ok {
			types[strings.Fields(name)[0]]++
		}
	}
	if len(helps) == 0 {
		t.Fatal("没有解析到任何 HELP 行")
	}
	for name, n := range helps {
		if n != 1 {
			t.Errorf("指标 %s 有 %d 条 HELP 行（规范要求恰好 1 条）", name, n)
		}
	}
	for name, n := range types {
		if n != 1 {
			t.Errorf("指标 %s 有 %d 条 TYPE 行（规范要求恰好 1 条）", name, n)
		}
	}
	// 同族多标签值（多个账号）必须共用一条 HELP/TYPE。
	if helps["workbuddy_account_credits"] != 1 || helps["workbuddy_account_healthy"] != 1 {
		t.Errorf("逐账号族应各只有一条 HELP: %v", helps)
	}
}

// TestToolChainObservabilitySeries 锁定工具调用链路的观测族：配对管线四向 +
// 原生标记还原三态必须渲染，且同族共用恰好一条 HELP/TYPE（严格解析器要求）。
func TestToolChainObservabilitySeries(t *testing.T) {
	out := render(t, Snapshot{
		PairingMergedAdjacent: 3,
		PairingRepacked:       2,
		PairingDroppedCalls:   1,
		PairingDroppedResults: 1,
		NativeRepairSeen:      10,
		NativeRepairConverted: 8,
		NativeRepairRejected:  2,
	})
	hasLine(t, out, `workbuddy_tool_pairing_events_total{direction="merged_adjacent"} 3`)
	hasLine(t, out, `workbuddy_tool_pairing_events_total{direction="repacked"} 2`)
	hasLine(t, out, `workbuddy_tool_pairing_events_total{direction="dropped_tool_call"} 1`)
	hasLine(t, out, `workbuddy_tool_pairing_events_total{direction="dropped_tool_result"} 1`)
	hasLine(t, out, `workbuddy_native_tool_repair_total{result="seen"} 10`)
	hasLine(t, out, `workbuddy_native_tool_repair_total{result="converted"} 8`)
	hasLine(t, out, `workbuddy_native_tool_repair_total{result="rejected"} 2`)

	helps := map[string]int{}
	for _, line := range strings.Split(out, "\n") {
		if name, ok := strings.CutPrefix(line, "# HELP "); ok {
			helps[strings.Fields(name)[0]]++
		}
	}
	if helps["workbuddy_tool_pairing_events_total"] != 1 || helps["workbuddy_native_tool_repair_total"] != 1 {
		t.Errorf("工具链路观测族应各只有一条 HELP: %v", helps)
	}
}
