package keyusage

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/usage"
)

func okDelta(pt, ct int64) usage.Delta {
	return usage.Delta{
		PromptTokens:     pt,
		HasPromptTokens:  true,
		CompletionTokens: ct,
		HasCompletion:    true,
		TotalTokens:      pt + ct,
		HasTotal:         true,
		Credit:           1.5,
		HasCredit:        true,
	}
}

func TestAddAccumulates(t *testing.T) {
	l := New("")
	now := time.Now()
	l.Add(now, "fp-a", "A", okDelta(10, 5), true)
	l.Add(now, "fp-a", "A", okDelta(1, 1), false) // 失败也计入请求数
	l.Add(now, "fp-b", "B", okDelta(100, 50), true)

	snap := l.Snapshot(0, nil)
	if len(snap.Rows) != 2 {
		t.Fatalf("应有两行，got %+v", snap.Rows)
	}
	// 按总量降序：B(150) 在前。
	if snap.Rows[0].Fingerprint != "fp-b" || snap.Rows[1].Fingerprint != "fp-a" {
		t.Fatalf("排序错误: %+v", snap.Rows)
	}
	a := snap.Rows[1]
	if a.Requests != 2 || a.Errors != 1 {
		t.Fatalf("请求数/失败数错误: %+v", a)
	}
	if a.PromptTokens != 11 || a.CompletionTokens != 6 || a.TotalTokens != 17 {
		t.Fatalf("token 累计错误: %+v", a)
	}
	if a.Credits != 3.0 || a.CreditSamples != 2 {
		t.Fatalf("积分累计错误: %+v", a)
	}
	if a.ActiveDays != 1 {
		t.Fatalf("活跃天应为 1，got %d", a.ActiveDays)
	}
	if a.LastUsed == "" {
		t.Fatal("最近使用应非空")
	}
}

func TestAddIgnoresEmptyFingerprint(t *testing.T) {
	l := New("")
	l.Add(time.Now(), "", "", okDelta(1, 1), true)
	if snap := l.Snapshot(0, nil); len(snap.Rows) != 0 {
		t.Fatalf("指纹为空不应产生台账行，got %+v", snap.Rows)
	}
}

func TestSnapshotWindow(t *testing.T) {
	l := New("")
	now := time.Now()
	l.Add(now, "fp-new", "新", okDelta(10, 10), true)
	l.Add(now.AddDate(0, 0, -10), "fp-old", "旧", okDelta(1000, 1000), true)
	l.Add(now.AddDate(0, 0, -3), "fp-mid", "中", okDelta(100, 100), true)

	// 今天：只有 fp-new。
	snap := l.Snapshot(1, nil)
	if len(snap.Rows) != 1 || snap.Rows[0].Fingerprint != "fp-new" {
		t.Fatalf("days=1 应只含今天，got %+v", snap.Rows)
	}
	// 近 7 天：新 + 中（3 天前），不含 10 天前。
	snap = l.Snapshot(7, nil)
	if len(snap.Rows) != 2 {
		t.Fatalf("days=7 应含两行，got %+v", snap.Rows)
	}
	for _, r := range snap.Rows {
		if r.Fingerprint == "fp-old" {
			t.Fatal("days=7 不应含 10 天前的数据")
		}
	}
	// 全部历史：三行。
	if snap = l.Snapshot(0, nil); len(snap.Rows) != 3 {
		t.Fatalf("days=0 应含全部三行，got %+v", snap.Rows)
	}
	if snap.Since == "" {
		t.Fatal("全部历史应给出数据起点")
	}
}

func TestSnapshotIncludesConfiguredButUnused(t *testing.T) {
	l := New("")
	l.Add(time.Now(), "fp-used", "用过", okDelta(10, 10), true)

	snap := l.Snapshot(0, map[string]string{"fp-used": "用过", "fp-idle": "配了没用"})
	if len(snap.Rows) != 2 {
		t.Fatalf("配了但没用过的钥匙也应占一行，got %+v", snap.Rows)
	}
	var found bool
	for _, r := range snap.Rows {
		if r.Fingerprint == "fp-idle" {
			found = true
			if r.Requests != 0 || r.Name != "配了没用" {
				t.Fatalf("未使用行应为零值 + 备注名，got %+v", r)
			}
		}
	}
	if !found {
		t.Fatal("未使用的配置项缺失")
	}
}

func TestPersistenceRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "key-usage.json")
	l := New(path)
	l.Add(time.Now(), "fp-a", "A", okDelta(7, 3), true)
	l.Save()

	l2 := New(path)
	snap := l2.Snapshot(0, nil)
	if len(snap.Rows) != 1 || snap.Rows[0].Fingerprint != "fp-a" || snap.Rows[0].TotalTokens != 10 {
		t.Fatalf("重启后台账未恢复: %+v", snap.Rows)
	}
	if snap.FileBytes == 0 {
		t.Fatal("应报告文件大小")
	}
}

func TestNilLedgerSafe(t *testing.T) {
	var l *Ledger
	l.Add(time.Now(), "fp", "n", okDelta(1, 1), true) // 不应 panic
	if snap := l.Snapshot(0, nil); len(snap.Rows) != 0 {
		t.Fatalf("nil 台账应返回空快照，got %+v", snap.Rows)
	}
}
