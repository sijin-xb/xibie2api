package server

import "testing"

func TestNewAliasTableEmptyIsNil(t *testing.T) {
	tbl, err := NewAliasTable(nil)
	if err != nil {
		t.Fatalf("nil entries: err=%v", err)
	}
	if tbl != nil {
		t.Fatalf("nil entries should yield nil table, got %+v", tbl)
	}
	// nil 表必须恒等（未配置别名 = 行为不变）。
	if got := tbl.Apply("anything"); got != "anything" {
		t.Fatalf("nil table Apply=%q want identity", got)
	}
	if tbl.Len() != 0 {
		t.Fatalf("nil table Len=%d want 0", tbl.Len())
	}
}

func TestAliasTableApply(t *testing.T) {
	tbl, err := NewAliasTable([]ModelAlias{
		{From: "claude-sonnet-4-5", To: "cn:glm-5.3"},
		{From: "gpt-5", To: "global:deepseek-v4.1-flash"},
		{From: "my-bare", To: "glm-5.3"},
	})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if tbl.Len() != 3 {
		t.Fatalf("Len=%d want 3", tbl.Len())
	}
	cases := []struct{ in, want string }{
		{"claude-sonnet-4-5", "cn:glm-5.3"},        // 命中（带 realm 目标）
		{"gpt-5", "global:deepseek-v4.1-flash"},    // 命中（global 目标）
		{"my-bare", "glm-5.3"},                     // 命中（裸名目标）
		{"not-aliased", "not-aliased"},             // 未命中 → 原样
		{"cn:glm-5.3", "cn:glm-5.3"},               // realm 前缀名不走别名（键不含 ':'）
		{"CLAUDE-SONNET-4-5", "CLAUDE-SONNET-4-5"}, // 大小写敏感：不命中
	}
	for _, c := range cases {
		if got := tbl.Apply(c.in); got != c.want {
			t.Errorf("Apply(%q)=%q want %q", c.in, got, c.want)
		}
	}
	// 别名目标继续经过既有 realm 解析（不绕过解析管线）。
	realm, bare := resolveModel(tbl.Apply("claude-sonnet-4-5"))
	if realm != "cn" || bare != "glm-5.3" {
		t.Fatalf("resolveModel(alias target)=%q,%q want cn,glm-5.3", realm, bare)
	}
}

func TestNewAliasTableInvalid(t *testing.T) {
	cases := []struct {
		name    string
		entries []ModelAlias
	}{
		{"empty from", []ModelAlias{{From: "", To: "glm-5.3"}}},
		{"empty to", []ModelAlias{{From: "a", To: ""}}},
		{"from with colon", []ModelAlias{{From: "cn:glm", To: "glm-5.3"}}},
		{"from with space", []ModelAlias{{From: "a b", To: "glm-5.3"}}},
		{"from with wildcard", []ModelAlias{{From: "a*", To: "glm-5.3"}}},
		{"to with wildcard", []ModelAlias{{From: "a", To: "glm-*"}}},
		{"duplicate from", []ModelAlias{{From: "a", To: "x"}, {From: "a", To: "y"}}},
		{"to resolves empty", []ModelAlias{{From: "a", To: "cn:"}}},
		{"chained alias", []ModelAlias{{From: "a", To: "b"}, {From: "b", To: "c"}}},
	}
	for _, c := range cases {
		tbl, err := NewAliasTable(c.entries)
		if err == nil {
			t.Errorf("%s: want error, got table=%+v", c.name, tbl)
		}
	}
}
