package main

import "testing"

// TestModelAliasesConfig 覆盖 model_aliases 的配置解析：缺省零影响、正常映射、
// 非法配置 fail fast（映射冲突 / 非法 from / 链式别名）。
func TestModelAliasesConfig(t *testing.T) {
	// 缺省（键缺席）：无别名表，行为不变。
	c, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("parse default: %v", err)
	}
	if c.AliasTable != nil {
		t.Fatalf("absent model_aliases should yield nil table, got %+v", c.AliasTable)
	}

	// 显式空数组：同样无别名。
	c, err = ParseConfig([]byte(`{"model_aliases":[]}`))
	if err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if c.AliasTable != nil {
		t.Fatalf("empty model_aliases should yield nil table, got %+v", c.AliasTable)
	}

	// 正常映射：目标可为带 realm 前缀或裸名。
	c, err = ParseConfig([]byte(`{"model_aliases":[{"from":"claude-sonnet-4-5","to":"cn:glm-5.3"},{"from":"gpt-5","to":"global:deepseek-v4.1-flash"}]}`))
	if err != nil {
		t.Fatalf("parse valid: %v", err)
	}
	if c.AliasTable == nil {
		t.Fatal("valid model_aliases should build a table")
	}
	if got := c.AliasTable.Apply("claude-sonnet-4-5"); got != "cn:glm-5.3" {
		t.Fatalf("Apply(claude)=%q want cn:glm-5.3", got)
	}
	if got := c.AliasTable.Apply("gpt-5"); got != "global:deepseek-v4.1-flash" {
		t.Fatalf("Apply(gpt-5)=%q want global:deepseek-v4.1-flash", got)
	}
	if got := c.AliasTable.Apply("unknown"); got != "unknown" {
		t.Fatalf("Apply(unknown)=%q want identity", got)
	}

	// 非法配置：均应报错（不静默忽略）。
	invalid := []struct {
		name string
		raw  string
	}{
		{"duplicate from", `{"model_aliases":[{"from":"a","to":"x"},{"from":"a","to":"y"}]}`},
		{"from with colon", `{"model_aliases":[{"from":"cn:glm","to":"glm-5.3"}]}`},
		{"empty from", `{"model_aliases":[{"from":"","to":"glm-5.3"}]}`},
		{"empty to", `{"model_aliases":[{"from":"a","to":""}]}`},
		{"chained alias", `{"model_aliases":[{"from":"a","to":"b"},{"from":"b","to":"c"}]}`},
		{"to resolves empty", `{"model_aliases":[{"from":"a","to":"cn:"}]}`},
	}
	for _, c := range invalid {
		if _, err := ParseConfig([]byte(c.raw)); err == nil {
			t.Errorf("%s: want error, got nil", c.name)
		}
	}
}
