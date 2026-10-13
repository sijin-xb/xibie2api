package main

import "testing"

// TestAPIKeysConfig 覆盖 api_keys 的配置解析：缺省零影响、正常条目、以及各类
// 非法配置 fail fast。与 model_aliases 的测试同构（同一套 ParseConfig 校验路径）。
func TestAPIKeysConfig(t *testing.T) {
	// 缺省（键缺席）：只有顶层 api_key，鉴权语义与引入前一致。
	c, err := ParseConfig([]byte(`{"api_key":"legacy"}`))
	if err != nil {
		t.Fatalf("parse default: %v", err)
	}
	if c.KeyTable == nil {
		t.Fatal("KeyTable 应恒非 nil")
	}
	if len(c.APIKeys) != 0 {
		t.Fatalf("缺省 api_keys 应为空，got %+v", c.APIKeys)
	}
	if n := len(c.KeyTable.Entries()); n != 0 {
		t.Fatalf("缺省不应有启用条目，got %d", n)
	}

	// 显式空数组：同样不启用多 Key。
	c, err = ParseConfig([]byte(`{"api_key":"legacy","api_keys":[]}`))
	if err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if n := len(c.KeyTable.Entries()); n != 0 {
		t.Fatalf("空 api_keys 不应有启用条目，got %d", n)
	}

	// 正常条目：字段完整落库（含 enabled 缺省 true）。
	c, err = ParseConfig([]byte(`{"api_keys":[
		{"key":"sk-a","name":"Codex","models":["gpt-*"],"realm":"global"},
		{"key":"sk-b","name":"手机端"},
		{"key":"sk-off","name":"停用","enabled":false}
	]}`))
	if err != nil {
		t.Fatalf("parse valid: %v", err)
	}
	entries := c.KeyTable.Entries()
	if len(entries) != 2 {
		t.Fatalf("应有 2 个启用条目（禁用条目不计），got %d: %+v", len(entries), entries)
	}
	if entries[0].Key != "sk-a" || entries[0].Realm != "global" || entries[0].Name != "Codex" {
		t.Fatalf("条目 0 解析错误: %+v", entries[0])
	}
	if !entries[1].IsEnabled() {
		t.Fatal("enabled 缺省应为 true")
	}
	if c.APIKeys[2].IsEnabled() {
		t.Fatal("显式 enabled=false 应被保留")
	}

	// 非法配置：一律 fail fast（不静默忽略）。
	bad := []struct {
		name string
		body string
	}{
		{"启用条目密钥为空", `{"api_keys":[{"key":""}]}`},
		{"密钥重复", `{"api_keys":[{"key":"sk-x"},{"key":"sk-x"}]}`},
		{"密钥含通配符", `{"api_keys":[{"key":"sk-*"}]}`},
		{"realm 非法", `{"api_keys":[{"key":"sk-x","realm":"us"}]}`},
		{"models 含空模式", `{"api_keys":[{"key":"sk-x","models":["glm-*"," "]}]}`},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseConfig([]byte(tc.body)); err == nil {
				t.Fatal("非法配置应报错")
			}
		})
	}
}
