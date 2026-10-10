package server

import "testing"

// TestResolveModel 回归既有 realm 前缀解析行为（D6），确保别名改造未改变原语义。
func TestResolveModel(t *testing.T) {
	cases := []struct {
		in        string
		wantRealm string
		wantBare  string
	}{
		{"glm-5.3", "cn", "glm-5.3"},                                    // 裸名 → cn
		{"cn:glm-5.3", "cn", "glm-5.3"},                                 // 显式 cn 前缀
		{"global:deepseek-v4.1-flash", "global", "deepseek-v4.1-flash"}, // global 前缀
		{"Global:x", "cn", "Global:x"},                                  // 前缀大小写敏感：非枚举 → 裸名
		{"a:b:c", "cn", "a:b:c"},                                        // 首段非枚举 → 整串裸名
		{"cn:", "cn", ""},                                               // 前缀后为空
		{"", "cn", ""},                                                  // 空串 → cn + 空
	}
	for _, c := range cases {
		realm, bare := resolveModel(c.in)
		if realm != c.wantRealm || bare != c.wantBare {
			t.Errorf("resolveModel(%q)=%q,%q want %q,%q", c.in, realm, bare, c.wantRealm, c.wantBare)
		}
	}
}
