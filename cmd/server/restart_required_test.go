package main

import (
	"fmt"
	"testing"
)

// TestRestartRequiredFieldsIncludePrompt 锁定：prompt.mode / prompt.file 属于装配期
// 静态捕获（main.go 的 NewHandler 取 cfg.Prompt.Mode / cfg.PromptText），不在
// livecfg 热快照里，必须出现在 restartRequiredFields 清单中——否则面板改完提示词
// 显示「已保存」却不提示重启，用户以为没生效。该行为是此前修复加入的，此处补回归
// 测试防止回退（此前无任何 restartRequiredFields 的测试）。
func TestRestartRequiredFieldsIncludePrompt(t *testing.T) {
	got := restartRequiredFields(Default())
	want := map[string]bool{
		"prompt.mode":              false,
		"prompt.file":              false,
		"upstream.timeout_seconds": false, // 既有恒列项，作为清单非空/结构健全的对照
	}
	for _, f := range got {
		if _, ok := want[f]; ok {
			want[f] = true
		}
	}
	for f, seen := range want {
		if !seen {
			t.Errorf("restartRequiredFields 缺少 %q；got=%v", f, got)
		}
	}
}

// TestPromptModeCanonicalized 锁定 prompt.mode 的归一化写回：normalizePrompt 把
// c.Prompt.Mode 改写为规范小写值（passthrough/custom/append），大小写与首尾空白均
// 归一。这条写回是承重的：main.go 直接把 cfg.Prompt.Mode 传给 handler，而 handler
// 用精确字符串比较分派（handler.go 的 PromptMode == "custom"/"append"/"passthrough"）。
// 若归一化不回写，用户写 "Custom" 会让 normalize 通过、提示词文件也加载，但 handler
// 不匹配任何分支 → 提示词静默不生效（正是「配置看似成功却未生效」的隐患）。
func TestPromptModeCanonicalized(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"  Custom  ", "custom"},
		{"APPEND", "append"},
		{"Passthrough", "passthrough"},
		{"", "passthrough"}, // 缺省 passthrough（对齐 Default() 与 README）
	}
	for _, tc := range cases {
		raw := []byte(fmt.Sprintf(`{"prompt":{"mode":%q}}`, tc.in))
		c, err := ParseConfig(raw)
		if err != nil {
			t.Fatalf("mode %q: ParseConfig err: %v", tc.in, err)
		}
		if c.Prompt.Mode != tc.want {
			t.Errorf("mode %q 归一化后为 %q，want %q", tc.in, c.Prompt.Mode, tc.want)
		}
		// custom/append 必须加载非空 PromptText（内置默认），否则 handler 的空串守卫
		// 会让改写静默失效；passthrough 不加载。
		if tc.want == "passthrough" {
			if c.PromptText != "" {
				t.Errorf("mode %q: passthrough 不应加载 PromptText，got len=%d", tc.in, len(c.PromptText))
			}
		} else if c.PromptText == "" {
			t.Errorf("mode %q: custom/append 应加载非空 PromptText", tc.in)
		}
	}
}
