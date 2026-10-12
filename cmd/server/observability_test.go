package main

import (
	"testing"
	"time"
)

// TestMetricsConfig 覆盖 metrics 段：缺省关闭。
func TestMetricsConfig(t *testing.T) {
	c, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("parse default: %v", err)
	}
	if c.Metrics.Enabled {
		t.Fatal("metrics 缺省应关闭")
	}
	c, err = ParseConfig([]byte(`{"metrics":{"enabled":true}}`))
	if err != nil {
		t.Fatalf("parse enabled: %v", err)
	}
	if !c.Metrics.Enabled {
		t.Fatal("metrics.enabled=true 应被解析")
	}
}

// TestAlertingConfig 覆盖 alerting 段：缺省值、时长解析、以及各类非法配置 fail fast。
func TestAlertingConfig(t *testing.T) {
	// 缺省：关闭 + 默认阈值/时长（阈值 0 视为未填写 → 回落 1，否则条件恒假等于静默失效）。
	c, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("parse default: %v", err)
	}
	if c.Alerting.Enabled {
		t.Fatal("alerting 缺省应关闭")
	}
	if c.AlertConfig.MinHealthy != 1 {
		t.Fatalf("缺省 min_healthy 应为 1，got %d", c.AlertConfig.MinHealthy)
	}
	if c.AlertConfig.Cooldown != 30*time.Minute {
		t.Fatalf("缺省 cooldown 应为 30m，got %v", c.AlertConfig.Cooldown)
	}
	if c.AlertConfig.Interval != time.Minute {
		t.Fatalf("缺省 interval 应为 1m，got %v", c.AlertConfig.Interval)
	}
	if c.AlertConfig.Enabled {
		t.Fatal("缺省 alerting 配置不应启用")
	}

	// 正常配置：字段落到 alert.Config。
	c, err = ParseConfig([]byte(`{"alerting":{
		"enabled":true,"webhook_url":"https://hooks.example/alert","secret":"k",
		"min_healthy":2,"cooldown":"15m","interval":"30s"}}`))
	if err != nil {
		t.Fatalf("parse valid: %v", err)
	}
	a := c.AlertConfig
	if !a.Enabled || a.WebhookURL != "https://hooks.example/alert" || a.Secret != "k" ||
		a.MinHealthy != 2 || a.Cooldown != 15*time.Minute || a.Interval != 30*time.Second {
		t.Fatalf("alert.Config 解析错误: %+v", a)
	}

	// 非法配置一律 fail fast。
	bad := []struct {
		name string
		body string
	}{
		{"启用但缺 webhook_url", `{"alerting":{"enabled":true}}`},
		{"webhook_url 非法 scheme", `{"alerting":{"enabled":true,"webhook_url":"ftp://x/y"}}`},
		{"webhook_url 缺主机", `{"alerting":{"enabled":true,"webhook_url":"https:///path"}}`},
		{"cooldown 非法", `{"alerting":{"cooldown":"半小时"}}`},
		{"cooldown 非正", `{"alerting":{"cooldown":"-5m"}}`},
		{"interval 非法", `{"alerting":{"interval":"abc"}}`},
		{"interval 非正", `{"alerting":{"interval":"0s"}}`},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseConfig([]byte(tc.body)); err == nil {
				t.Fatal("非法配置应报错")
			}
		})
	}
}
