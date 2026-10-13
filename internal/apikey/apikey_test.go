package apikey

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func boolPtr(b bool) *bool { return &b }

// req 造一个带 Authorization 头的请求。
func req(key string) *http.Request {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	return r
}

// ---------------------------------------------------------------- 校验 ----

func TestNewTableValidation(t *testing.T) {
	cases := []struct {
		name    string
		entries []Entry
		wantErr bool
	}{
		{"空配置合法", nil, false},
		{"正常条目", []Entry{{Key: "sk-a", Name: "a"}}, false},
		{"启用中但密钥为空", []Entry{{Key: ""}}, true},
		{"启用中但密钥全空白", []Entry{{Key: "   "}}, true},
		{"密钥含空格", []Entry{{Key: "sk a"}}, true},
		{"密钥含通配符", []Entry{{Key: "sk-*"}}, true},
		{"重复密钥", []Entry{{Key: "sk-a"}, {Key: "sk-a"}}, true},
		{"realm 非法", []Entry{{Key: "sk-a", Realm: "us"}}, true},
		{"realm cn", []Entry{{Key: "sk-a", Realm: "cn"}}, false},
		{"realm global", []Entry{{Key: "sk-a", Realm: "global"}}, false},
		{"models 含空模式", []Entry{{Key: "sk-a", Models: []string{"glm-*", " "}}}, true},
		// 禁用条目允许空密钥（"先禁用、回头换密钥"是正常操作），但 realm 仍要合法。
		{"禁用条目允许空密钥", []Entry{{Key: "", Enabled: boolPtr(false)}}, false},
		{"禁用条目 realm 仍校验", []Entry{{Key: "", Enabled: boolPtr(false), Realm: "us"}}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewTable(tc.entries, "")
			if (err != nil) != tc.wantErr {
				t.Fatalf("err=%v wantErr=%v", err, tc.wantErr)
			}
		})
	}
}

func TestFingerprintStableAndNonRevealing(t *testing.T) {
	a := Fingerprint("sk-secret")
	if a != Fingerprint("sk-secret") {
		t.Fatal("指纹必须稳定")
	}
	if a == Fingerprint("sk-secret2") {
		t.Fatal("不同密钥指纹应不同")
	}
	if len(a) != 16 {
		t.Fatalf("指纹应为 sha256 前 8 字节 hex（16 字符），got %d", len(a))
	}
	if a == "sk-secret" {
		t.Fatal("指纹不得等于密钥原文")
	}
}

// ---------------------------------------------------------------- 通配 ----

func TestGlobMatch(t *testing.T) {
	cases := []struct {
		pat, s string
		want   bool
	}{
		{"glm-5.2", "glm-5.2", true},
		{"glm-5.2", "glm-5.3", false},
		{"glm-*", "glm-5.2", true},
		{"glm-*", "glm-", true},
		{"*", "anything", true},
		{"*", "", true},
		{"deepseek*", "deepseek-v4.1-flash", true},
		{"*-flash", "deepseek-v4.1-flash", true},
		{"*-flash", "deepseek-v4.1", false},
		{"g?m-5.2", "glm-5.2", true},
		{"g?m-5.2", "gllm-5.2", false},
		{"a*b*c", "axxbyyc", true},
		{"a*b*c", "axxbyy", false},
		{"", "", true},
		{"", "x", false},
	}
	for _, tc := range cases {
		if got := globMatch(tc.pat, tc.s); got != tc.want {
			t.Errorf("globMatch(%q,%q)=%v want %v", tc.pat, tc.s, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------- 白名单 ----

func TestAllows(t *testing.T) {
	id := Identity{models: []string{"glm-*", "global:gpt-*"}, configured: true}
	cases := []struct {
		eff, bare string
		want      bool
	}{
		{"glm-5.2", "glm-5.2", true},        // 裸名匹配裸模式
		{"cn:glm-5.2", "glm-5.2", true},     // 带 cn: 前缀仍按裸名匹配裸模式
		{"global:gpt-6", "gpt-6", true},     // 带前缀模式比带前缀名
		{"gpt-6", "gpt-6", false},           // 前缀模式不匹配裸名
		{"global:glm-5.2", "glm-5.2", true}, // 裸模式对任何前缀都按裸名比
		{"deepseek-v4.1", "deepseek-v4.1", false},
	}
	for _, tc := range cases {
		if got := id.Allows(tc.eff, tc.bare); got != tc.want {
			t.Errorf("Allows(%q,%q)=%v want %v", tc.eff, tc.bare, got, tc.want)
		}
	}
	// 大小写不敏感。
	if !id.Allows("GLM-5.2", "GLM-5.2") {
		t.Error("白名单匹配应大小写不敏感")
	}
	// 未配置白名单 = 不限制。
	var open Identity
	if !open.Allows("anything", "anything") {
		t.Error("空白名单应放行一切")
	}
}

// ---------------------------------------------------------------- 鉴权 ----

func TestAuthenticateNoConfigAllowsEverything(t *testing.T) {
	tbl, err := NewTable(nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if tbl.Required() {
		t.Fatal("空配置不应要求鉴权")
	}
	id, ok := tbl.Authenticate(req(""))
	if !ok {
		t.Fatal("不鉴权模式应恒放行")
	}
	if id.Configured() {
		t.Fatal("不鉴权模式身份应为零值")
	}
}

func TestAuthenticateLegacySingleKey(t *testing.T) {
	tbl, err := NewTable(nil, "legacy-key")
	if err != nil {
		t.Fatal(err)
	}
	if !tbl.Required() {
		t.Fatal("配了顶层 api_key 应要求鉴权")
	}
	if _, ok := tbl.Authenticate(req("legacy-key")); !ok {
		t.Fatal("正确密钥应通过")
	}
	if _, ok := tbl.Authenticate(req("wrong")); ok {
		t.Fatal("错误密钥应拒绝")
	}
	if _, ok := tbl.Authenticate(req("")); ok {
		t.Fatal("缺头应拒绝")
	}
	// 单密钥身份为零值（不限制模型/域）——老配置语义不变。
	id, _ := tbl.Authenticate(req("legacy-key"))
	if id.Configured() || id.RealmOverride() != "" || !id.Allows("x", "x") {
		t.Fatal("顶层单 key 身份应不受限")
	}
}

func TestAuthenticateMultiKeyAndLegacyRejected(t *testing.T) {
	tbl, err := NewTable([]Entry{
		{Key: "sk-codex", Name: "Codex", Models: []string{"gpt-*"}, Realm: RealmGlobal},
		{Key: "sk-phone", Name: "手机端", Models: []string{"glm-*"}},
		{Key: "sk-off", Name: "停用", Enabled: boolPtr(false)},
	}, "legacy-key")
	if err != nil {
		t.Fatal(err)
	}

	// 每把钥匙命中自己的条目。
	id, ok := tbl.Authenticate(req("sk-codex"))
	if !ok || id.Name != "Codex" || id.RealmOverride() != RealmGlobal {
		t.Fatalf("sk-codex 身份错误: ok=%v id=%+v", ok, id)
	}
	if id.Fingerprint != Fingerprint("sk-codex") {
		t.Fatal("身份应带正确的指纹")
	}
	if !id.Allows("gpt-6", "gpt-6") || id.Allows("glm-5.2", "glm-5.2") {
		t.Fatal("sk-codex 白名单应为 gpt-*")
	}

	id2, ok := tbl.Authenticate(req("sk-phone"))
	if !ok || id2.Name != "手机端" || id2.RealmOverride() != "" {
		t.Fatalf("sk-phone 身份错误: ok=%v id=%+v", ok, id2)
	}

	// 禁用条目不可命中。
	if _, ok := tbl.Authenticate(req("sk-off")); ok {
		t.Fatal("禁用条目不应通过")
	}

	// 关键语义：配了 api_keys 之后，顶层 api_key 不再放行 /v1/*——
	// 否则"收窄权限"可被万能钥匙绕过。
	if _, ok := tbl.Authenticate(req("legacy-key")); ok {
		t.Fatal("配了 api_keys 后顶层 api_key 不应再放行")
	}

	if _, ok := tbl.Authenticate(req("nope")); ok {
		t.Fatal("未知密钥应拒绝")
	}
	if _, ok := tbl.Authenticate(req("")); ok {
		t.Fatal("缺头应拒绝")
	}
}

func TestEntriesAndLookup(t *testing.T) {
	tbl, err := NewTable([]Entry{
		{Key: "sk-a", Name: "A"},
		{Key: "sk-off", Name: "停用", Enabled: boolPtr(false)},
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	got := tbl.Entries()
	if len(got) != 1 || got[0].Key != "sk-a" {
		t.Fatalf("Entries 应只含启用中的条目，got %+v", got)
	}
	if e, ok := tbl.LookupFingerprint(Fingerprint("sk-a")); !ok || e.Name != "A" {
		t.Fatalf("按指纹反查失败: %+v %v", e, ok)
	}
	if _, ok := tbl.LookupFingerprint("deadbeefdeadbeef"); ok {
		t.Fatal("未知指纹不应命中")
	}
	// nil 表上的方法必须安全（面板在未配置 api_keys 时会走到）。
	var nilTbl *Table
	if nilTbl.Required() || nilTbl.Entries() != nil {
		t.Fatal("nil 表方法应安全返回")
	}
	if _, ok := nilTbl.Authenticate(req("x")); !ok {
		t.Fatal("nil 表 = 未配置 = 不鉴权，应放行")
	}
}
