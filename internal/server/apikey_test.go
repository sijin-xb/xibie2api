package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/apikey"
	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/keyusage"
)

// 本文件锁定多 API Key（api_keys）的四条行为：鉴权口径、模型白名单本地拒绝、
// 出口绑定覆盖 realm 前缀、逐 Key 台账归属。全部经真实入口（NewHandler + ServeHTTP
// 或 prepareTurn）验证，不直接调内部判定函数，避免"测了实现没测行为"。

func apiKeyTable(t *testing.T, entries []apikey.Entry, legacy string) *apikey.Table {
	t.Helper()
	tbl, err := apikey.NewTable(entries, legacy)
	if err != nil {
		t.Fatalf("NewTable: %v", err)
	}
	return tbl
}

func postChat(h *Handler, key, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	return rec
}

// TestAPIKeyAuthRejectsUnauthorized 覆盖 401 口径：错误密钥、缺头、以及
// "配了 api_keys 之后顶层 api_key 不再放行 /v1/*"这条关键语义。
func TestAPIKeyAuthRejectsUnauthorized(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	tbl := apiKeyTable(t, []apikey.Entry{{Key: "sk-ok", Name: "OK"}}, "legacy-key")
	h := NewHandler(Config{Pool: p, Upstream: up, KeyTable: tbl})

	const body = `{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	for _, tc := range []struct {
		name, key string
	}{
		{"错误密钥", "sk-wrong"},
		{"缺头", ""},
		{"顶层 api_key 不再放行", "legacy-key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if rec := postChat(h, tc.key, body); rec.Code != http.StatusUnauthorized {
				t.Fatalf("code=%d want 401 body=%s", rec.Code, rec.Body)
			}
		})
	}
	// 配置中的密钥放行。
	if rec := postChat(h, "sk-ok", body); rec.Code != http.StatusOK {
		t.Fatalf("正确密钥应放行，code=%d body=%s", rec.Code, rec.Body)
	}
}

// TestAPIKeyModelWhitelistRejectedLocally 锁定：白名单外的模型在网关本地 400，
// **且不触达上游**（不选号、不出站、不消耗额度）。
func TestAPIKeyModelWhitelistRejectedLocally(t *testing.T) {
	up, calls := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	tbl := apiKeyTable(t, []apikey.Entry{{Key: "sk-cheap", Name: "省钱", Models: []string{"glm-*"}}}, "")
	h := NewHandler(Config{Pool: p, Upstream: up, KeyTable: tbl})

	rec := postChat(h, "sk-cheap", `{"model":"deepseek-v4.1","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("越权模型应 400，code=%d body=%s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "model_not_allowed") {
		t.Fatalf("错误码应为 model_not_allowed，body=%s", rec.Body)
	}
	if n := len(calls()); n != 0 {
		t.Fatalf("越权请求不得触达上游，got %d 次调用", n)
	}

	// 白名单内的模型正常放行。
	rec = postChat(h, "sk-cheap", `{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("白名单内模型应放行，code=%d body=%s", rec.Code, rec.Body)
	}
	if n := len(calls()); n != 1 {
		t.Fatalf("放行请求应触达上游一次，got %d", n)
	}
}

// TestAPIKeyWhitelistMatchesAliasedModel 锁定：别名命中的目标名参与白名单判定
// （否则"给别名起的名字"会成为绕过白名单的后门）。
func TestAPIKeyWhitelistMatchesAliasedModel(t *testing.T) {
	up, calls := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	tbl := apiKeyTable(t, []apikey.Entry{{Key: "sk-cheap", Models: []string{"glm-*"}}}, "")
	aliases, err := NewAliasTable([]ModelAlias{{From: "my-cheap", To: "deepseek-v4.1"}})
	if err != nil {
		t.Fatal(err)
	}
	h := NewHandler(Config{Pool: p, Upstream: up, KeyTable: tbl, ModelAliases: aliases})

	rec := postChat(h, "sk-cheap", `{"model":"my-cheap","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("别名指向白名单外模型应 400，code=%d body=%s", rec.Code, rec.Body)
	}
	if n := len(calls()); n != 0 {
		t.Fatalf("越权请求不得触达上游，got %d", n)
	}
}

// TestAPIKeyRealmOverride 锁定：条目的 realm 绑定覆盖模型名上的 [realm:] 前缀，
// 且覆盖发生在 modelRate / 选号域之前。
func TestAPIKeyRealmOverride(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	tbl := apiKeyTable(t, []apikey.Entry{{Key: "sk-g", Name: "只走国际", Realm: apikey.RealmGlobal}}, "")
	h := NewHandler(Config{Pool: p, Upstream: up, KeyTable: tbl})

	cases := []struct{ name, model string }{
		{"无前缀被绑到 global", "glm-5.2"},
		{"cn: 前缀被覆盖", "cn:glm-5.2"},
		{"global: 前缀保持一致", "global:glm-5.2"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"` + tc.model + `","stream":true,"messages":[{"role":"user","content":"hi"}]}`
			r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
			r.Header.Set("Authorization", "Bearer sk-g")
			id, ok := h.authenticate(r)
			if !ok {
				t.Fatal("鉴权应通过")
			}
			r = r.WithContext(context.WithValue(r.Context(), apiKeyIdentityKey{}, id))

			st, tp, derr := h.prepareTurn(r, []byte(body), openAIRenderer{})
			if derr != nil {
				t.Fatalf("不应被拒: %+v", derr)
			}
			defer st.done()
			if tp.realm != apikey.RealmGlobal {
				t.Fatalf("realm=%q want global", tp.realm)
			}
			if tp.bareModel != "glm-5.2" {
				t.Fatalf("bareModel=%q want glm-5.2", tp.bareModel)
			}
			if tp.keyFP != apikey.Fingerprint("sk-g") || tp.keyName != "只走国际" {
				t.Fatalf("身份未透传到 turnParams: fp=%q name=%q", tp.keyFP, tp.keyName)
			}
		})
	}
}

// TestAPIKeyMessagesEnvelope 锁定：/v1/messages 的越权拒绝走 Anthropic 错误信封
// （而不是 OpenAI 信封），且 401 也是 Anthropic 口径。
func TestAPIKeyMessagesEnvelope(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	tbl := apiKeyTable(t, []apikey.Entry{{Key: "sk-cheap", Models: []string{"glm-*"}}}, "")
	h := NewHandler(Config{Pool: p, Upstream: up, KeyTable: tbl})

	const body = `{"model":"deepseek-v4.1","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`

	// 401：Anthropic 信封的 type 是 authentication_error。
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusUnauthorized || !strings.Contains(rec.Body.String(), "authentication_error") {
		t.Fatalf("401 应为 Anthropic 信封，code=%d body=%s", rec.Code, rec.Body)
	}

	// 400：越权模型在 Anthropic 路径上仍走 Anthropic 信封。注意 Anthropic 错误
	// 信封本身**没有 code 字段**（与仓库既有 writeMessagesError 口径一致），因此
	// 机器可读码只在 OpenAI / Responses 路径上可见；这里断言状态、信封 type 与
	// 文案里出现被拒的模型名。
	r = httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer sk-cheap")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("code=%d want 400 body=%s", rec.Code, rec.Body)
	}
	if !strings.Contains(rec.Body.String(), "invalid_request_error") || !strings.Contains(rec.Body.String(), "deepseek-v4.1") {
		t.Fatalf("越权拒绝应为 Anthropic 信封并指出模型名，body=%s", rec.Body)
	}
}

// TestAPIKeyLedgerAttribution 锁定：逐 Key 台账按指纹归属，且顶层单 key 不产生台账行。
func TestAPIKeyLedgerAttribution(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	tbl := apiKeyTable(t, []apikey.Entry{{Key: "sk-tracked", Name: "被追踪"}}, "")
	ledger := keyusage.New("")
	h := NewHandler(Config{Pool: p, Upstream: up, KeyTable: tbl, KeyUsage: ledger})

	if rec := postChat(h, "sk-tracked", `{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`); rec.Code != http.StatusOK {
		t.Fatalf("code=%d body=%s", rec.Code, rec.Body)
	}
	snap := ledger.Snapshot(0, nil)
	if len(snap.Rows) != 1 {
		t.Fatalf("应有一行台账，got %+v", snap.Rows)
	}
	row := snap.Rows[0]
	if row.Fingerprint != apikey.Fingerprint("sk-tracked") || row.Name != "被追踪" {
		t.Fatalf("台账归属错误: %+v", row)
	}
	if row.Requests != 1 {
		t.Fatalf("请求数应为 1，got %d", row.Requests)
	}
}

// TestAPIKeyLegacyModeNoLedgerRow 锁定：未配置 api_keys 时（老配置）行为不变，
// 且不产生任何逐 Key 台账行。
func TestAPIKeyLegacyModeNoLedgerRow(t *testing.T) {
	up, _ := staticRegressionUpstream(200, sseOK, true)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	tbl := apiKeyTable(t, nil, "legacy-key")
	ledger := keyusage.New("")
	h := NewHandler(Config{Pool: p, Upstream: up, KeyTable: tbl, KeyUsage: ledger})

	if rec := postChat(h, "legacy-key", `{"model":"glm-5.2","stream":true,"messages":[{"role":"user","content":"hi"}]}`); rec.Code != http.StatusOK {
		t.Fatalf("老配置语义应不变，code=%d body=%s", rec.Code, rec.Body)
	}
	if snap := ledger.Snapshot(0, nil); len(snap.Rows) != 0 {
		t.Fatalf("顶层单 key 不应产生台账行，got %+v", snap.Rows)
	}
}
