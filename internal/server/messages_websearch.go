// messages_websearch.go 把 Anthropic web_search 服务端工具接到网关自有的检索后端：
// 出站前实际检索 → 结果回填到 messagesWebSearch（渲染层据此产出 server_tool_use +
// web_search_tool_result 内容块）→ 结果摘要作为一条 system 消息注入出站 prompt，
// 让上游模型基于实时结果作答。
package server

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/prompt"
	"github.com/linguo2625469/workbuddy2api-panel/internal/websearch"
)

// runWebSearch 执行检索并把结果/错误回填到 ws。检索失败**不**阻断请求：把错误
// 渲染成 web_search_tool_result_error，模型照常作答（与 Anthropic 语义一致——
// 检索失败是结果块里的错误，不是整个请求失败）。
//
// 查询串：query_mode="model"（缺省）时先用模型把用户问题提炼成一条搜索查询
// （Anthropic 的 web_search 本就是模型决定查询词；原始中文问句直接打搜索引擎
// 命中率很差），失败则回落到原始用户文本；query_mode="raw" 时直接用原始文本。
//
// TTL 缓存（webSearchCache）：agent 客户端（Codex / Claude Code）的工具循环会在
// 同一轮里逐个工具调用重发请求——最后一条 user 消息不变、查询串相同，无缓存时
// 一个 273 步的循环就是 273 次同查询检索。缓存键为查询原文（命中时连查询提炼的
// 上游调用也一并省掉）；命中只回填结果（fresh=false），渲染层不再产出新的
// web_search_call / server_tool_use 条目。
func (h *Handler) runWebSearch(ctx context.Context, ws *messagesWebSearch, p *turnParams) {
	if ws == nil {
		return
	}
	c := h.cfg.WebSearch
	if c == nil || !c.Enabled() {
		ws.err = "web search is not enabled on this gateway"
		log.Printf("WARN: [server] web_search: declared but gateway web_search is disabled")
		return
	}
	if ws.query == "" {
		ws.err = "no search query could be derived from the request"
		return
	}
	if hit, ok := h.searchCache.get(ws.searchKey()); ok {
		ws.results = hit.results
		ws.err = hit.err
		return
	}
	if p != nil && h.cfg.WebSearchQueryMode == "model" {
		if q, ok := h.generateSearchQuery(ctx, p, ws.query); ok {
			ws.query = q
		}
	}
	// max_uses caps how many times the tool may run per turn, not how many
	// results one run returns; result count belongs to web_search.max_results.
	results, err := c.Search(ctx, ws.query, 0, ws.allowedDomains, ws.blockedDomains)
	if err != nil {
		ws.err = err.Error()
		ws.fresh = true
		h.searchCache.put(ws.searchKey(), nil, ws.err)
		log.Printf("WARN: [server] web_search: backend=%s query=%q failed: %v", c.Backend(), ws.query, err)
		return
	}
	ws.results = results
	ws.fresh = true
	h.searchCache.put(ws.searchKey(), results, "")
	log.Printf("INFO: [server] web_search: backend=%s query=%q results=%d", c.Backend(), ws.query, len(results))
}

// searchKey 缓存键：查询原文 + 域过滤（过滤不同则结果不同）。用提炼前的原文做键，
// 命中时连带跳过查询提炼；域过滤拼进键防止串味。
func (ws *messagesWebSearch) searchKey() string {
	return ws.query + "\x00" + strings.Join(ws.allowedDomains, ",") + "\x00" + strings.Join(ws.blockedDomains, ",")
}

// webSearchCacheEntry 一条缓存结果（err 非空 = 上次检索失败，同样缓存以防风暴）。
type webSearchCacheEntry struct {
	results []websearch.Result
	err     string
	expires time.Time
}

// webSearchCache 查询级 TTL 缓存（容量上限防无界增长；简单整表淘汰即可——
// 命中率场景是「同轮重复」，不需要 LRU）。
type webSearchCache struct {
	mu      sync.Mutex
	ttl     time.Duration
	maxKeys int
	entries map[string]webSearchCacheEntry
}

func newWebSearchCache(ttl time.Duration, maxKeys int) webSearchCache {
	if ttl <= 0 {
		ttl = 10 * time.Minute
	}
	if maxKeys <= 0 {
		maxKeys = 64
	}
	return webSearchCache{ttl: ttl, maxKeys: maxKeys, entries: map[string]webSearchCacheEntry{}}
}

func (c *webSearchCache) get(key string) (webSearchCacheEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok || time.Now().After(e.expires) {
		if ok {
			delete(c.entries, key)
		}
		return webSearchCacheEntry{}, false
	}
	return e, true
}

func (c *webSearchCache) put(key string, results []websearch.Result, err string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.entries) >= c.maxKeys {
		now := time.Now()
		for k, e := range c.entries {
			if now.After(e.expires) {
				delete(c.entries, k)
			}
		}
		// 过期清扫后仍满：整表淘汰（重建成本低于 LRU 复杂度）。
		if len(c.entries) >= c.maxKeys {
			c.entries = map[string]webSearchCacheEntry{}
		}
	}
	c.entries[key] = webSearchCacheEntry{results: results, err: err, expires: time.Now().Add(c.ttl)}
}

// searchQuerySystemPrompt 查询提炼提示词：让模型输出一条搜索引擎友好的查询串。
const searchQuerySystemPrompt = "你是搜索查询生成器。根据用户问题，输出一条最适合网页搜索引擎的查询串。" +
	"要求：只输出查询串本身（英文关键词优先，技术名词/版本号/专有名词保留原文），" +
	"不要问句、不要标点、不要引号、不要任何解释。"

// generateSearchQuery 用一次轻量上游调用把用户问题提炼为搜索查询。任何失败都返回
// ok=false（调用方回落原始文本），绝不阻断主请求。
// generateSearchQuery 用一次轻量上游调用把用户问题提炼为搜索查询。任何失败都返回
// ok=false（调用方回落原始文本），绝不阻断主请求。
//
// max_tokens 给足 1024：上游模型恒有思维链（reasoning_content），预算过小会被思考
// 吃光、content 为空导致提炼失败（实测 64 会空、1024 稳定产出查询串）。
func (h *Handler) generateSearchQuery(ctx context.Context, p *turnParams, question string) (string, bool) {
	if h.cfg.Pool == nil || h.cfg.Upstream == nil {
		return "", false
	}
	model := p.bareModel
	realm := p.realm
	if h.cfg.WebSearchQueryModel != "" {
		// 配置可带 realm 前缀（cn:/global:）。上游只认裸名，且选号必须按前缀的 realm
		// ——否则会「按 global 前缀发 cn 号」并让带前缀的模型名出站（上游必拒）。
		realm, model = resolveModel(h.cfg.WebSearchQueryModel)
	}
	acct := h.cfg.Pool.PickExcludingForRealm(map[string]bool{}, model, realm)
	if acct == nil {
		return "", false
	}
	if !h.cfg.Pool.Acquire(acct.UID) {
		return "", false
	}
	defer h.cfg.Pool.Release(acct.UID)
	if acct.NeedsRefresh(h.cfg.RefreshSkew) {
		if err := h.cfg.Upstream.RefreshToken(acct); err != nil {
			return "", false
		}
	}
	body, err := json.Marshal(map[string]any{
		"model":       model,
		"stream":      true,
		"max_tokens":  1024,
		"temperature": 0,
		"messages": []any{
			map[string]any{"role": "system", "content": searchQuerySystemPrompt},
			map[string]any{"role": "user", "content": question},
		},
	})
	if err != nil {
		return "", false
	}
	qctx, cancel := context.WithTimeout(ctx, 25*time.Second)
	defer cancel()
	rc, status, _, err := h.cfg.Upstream.ChatStreamContext(qctx, acct, body, p.clientIP, p.chatMeta)
	if err != nil || status >= 400 || rc == nil {
		return "", false
	}
	defer rc.Close()
	txt := strings.TrimSpace(collectSSEText(rc))
	txt = strings.Trim(txt, "\"'` \t\r\n")
	txt = strings.Join(strings.Fields(txt), " ")
	if txt == "" || len(txt) > 300 {
		return "", false
	}
	return txt, true
}

// collectSSEText 从上游 OpenAI SSE 流里拼接 delta.content（忽略 reasoning_content、
// tool_calls、usage 帧）。解析失败的帧跳过；读取上限 1MiB 防呆。
func collectSSEText(rc io.Reader) string {
	var sb strings.Builder
	sc := bufio.NewScanner(rc)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var frame struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &frame); err != nil {
			continue
		}
		for _, ch := range frame.Choices {
			sb.WriteString(ch.Delta.Content)
		}
	}
	return sb.String()
}

// searchContextMessage 把检索结果拼成注入出站 prompt 的系统文本。
func searchContextMessage(results []websearch.Result) string {
	var sb strings.Builder
	sb.WriteString("以下是网关执行 web_search 得到的实时联网搜索结果。请基于这些结果回答用户，")
	sb.WriteString("并在引用信息处标注来源 URL；若结果不足以回答，请如实说明，不要臆造。\n")
	for i, r := range results {
		fmt.Fprintf(&sb, "\n[%d] %s\n%s\n", i+1, r.Title, r.URL)
		if r.Snippet != "" {
			sb.WriteString(r.Snippet)
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

// injectSearchContext 在 prepareTurn 之后把检索结果作为一条 system 消息插到出站
// body 的开头 system 块之后（复用 prompt.Append 的插入语义，不重排既有消息，也不
// 被 custom 模式的提示词替换冲掉——注入发生在提示词改写之后）。
func injectSearchContext(body []byte, results []websearch.Result) []byte {
	if len(results) == 0 {
		return body
	}
	return prompt.Append(body, searchContextMessage(results))
}
