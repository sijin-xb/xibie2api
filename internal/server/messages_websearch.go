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
func (h *Handler) runWebSearch(ctx context.Context, ws *messagesWebSearch, p *turnParams) {
	if ws == nil {
		return
	}
	c := h.cfg.WebSearch
	if c == nil || !c.Enabled() {
		ws.err = "web search is not enabled on this gateway"
		log.Printf("WARN: [server] messages: web_search declared but gateway web_search is disabled")
		return
	}
	if ws.query == "" {
		ws.err = "no search query could be derived from the request"
		return
	}
	if p != nil && h.cfg.WebSearchQueryMode == "model" {
		if q, ok := h.generateSearchQuery(ctx, p, ws.query); ok {
			ws.query = q
		}
	}
	max := ws.maxUses
	if max <= 0 {
		max = 5
	}
	results, err := c.Search(ctx, ws.query, max, ws.allowedDomains, ws.blockedDomains)
	if err != nil {
		ws.err = err.Error()
		log.Printf("WARN: [server] messages: web_search backend=%s query=%q failed: %v", c.Backend(), ws.query, err)
		return
	}
	ws.results = results
	log.Printf("INFO: [server] messages: web_search backend=%s query=%q results=%d", c.Backend(), ws.query, len(results))
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
	if h.cfg.WebSearchQueryModel != "" {
		model = h.cfg.WebSearchQueryModel
	}
	acct := h.cfg.Pool.PickExcludingForRealm(map[string]bool{}, model, p.realm)
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
