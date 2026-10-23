// responses_searchloop.go —— /v1/responses 的 web_search「模型主动调用」模式。
//
// 旧模式（预检索）的问题：客户端声明 web_search 工具时，网关对**每个新 user
// 消息**无条件先检索一次——纯编码任务的整段任务书被当成查询词打给搜索引擎，
// 结果无用还污染 prompt；模型也失去「自己决定搜什么、搜几次」的能力。
//
// 本模式：出站 body 注入合成 function 工具 `web_search`（模型可调用）；
// 上游若调用它，本流**不出站给客户端**（Codex 无法执行该工具），由网关执行
// 检索 → 结果以 assistant(tool_calls)+tool 消息对回填 body → 重发上游 →
// 直到模型不再调用搜索（或达到轮次上限）。客户端最终只看到：web_search_call
// item（每轮实际执行的检索，UI 可见）+ 模型的最终输出，与 OpenAI 服务端
// web_search 的客户端视角一致。
//
// 实现约束：中间轮不经过 runTurn（那是给客户端渲染的路径），走与
// generateSearchQuery 同款的轻量直连（选号→刷新→ChatStreamContext→聚合）。
// 任何中间轮失败都降级为「剥掉合成工具 + 正常 runTurn」——已回填的检索结果
// 仍留在 body 里，模型照样能基于它作答。
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
	"github.com/linguo2625469/workbuddy2api-panel/internal/reqlog"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
	"github.com/linguo2625469/workbuddy2api-panel/internal/websearch"
)

// searchFunctionName 合成工具名。与 Codex 声明的内置工具同名：历史里若出现同名
// function_call，归一化层照常映射（本模式不会向客户端透出它，仅网关内部消化）。
const searchFunctionName = "web_search"

// maxSearchRounds 单次客户端请求内最多执行的检索轮数（每轮 = 一次上游往返）。
// 3 = 模型最多连续检索三次后必须给出回答，防退化循环烧 token。
const maxSearchRounds = 3

// searchRoundTimeout 单个中间轮的上游调用预算。
const searchRoundTimeout = 180 * time.Second

// searchToolDescription 合成工具描述：引导模型只在时效性/外部事实场景调用，
// 纯编码任务不搜（旧预检索模式的浪费点）。
const searchToolDescription = "Search the web for up-to-date information (news, releases, library docs, prices, anything you are unsure about). " +
	"Input is search query keywords; returns ranked results with title, url and snippet. " +
	"Use it when freshness or external facts matter; skip it for tasks solvable from the local context."

// searchExecRecord 一轮已执行的检索（回放时产出 web_search_call item）。
type searchExecRecord struct {
	query   string
	results []websearch.Result
	err     string
}

// runResponsesSearchLoop 模型主动调用模式的编排入口。返回 true = 已完整写出
// 客户端响应（调用方直接返回）；false = 已降级（合成工具已剥除，调用方走正常
// runTurn，body 中已回填的检索结果保留）。
func (h *Handler) runResponsesSearchLoop(ctx context.Context, w http.ResponseWriter,
	st *chatStat, p *turnParams, meta responsesMeta) bool {
	p.body = injectSearchFunctionTool(p.body)
	var executed []searchExecRecord
	for round := 0; round <= maxSearchRounds; round++ {
		resp, err := h.searchRoundUpstream(ctx, st, p)
		if err != nil {
			log.Printf("WARN: [server] responses: search loop round %d upstream failed: %v -> falling back to normal turn", round, err)
			p.body = stripSearchFunctionTool(p.body)
			return false
		}
		searchCalls, otherCalls, err := partitionSearchCalls(resp)
		if err != nil {
			log.Printf("WARN: [server] responses: search loop partition: %v -> falling back", err)
			p.body = stripSearchFunctionTool(p.body)
			return false
		}
		if len(searchCalls) == 0 || otherCalls > 0 {
			// 回答轮（或模型把搜索与客户端工具混发——客户端工具优先，丢弃搜索调用）。
			if otherCalls > 0 && len(searchCalls) > 0 {
				log.Printf("WARN: [server] responses: model mixed web_search with client tool calls; dropping %d search call(s)",
					len(searchCalls))
			}
			h.replayResponsesAggregate(w, st, meta, resp, executed)
			return true
		}
		if round == maxSearchRounds {
			// 轮次耗尽仍要搜：剥掉工具正常跑，已执行的检索结果都在 body 里。
			// 附引导 system：模型见历史里有 web_search 调用会倾向于继续调——
			// 工具已不在声明里，硬调会被标记修复还原成结构化调用泄漏给客户端
			//（渲染层另有兜底过滤），引导它直接基于已有结果作答。
			log.Printf("WARN: [server] responses: search loop budget exhausted after %d rounds, falling back", round)
			p.body = appendSystemNote(stripSearchFunctionTool(p.body),
				"The web_search tool is no longer available. Answer directly using the search results already present in the conversation.")
			return false
		}
		// 检索轮：执行全部搜索调用，结果回填 body，进入下一轮。
		if err := h.appendSearchRound(p, searchCalls, &executed); err != nil {
			log.Printf("WARN: [server] responses: search loop append round: %v -> falling back", err)
			p.body = stripSearchFunctionTool(p.body)
			return false
		}
	}
	p.body = stripSearchFunctionTool(p.body)
	return false
}

// searchToolCall 一个待执行的搜索调用（来自聚合响应）。
type searchToolCall struct {
	id    string
	query string
	args  string
}

// partitionSearchCalls 把聚合响应里的 tool_calls 分成（搜索调用, 其他调用）。
// err = 搜索调用的 arguments 非法 JSON 或缺 query 字段（此时无法安全继续，
// 由调用方降级）。
func partitionSearchCalls(resp map[string]any) ([]searchToolCall, int, error) {
	msg, ok := firstMessage(resp)
	if !ok {
		return nil, 0, nil
	}
	calls := toolCallsOf(msg)
	var searches []searchToolCall
	others := 0
	for _, tc := range calls {
		fn, _ := tc["function"].(map[string]any)
		if fn == nil {
			others++
			continue
		}
		name, _ := fn["name"].(string)
		if name != searchFunctionName {
			others++
			continue
		}
		id, _ := tc["id"].(string)
		args, _ := fn["arguments"].(string)
		var parsed struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(args), &parsed); err != nil || strings.TrimSpace(parsed.Query) == "" {
			return nil, 0, fmt.Errorf("web_search arguments invalid: %q", truncateForLog(args, 120))
		}
		searches = append(searches, searchToolCall{id: id, query: strings.TrimSpace(parsed.Query), args: args})
	}
	return searches, others, nil
}

// appendSearchRound 执行一批搜索调用并把「assistant(tool_calls) + tool 结果」
// 消息对追加进出站 body，executed 累积回放记录。
func (h *Handler) appendSearchRound(p *turnParams, calls []searchToolCall, executed *[]searchExecRecord) error {
	var obj map[string]any
	if err := json.Unmarshal(p.body, &obj); err != nil {
		return err
	}
	msgs, _ := obj["messages"].([]any)
	toolCalls := make([]any, 0, len(calls))
	var toolMsgs []any
	for _, c := range calls {
		rec := searchExecRecord{query: c.query}
		results, err := h.executeSearchTool(p, c.query)
		if err != nil {
			rec.err = err.Error()
			log.Printf("WARN: [server] responses: web_search query=%q failed: %v", c.query, err)
		} else {
			rec.results = results
			log.Printf("INFO: [server] responses: web_search query=%q results=%d", c.query, len(results))
		}
		*executed = append(*executed, rec)
		toolCalls = append(toolCalls, map[string]any{
			"id":   c.id,
			"type": "function",
			"function": map[string]any{
				"name":      searchFunctionName,
				"arguments": c.args,
			},
		})
		payload := map[string]any{}
		if rec.err != "" {
			payload["error"] = rec.err
		} else {
			items := make([]any, 0, len(rec.results))
			for _, r := range rec.results {
				items = append(items, map[string]any{"title": r.Title, "url": r.URL, "snippet": r.Snippet})
			}
			payload["results"] = items
		}
		b, _ := json.Marshal(payload)
		toolMsgs = append(toolMsgs, map[string]any{
			"role": "tool", "tool_call_id": c.id, "content": string(b),
		})
	}
	msgs = append(msgs, map[string]any{"role": "assistant", "tool_calls": toolCalls})
	msgs = append(msgs, toolMsgs...)
	obj["messages"] = msgs
	out, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	p.body = out
	return nil
}

// executeSearchTool 执行一次检索（带 TTL 缓存去重：同问只打一次引擎）。
func (h *Handler) executeSearchTool(p *turnParams, query string) ([]websearch.Result, error) {
	c := h.cfg.WebSearch
	if c == nil || !c.Enabled() {
		return nil, errors.New("web search backend disabled")
	}
	if hit, ok := h.searchCache.get(query); ok {
		if hit.err != "" {
			return nil, errors.New(hit.err)
		}
		return hit.results, nil
	}
	results, err := c.Search(context.Background(), query, 0, nil, nil)
	if err != nil {
		h.searchCache.put(query, nil, err.Error())
		return nil, err
	}
	h.searchCache.put(query, results, "")
	return results, nil
}

// appendSystemNote 在 body 消息末尾追加一条 system 提示（降级引导等）。
func appendSystemNote(body []byte, note string) []byte {
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return body
	}
	msgs, _ := obj["messages"].([]any)
	obj["messages"] = append(msgs, map[string]any{"role": "system", "content": note})
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}

// injectSearchFunctionTool 往出站 body 注入合成 web_search function 工具。
// 客户端已有同名工具（如 MCP 定义）时不注入——名字撞车时让位给客户端工具。
func injectSearchFunctionTool(body []byte) []byte {
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return body
	}
	tools, _ := obj["tools"].([]any)
	for _, t := range tools {
		if tool, ok := t.(map[string]any); ok {
			if fn, ok := tool["function"].(map[string]any); ok {
				if name, _ := fn["name"].(string); name == searchFunctionName {
					return body
				}
			}
		}
	}
	obj["tools"] = append(tools, map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        searchFunctionName,
			"description": searchToolDescription,
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Search query keywords",
					},
				},
				"required": []any{"query"},
			},
		},
	})
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}

// stripSearchFunctionTool 从出站 body 移除合成 web_search 工具（降级路径）。
func stripSearchFunctionTool(body []byte) []byte {
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return body
	}
	tools, ok := obj["tools"].([]any)
	if !ok {
		return body
	}
	kept := make([]any, 0, len(tools))
	for _, t := range tools {
		if tool, ok := t.(map[string]any); ok {
			if fn, ok := tool["function"].(map[string]any); ok {
				if name, _ := fn["name"].(string); name == searchFunctionName {
					continue
				}
			}
		}
		kept = append(kept, t)
	}
	if len(kept) == len(tools) {
		return body
	}
	if len(kept) == 0 {
		delete(obj, "tools")
	} else {
		obj["tools"] = kept
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}

// searchRoundUpstream 一个中间轮的直连上游调用（选号→刷新→出站→聚合），
// 记账与成功号标记与主管线同口径。单号失败（限流/传输错）换号重试至多
// searchRoundMaxPicks 次，仍失败返回 err（调用方降级到 runTurn——那里还有
// 完整的轮转与错误分类兜底）。
func (h *Handler) searchRoundUpstream(ctx context.Context, st *chatStat, p *turnParams) (map[string]any, error) {
	if h.cfg.Pool == nil || h.cfg.Upstream == nil {
		return nil, errors.New("pool/upstream unavailable")
	}
	tried := map[string]bool{}
	var lastErr error
	for pick := 0; pick < searchRoundMaxPicks; pick++ {
		acct := h.cfg.Pool.PickExcludingForRealm(tried, p.bareModel, p.realm)
		if acct == nil {
			break
		}
		tried[acct.UID] = true
		resp, err := h.searchRoundOnce(ctx, st, p, acct)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		log.Printf("WARN: [server] responses: search round pick %s failed: %v", acct.UID, err)
	}
	if lastErr == nil {
		lastErr = errors.New("no healthy account")
	}
	return nil, lastErr
}

// searchRoundMaxPicks 单个搜索轮最多尝试的账号数。
const searchRoundMaxPicks = 3

// searchRoundOnce 单账号执行一次搜索轮上游调用。
func (h *Handler) searchRoundOnce(ctx context.Context, st *chatStat, p *turnParams, acct *auth.Auth) (map[string]any, error) {
	if !h.cfg.Pool.Acquire(acct.UID) {
		return nil, errors.New("account lease contended")
	}
	defer h.cfg.Pool.Release(acct.UID)
	if acct.NeedsRefresh(h.cfg.RefreshSkew) {
		if err := h.cfg.Upstream.RefreshToken(acct); err != nil {
			return nil, fmt.Errorf("refresh: %w", err)
		}
	}
	rctx, cancel := context.WithTimeout(ctx, searchRoundTimeout)
	defer cancel()
	started := time.Now()
	rc, status, _, err := h.cfg.Upstream.ChatStreamContext(rctx, acct, p.body, p.clientIP, p.chatMeta)
	if err != nil || rc == nil {
		return nil, fmt.Errorf("transport: %v", err)
	}
	defer rc.Close()
	if status >= 400 {
		return nil, fmt.Errorf("upstream status %d", status)
	}
	resp, aerr := upstream.AggregateObserved(rc, func(string) {})
	if aerr != nil {
		return nil, fmt.Errorf("aggregate: %w", aerr)
	}
	h.recordAttempt(st, p, acct.UID, usageDeltaFromResponse(resp), 0, false, started, 0)
	h.cfg.Pool.NoteSuccess(acct.UID)
	if p.sessKey != "" && h.cfg.Session != nil {
		h.cfg.Session.Bind(p.sessKey, acct.UID)
	}
	return resp, nil
}

// replayResponsesAggregate 把聚合响应回放给客户端：流式 = 合成 SSE 事件序列
// （created/in_progress + web_search_call items + output items + completed）；
// 非流式 = 直接写 Responses 对象。end_turn:false 强制续跑语义与流式主管线
// 一致（classifyPendingNarration + trailing 防抖）。
func (h *Handler) replayResponsesAggregate(w http.ResponseWriter, st *chatStat,
	meta responsesMeta, resp map[string]any, executed []searchExecRecord) {
	obj := responsesFromAggregate(resp, meta)
	// 执行过的检索以 web_search_call item 形式置于 output 最前（UI 可见）。
	if len(executed) > 0 {
		wsItems := make([]any, 0, len(executed))
		for _, e := range executed {
			item := map[string]any{"id": "ws_" + randHex(16), "type": "web_search_call", "status": "completed"}
			if e.err == "" {
				item["action"] = map[string]any{"type": "search", "query": e.query}
			}
			wsItems = append(wsItems, item)
		}
		obj["output"] = append(wsItems, obj["output"].([]any)...)
	}
	applyReplayPhases(obj)
	force := replayForceFollowUp(obj, meta)
	if force {
		obj["end_turn"] = false
		log.Printf("INFO: [server] responses: end_turn=false forced on replay (pending narration without tool calls; trailing assistant msgs=%d)",
			meta.trailingAssistantMsgs)
	}
	st.status = http.StatusOK
	st.outcome = reqlog.OutcomeSuccess
	st.toks = completionTokens(resp)
	if !meta.stream {
		writeJSON(w, http.StatusOK, obj)
		return
	}
	hdr := w.Header()
	hdr.Set("Content-Type", "text/event-stream")
	hdr.Set("Cache-Control", "no-cache")
	hdr.Set("Connection", "keep-alive")
	hdr.Set("X-Accel-Buffering", "no")
	fl, _ := w.(http.Flusher)
	sink := newResponsesSink(w, fl, meta)
	_ = sink.ensureStarted()
	seq := 0
	for _, raw := range obj["output"].([]any) {
		item, _ := raw.(map[string]any)
		if item == nil {
			continue
		}
		_ = sink.emit("response.output_item.added", map[string]any{
			"output_index": seq, "item": item,
		})
		_ = sink.emit("response.output_item.done", map[string]any{
			"output_index": seq, "item": item,
		})
		seq++
	}
	obj["completed_at"] = time.Now().Unix()
	_ = sink.emit("response.completed", map[string]any{"response": obj})
	if fl != nil {
		fl.Flush()
	}
}

// applyReplayPhases 给回放 output 里的 message item 补 phase（与流式主管线
// 同口径：过程播报 → commentary，其余 → final_answer）。缺 phase 的消息在
// Codex 侧相位统计里是 null，与流式路径不一致。
func applyReplayPhases(obj map[string]any) {
	output, _ := obj["output"].([]any)
	for _, raw := range output {
		item, _ := raw.(map[string]any)
		if item == nil || item["type"] != "message" {
			continue
		}
		text := ""
		if content, ok := item["content"].([]any); ok && len(content) > 0 {
			if part, ok := content[0].(map[string]any); ok {
				text, _ = part["text"].(string)
			}
		}
		if classifyPendingNarration(text) {
			item["phase"] = "commentary"
		} else {
			item["phase"] = "final_answer"
		}
	}
}

// replayForceFollowUp 回放路径的 end_turn:false 判定：output 里唯一/最后的
// assistant message 是过程播报文本且没有工具调用 item。
func replayForceFollowUp(obj map[string]any, meta responsesMeta) bool {
	if meta.trailingAssistantMsgs >= maxForcedFollowUps {
		return false
	}
	output, _ := obj["output"].([]any)
	var lastText string
	sawTool := false
	for _, raw := range output {
		item, _ := raw.(map[string]any)
		if item == nil {
			continue
		}
		switch item["type"] {
		case "message":
			if content, ok := item["content"].([]any); ok && len(content) > 0 {
				if part, ok := content[0].(map[string]any); ok {
					lastText, _ = part["text"].(string)
				}
			}
		case "function_call", "custom_tool_call":
			sawTool = true
		}
	}
	return !sawTool && classifyPendingNarration(lastText)
}

// truncateForLog 面向日志的短截断。
func truncateForLog(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
