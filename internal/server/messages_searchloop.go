// messages_searchloop.go —— /v1/messages 的 web_search「模型主动调用」模式，
// 与 responses_searchloop.go 同一套循环机制（注入合成工具 → 模型调用 → 网关
// 执行检索回填 → 重发上游），差异只在最终输出形态：Anthropic 事件序列里以
// server_tool_use + web_search_tool_result 内容块呈现每次实际执行的检索。
//
// 回放实现：最终轮的聚合结果合成为 OpenAI SSE 帧序列，喂给既有 messagesRenderer
// （messagesSink）——零新渲染代码，thinking/text/tool_use 块与 usage 语义全部
// 复用；已执行检索通过 meta.webSearch.execs（指针共享）驱动 emitSearchPrelude
// 逐对输出。
package server

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/reqlog"
)

// runMessagesSearchLoop /v1/messages 的模型主动调用编排。返回 true = 已完整
// 写出客户端响应；false = 已降级（合成工具剥除，调用方走正常 runTurn）。
func (h *Handler) runMessagesSearchLoop(ctx context.Context, w http.ResponseWriter,
	st *chatStat, p *turnParams, meta messagesMeta) bool {
	ws := meta.webSearch
	p.body = injectSearchFunctionTool(p.body)
	for round := 0; round <= maxSearchRounds; round++ {
		resp, err := h.searchRoundUpstream(ctx, st, p)
		if err != nil {
			log.Printf("WARN: [server] messages: search loop round %d upstream failed: %v -> falling back to normal turn", round, err)
			p.body = stripSearchFunctionTool(p.body)
			return false
		}
		searchCalls, otherCalls, err := partitionSearchCalls(resp)
		if err != nil {
			log.Printf("WARN: [server] messages: search loop partition: %v -> falling back", err)
			p.body = stripSearchFunctionTool(p.body)
			return false
		}
		budgetHit := ws.maxUses > 0 && len(ws.execs) >= ws.maxUses
		if len(searchCalls) == 0 || otherCalls > 0 || budgetHit {
			if otherCalls > 0 && len(searchCalls) > 0 {
				log.Printf("WARN: [server] messages: model mixed web_search with client tool calls; dropping %d search call(s)",
					len(searchCalls))
			}
			if budgetHit && len(searchCalls) > 0 {
				log.Printf("WARN: [server] messages: web_search max_uses=%d exhausted; replaying final round without new searches", ws.maxUses)
			}
			h.replayMessagesAggregate(w, st, meta, resp)
			return true
		}
		if round == maxSearchRounds {
			log.Printf("WARN: [server] messages: search loop budget exhausted after %d rounds, falling back", round)
			p.body = appendSystemNote(stripSearchFunctionTool(p.body),
				"The web_search tool is no longer available. Answer directly using the search results already present in the conversation.")
			return false
		}
		if err := h.appendSearchRound(p, searchCalls, &ws.execs); err != nil {
			log.Printf("WARN: [server] messages: search loop append round: %v -> falling back", err)
			p.body = stripSearchFunctionTool(p.body)
			return false
		}
	}
	p.body = stripSearchFunctionTool(p.body)
	return false
}

// replayMessagesAggregate 把最终轮聚合结果写回客户端：流式 = 合成 OpenAI SSE
// 帧喂给既有 messagesRenderer（execs 驱动 server_tool_use 前导块）；非流式 =
// 直接 renderNonStream。usage/状态记账与主管线同口径。
func (h *Handler) replayMessagesAggregate(w http.ResponseWriter, st *chatStat,
	meta messagesMeta, resp map[string]any) {
	st.status = http.StatusOK
	st.outcome = reqlog.OutcomeSuccess
	st.toks = completionTokens(resp)
	renderer := messagesRenderer{meta: meta}
	if !meta.stream {
		renderer.renderNonStream(w, resp)
		return
	}
	if err := renderer.renderStream(w, synthesizedAggregateReader(resp), nil, nil, nil); err != nil {
		log.Printf("WARN: [server] messages: search loop replay render: %v", err)
	}
}

// synthesizedAggregateReader 把聚合响应还原成一小段合法的 OpenAI SSE 流：
// role 帧 → reasoning 帧 → content 帧 → 完整 tool_calls 帧 → finish+usage 帧
// → [DONE]。messagesSink 按既有状态机消费，产出与真实流一致的 Anthropic 事件。
func synthesizedAggregateReader(resp map[string]any) *bytes.Reader {
	var sb bytes.Buffer
	writeChunk := func(obj map[string]any) {
		b, _ := json.Marshal(obj)
		sb.WriteString("data: " + string(b) + "\n\n")
	}
	base := map[string]any{"id": "chatcmpl-replay", "object": "chat.completion.chunk",
		"created": time.Now().Unix(), "model": "replay"}
	writeChunk(withDelta(base, map[string]any{"role": "assistant"}))
	msg, hasMsg := firstMessage(resp)
	if hasMsg {
		if rc, _ := msg["reasoning_content"].(string); rc != "" {
			writeChunk(withDelta(base, map[string]any{"reasoning_content": rc}))
		}
		if txt, _ := msg["content"].(string); txt != "" {
			writeChunk(withDelta(base, map[string]any{"content": txt}))
		}
		for i, tc := range toolCallsOf(msg) {
			writeChunk(withDelta(base, map[string]any{"tool_calls": []any{withIndex(tc, i)}}))
		}
	}
	fin := map[string]any{"delta": map[string]any{}}
	if fr := firstFinishReason(resp); fr != "" {
		fin["finish_reason"] = fr
	}
	if u, ok := resp["usage"].(map[string]any); ok {
		fin["usage"] = u
	}
	writeChunk(withDelta(base, fin))
	sb.WriteString("data: [DONE]\n\n")
	return bytes.NewReader(sb.Bytes())
}

func withDelta(base map[string]any, delta map[string]any) map[string]any {
	out := make(map[string]any, len(base)+2)
	for k, v := range base {
		out[k] = v
	}
	out["choices"] = []any{map[string]any{"index": 0, "delta": delta}}
	return out
}

func withIndex(tc map[string]any, i int) map[string]any {
	out := make(map[string]any, len(tc)+1)
	for k, v := range tc {
		out[k] = v
	}
	out["index"] = i
	return out
}
