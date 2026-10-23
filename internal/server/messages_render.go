// messages_render.go 实现 Anthropic Messages API（/v1/messages）的响应渲染：
//   - messagesRenderer：turnRenderer 的 Anthropic 实现（流式 / 非流式 / 终态错误）；
//   - messagesSink：upstream.FrameSink 的 Anthropic 实现，把上游 OpenAI Chat
//     Completions SSE 帧翻译成 Anthropic SSE 事件序列；
//   - messagesFromAggregate：把聚合后的 OpenAI chat.completion 转换为 Anthropic 消息；
//   - writeMessagesError：Anthropic 错误信封。
//
// 边界纪律：本层只做「协议表示翻译」，不持有账号选择/轮转/惩罚（runTurn 拥有），
// 也不做 usage 记账（runTurn 从 chatStatsReader 读）。
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
	"github.com/linguo2625469/workbuddy2api-panel/internal/websearch"
)

// ---------------------------------------------------------------------------
// turnRenderer 实现
// ---------------------------------------------------------------------------

type messagesRenderer struct {
	meta messagesMeta
}

// renderStream 把已归一化的上游 SSE 流翻译为 Anthropic SSE 事件序列。
// hint 被忽略：Anthropic 无 gateway_hint 帧形态（错误以 event: error 表达）。
func (r messagesRenderer) renderStream(w http.ResponseWriter, stats io.Reader,
	_ func() upstream.HintContext, repair *upstream.MarkupRepair, onErrorFrame func(string)) error {
	// 推理静默期补 SSE 注释心跳（见 keepalive.go）：Anthropic 客户端同样按字节
	// 空闲判流断开，长思考回合需要保活。先设头再启心跳（同 responses 路径）。
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	k := newSSEKeepalive(w, sseKeepaliveInterval)
	defer k.Close()
	var fl http.Flusher = k
	sink := newMessagesSink(k, fl, r.meta)
	return upstream.PumpFrames(stats, sink, streamOpts(repair, onErrorFrame)...)
}

// renderNonStream 把聚合后的 OpenAI chat.completion 转换为 Anthropic 消息写出。
// 工具参数非法 JSON → 502 api_error（不伪造 input），返回 false 让调用方知道本跳
// 未成功产出（不得记账号成功）。
func (r messagesRenderer) renderNonStream(w http.ResponseWriter, resp map[string]any) bool {
	obj, err := messagesFromAggregate(resp, r.meta)
	if err != nil {
		writeMessagesError(w, http.StatusBadGateway, "api_error", err.Error())
		return false
	}
	writeJSON(w, http.StatusOK, obj)
	return true
}

// renderError 写出终态错误（Anthropic 信封）。Anthropic 无 gateway_hint 字段，hint
// 非空时折进 message（不新增非标准字段）。
func (r messagesRenderer) renderError(w http.ResponseWriter, status int, code, msg, hint string) {
	writeMessagesError(w, status, mapMessagesErrorType(status), messagesErrorMessage(msg, hint))
}

// ---------------------------------------------------------------------------
// FrameSink 实现
// ---------------------------------------------------------------------------

// messagesToolCall 一个 tool_use 内容块的跨帧状态。
//
// Anthropic 内容块必须**顺序** start/delta/stop，而上游可能把多个并行 tool_call 的
// arguments 分片**交错**下发（index 0/1 交替）。若在交错时分片只写给「当前打开的块」，
// 另一个 index 的后续分片会被静默丢弃、拼出截断 JSON。因此这里按 index 缓冲全部分片，
// 只在「切换到正文/思维链」或「流结束」时按 index 顺序一次性完整发出（见 flushTools）。
type messagesToolCall struct {
	openAIIndex int
	blockIndex  int
	id          string
	name        string
	argFrags    []string // 该 index 的全部 arguments 分片（保序）
	emitted     bool     // 是否已发出 content_block_start/delta/stop
}

// messagesSink 是 FrameSink 的 Anthropic 实现：拥有事件状态机（message_start 惰性
// 一次、内容块生命周期、单调 index、per-tool 状态、usage 捕获、终态收尾）。
//
// web_search 服务端工具：meta.webSearch 非 nil 时，message_start 之后先补发
// server_tool_use（携带 query）+ web_search_tool_result 两个内容块，再进入常规
// 正文/工具块——与 Anthropic 服务端检索的事件序列一致（客户端只渲染，不执行）。
//
// 已知限制（诚实记录，不伪造）：
//   - message_start 的 usage.input_tokens 取「发出时已观测到的 prompt_tokens」。上游
//     通常在**末帧**才给 usage，因此流式下 input_tokens 多为 0；output_tokens 以
//     message_delta 的末帧 usage 为准。
//   - tool_use 块整块发出（按 index 顺序），不做逐分片流式：Anthropic 内容块必须严格
//     顺序闭合，而上游可交错下发并行 tool_call 分片，整块发出是保持协议正确性的前提。
type messagesSink struct {
	w    io.Writer
	fl   http.Flusher
	meta messagesMeta

	msgID   string
	started bool

	nextIndex int
	open      bool   // 是否有内容块处于打开状态（仅 text/thinking；tool 块为原子发出）
	openKind  string // "text" | "thinking"

	textIndex     int // 已打开的 text 块 index（-1 = 无）
	thinkingIndex int // 已打开的 thinking 块 index（-1 = 无）

	toolCalls map[int]*messagesToolCall
	toolOrder []int
	// droppedSearchCalls 模型主动调用模式下被拦截的 web_search 调用（按上游
	// tool_calls index）：合成工具只由网关执行，不得向客户端透出 tool_use 块
	// ——Claude Code 无法执行该名字的工具（与 responses sink 同语义）。
	droppedSearchCalls map[int]bool

	usage  map[string]any
	finish string

	// sawError 记录本流是否收到过上游 error 帧（含空流兜底帧）。置位后 Done 不再发出
	// 成功终态（message_delta + message_stop）——error 事件本身就是流的终止，与
	// Responses 的 sawError → response.failed 同语义（失败不得被读成正常收尾）。
	sawError     bool
	errorCode    string
	errorMessage string
}

func newMessagesSink(w io.Writer, fl http.Flusher, meta messagesMeta) *messagesSink {
	return &messagesSink{
		w:             w,
		fl:            fl,
		meta:          meta,
		msgID:         "msg_" + randHex(16),
		textIndex:     -1,
		thinkingIndex: -1,
		toolCalls:     map[int]*messagesToolCall{},
	}
}

// emitSearchPrelude 在 message_start 之后补发 web_search 服务端工具的事件序列：
// server_tool_use（input_json_delta 携带 {"query":...}）→ web_search_tool_result。
// 索引从当前 nextIndex 起单调推进；结束后 open=false，正文块自然从下一个 index 开始。
func (s *messagesSink) emitSearchPrelude() error {
	ws := s.meta.webSearch
	if ws == nil {
		return nil
	}
	// 模型主动调用模式：按已执行检索逐对输出（execs 由 messages_searchloop 回填）。
	if len(ws.execs) > 0 {
		for _, e := range ws.execs {
			if err := s.emitSearchBlockPair(e.query, webSearchResultContentOf(e.results, e.err)); err != nil {
				return err
			}
		}
		return nil
	}
	return s.emitSearchBlockPair(ws.query, webSearchResultContentOf(ws.results, ws.err))
}

// emitSearchBlockPair 发出一对 server_tool_use + web_search_tool_result 内容块。
func (s *messagesSink) emitSearchBlockPair(query string, resultContent any) error {
	id := "srvtoolu_" + randHex(16)

	idx := s.nextIndex
	s.nextIndex++
	if err := s.emit("content_block_start", map[string]any{
		"index": idx,
		"content_block": map[string]any{
			"type": "server_tool_use", "id": id, "name": "web_search", "input": map[string]any{},
		},
	}); err != nil {
		return err
	}
	partial, _ := json.Marshal(map[string]any{"query": query})
	if err := s.emit("content_block_delta", map[string]any{
		"index": idx,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": string(partial)},
	}); err != nil {
		return err
	}
	if err := s.emit("content_block_stop", map[string]any{"index": idx}); err != nil {
		return err
	}

	ridx := s.nextIndex
	s.nextIndex++
	if err := s.emit("content_block_start", map[string]any{
		"index": ridx,
		"content_block": map[string]any{
			"type": "web_search_tool_result", "tool_use_id": id,
			"content": resultContent,
		},
	}); err != nil {
		return err
	}
	return s.emit("content_block_stop", map[string]any{"index": ridx})
}

// webSearchResultContent 构造 web_search_tool_result 的 content：
// 检索失败 → web_search_tool_result_error；检索成功但 0 结果 → **空结果数组**
// （Anthropic 语义里「无结果」不是错误，错误码枚举也不含 no_results）；有结果 →
// 结果数组。
func webSearchResultContent(ws *messagesWebSearch) any {
	return webSearchResultContentOf(ws.results, ws.err)
}

func webSearchResultContentOf(results []websearch.Result, errText string) any {
	if errText != "" {
		return map[string]any{"type": "web_search_tool_result_error", "error_code": "unavailable"}
	}
	arr := make([]any, 0, len(results))
	for _, r := range results {
		item := map[string]any{
			"type":              "web_search_result",
			"url":               r.URL,
			"title":             r.Title,
			"encrypted_content": "",
		}
		if r.PageAge != "" {
			item["page_age"] = r.PageAge
		} else {
			item["page_age"] = nil
		}
		arr = append(arr, item)
	}
	return arr
}

// emit 写一条 SSE 事件：event: <name>\ndata: <json>\n\n（data 内附 type），写完 flush。
func (s *messagesSink) emit(event string, data map[string]any) error {
	data["type"] = event
	raw, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(s.w, "event: "+event+"\ndata: "+string(raw)+"\n\n"); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}

// ensureStarted 惰性发出 message_start（仅一次）。input_tokens 用当前已观测到的
// usage 拆分（未见 usage 时三件套均为 0，不伪造）。
func (s *messagesSink) ensureStarted() error {
	if s.started {
		return nil
	}
	s.started = true
	usage := messagesUsageFields(s.usage)
	usage["output_tokens"] = 0 // message_start 时上游尚未产出正文
	if err := s.emit("message_start", map[string]any{
		"message": map[string]any{
			"id":            s.msgID,
			"type":          "message",
			"role":          "assistant",
			"model":         s.meta.model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         usage,
		},
	}); err != nil {
		return err
	}
	return s.emitSearchPrelude()
}

// Chunk 处理一帧归一化后的 OpenAI chunk：捕获 usage、处理 content / tool_calls /
// finish_reason。无可用 choices 的帧只捕获 usage 后返回。
func (s *messagesSink) Chunk(frame map[string]any, _ string) error {
	if u, ok := frame["usage"].(map[string]any); ok && u != nil {
		// 按 key 合并而非整表替换：上游可能在**不同帧**分别下发 prompt_tokens 与
		// completion_tokens（末帧常只带 completion_tokens）。整表替换会丢掉前帧已知的
		// prompt_tokens，使 message_delta.input_tokens 退回 0，客户端上下文占用被清零。
		if s.usage == nil {
			s.usage = map[string]any{}
		}
		for k, v := range u {
			s.usage[k] = v
		}
	}
	if err := s.ensureStarted(); err != nil {
		return err
	}
	// error 帧即流终止：此后到达的任何 data 帧一律丢弃，绝不在 event: error 之后再发
	// content 块（Anthropic 语义里 error 之后不应有任何事件；且 Done 在 sawError 时早退，
	// 若此时仍开块会留下永不闭合的悬空块）。
	if s.sawError {
		return nil
	}
	choices, _ := frame["choices"].([]any)
	if len(choices) == 0 {
		return nil
	}
	c, _ := choices[0].(map[string]any)
	if c == nil {
		return nil
	}
	if fr, ok := c["finish_reason"].(string); ok && fr != "" {
		s.finish = fr
	}
	delta, _ := c["delta"].(map[string]any)
	if delta == nil {
		return nil
	}
	// 上游推理模型的思维链：Anthropic 有对应的 thinking 内容块，映射过去，
	// 否则 Claude Code / ZCode 这类客户端看不到思考过程（历史误判为「Anthropic
	// 无 reasoning 块形态」而丢弃）。thinking 块必须排在 text 之前，故在正文
	// 打开前先消费。
	if rc, ok := delta["reasoning_content"].(string); ok && rc != "" {
		if err := s.appendThinking(rc); err != nil {
			return err
		}
	}
	if txt, ok := delta["content"].(string); ok && txt != "" {
		if err := s.appendText(txt); err != nil {
			return err
		}
	}
	if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
		for _, tci := range tcs {
			tc, _ := tci.(map[string]any)
			if tc == nil {
				continue
			}
			if err := s.handleToolCall(tc); err != nil {
				return err
			}
		}
	}
	// delta.reasoning_content 已在上方映射为 thinking 块。
	return nil
}

// appendThinking 累积思维链并保证 thinking 块处于打开状态，随后发出 thinking_delta。
// thinking 块必须排在正文之前，正文块一旦打开就不再接受 thinking（上游实测顺序如此）。
func (s *messagesSink) appendThinking(txt string) error {
	opened, err := s.openThinkingBlock()
	if err != nil {
		return err
	}
	if !opened {
		// 已进入正文/tool 阶段：思维链阶段结束，忽略后续 thinking，绝不发 index:-1
		// 的 delta（Anthropic SDK 按 index 取值，-1 会污染最后一个内容块）。
		return nil
	}
	return s.emit("content_block_delta", map[string]any{
		"index": s.thinkingIndex,
		"delta": map[string]any{"type": "thinking_delta", "thinking": txt},
	})
}

// openThinkingBlock 保证当前打开的是 thinking 块。返回 opened=true 表示 thinking 块
// 当前已打开（可直接发 delta）；false 表示思维链阶段已结束（正文/tool 已开始），调用方
// 必须跳过本次 thinking 内容，不得发 delta。
func (s *messagesSink) openThinkingBlock() (bool, error) {
	if s.open {
		// 已打开某块：是 thinking 则可继续，否则说明思维链阶段已结束。
		return s.openKind == "thinking", nil
	}
	// 开新块前先冲掉缓冲的 tool 块，保持到达顺序。
	if err := s.flushTools(); err != nil {
		return false, err
	}
	s.thinkingIndex = s.nextIndex
	s.nextIndex++
	s.open = true
	s.openKind = "thinking"
	if err := s.emit("content_block_start", map[string]any{
		"index":         s.thinkingIndex,
		"content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""},
	}); err != nil {
		return false, err
	}
	return true, nil
}

// appendText 累积正文并保证 text 块处于打开状态，随后发出 text_delta。
func (s *messagesSink) appendText(txt string) error {
	if err := s.openTextBlock(); err != nil {
		return err
	}
	return s.emit("content_block_delta", map[string]any{
		"index": s.textIndex,
		"delta": map[string]any{"type": "text_delta", "text": txt},
	})
}

// openTextBlock 保证当前打开的是 text 块；若当前打开的是别的块（thinking），先 close
// 再新开 text 块（单调 index）。已在 text 块上时空操作。
func (s *messagesSink) openTextBlock() error {
	if s.open && s.openKind == "text" {
		return nil
	}
	// 顺序要紧：先关闭当前打开块（thinking），再冲掉缓冲的 tool 块，最后开 text。
	// 反序会在 thinking 未闭合时插入 tool 块，违反 Anthropic 顺序闭合约束。
	if err := s.closeBlock(); err != nil {
		return err
	}
	if err := s.flushTools(); err != nil {
		return err
	}
	s.textIndex = s.nextIndex
	s.nextIndex++
	s.open = true
	s.openKind = "text"
	return s.emit("content_block_start", map[string]any{
		"index":         s.textIndex,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
}

// handleToolCall 缓冲一个 tool_call 分片。**不立即发出**任何事件：Anthropic 内容块
// 必须严格顺序 start/delta/stop，而上游会把多个并行 tool_call 的 arguments 分片交错
// 下发（index 0/1 交替）。立即发出会导致「只写给当前打开的块」，另一 index 的后续分片
// 被静默丢弃、拼出截断 JSON。这里按 index 缓冲全部分片，由 flushTools 在「切到正文/
// 思维链」或「流结束」时按 index 顺序整块发出。
func (s *messagesSink) handleToolCall(tc map[string]any) error {
	idx := 0
	if v, ok := tc["index"].(float64); ok {
		idx = int(v)
	}
	if s.droppedSearchCalls[idx] {
		return nil
	}
	if fn, ok := tc["function"].(map[string]any); ok {
		if name, _ := fn["name"].(string); s.meta.webSearch != nil && name == searchFunctionName {
			if s.droppedSearchCalls == nil {
				s.droppedSearchCalls = map[int]bool{}
			}
			s.droppedSearchCalls[idx] = true
			log.Printf("WARN: [server] messages: dropped leaked web_search call (index %d) from client stream", idx)
			return nil
		}
	}
	st := s.toolCalls[idx]
	if st == nil {
		st = &messagesToolCall{openAIIndex: idx}
		s.toolCalls[idx] = st
		s.toolOrder = append(s.toolOrder, idx)
	}
	if id, ok := tc["id"].(string); ok && id != "" && st.id == "" {
		st.id = id
	}
	if fn, ok := tc["function"].(map[string]any); ok {
		if name, ok := fn["name"].(string); ok && name != "" && st.name == "" {
			st.name = name
		}
		if args, ok := fn["arguments"].(string); ok && args != "" {
			st.argFrags = append(st.argFrags, args)
		}
	}
	return nil
}

// flushTools 按 index 到达顺序把已缓冲的 tool_use 块整块发出：每个块
// start → N 个 input_json_delta（保序分片，保留流式粒度）→ stop，再推进下一个。
// 已发出（emitted）的块跳过，支持「工具 → 正文 → 工具」交错场景。
func (s *messagesSink) flushTools() error {
	for _, idx := range s.toolOrder {
		st := s.toolCalls[idx]
		if st == nil || st.emitted {
			continue
		}
		st.blockIndex = s.nextIndex
		s.nextIndex++
		st.emitted = true
		if err := s.emit("content_block_start", map[string]any{
			"index": st.blockIndex,
			"content_block": map[string]any{
				"type": "tool_use", "id": st.id, "name": st.name, "input": map[string]any{},
			},
		}); err != nil {
			return err
		}
		for _, frag := range st.argFrags {
			if err := s.emit("content_block_delta", map[string]any{
				"index": st.blockIndex,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": frag},
			}); err != nil {
				return err
			}
		}
		if err := s.emit("content_block_stop", map[string]any{"index": st.blockIndex}); err != nil {
			return err
		}
	}
	return nil
}

// closeBlock 关闭当前打开的内容块（无打开块时空操作）。仅 text/thinking 需要显式
// close——tool 块在 flushTools 内自成 start/delta/stop，不经过这里。
func (s *messagesSink) closeBlock() error {
	if !s.open {
		return nil
	}
	idx := s.textIndex
	if s.openKind == "thinking" {
		idx = s.thinkingIndex
	}
	if err := s.emit("content_block_stop", map[string]any{"index": idx}); err != nil {
		return err
	}
	s.open = false
	s.openKind = ""
	return nil
}

// Raw 非 JSON 帧无 Anthropic 等价物 → no-op（不报错）。
func (s *messagesSink) Raw(_ string) error { return nil }

// Comment 注释行无 Anthropic 等价物 → no-op。
func (s *messagesSink) Comment(_ string) error { return nil }

// ErrorFrame 发出 event: error。type 固定 "api_error"（Anthropic 流内错误的通用类型），
// message 从上游 error 信封派生（解析失败用原文）。
func (s *messagesSink) ErrorFrame(payload string) error {
	if err := s.ensureStarted(); err != nil {
		return err
	}
	code, message := responsesErrorFromFrame(payload)
	s.sawError = true
	s.errorCode = code
	s.errorMessage = message
	return s.emit("error", map[string]any{
		"error": map[string]any{"type": "api_error", "message": message},
	})
}

// Done 收尾：
//   - 收到过上游 error 帧（含空流兜底帧）→ 不发任何成功终态。error 事件已是流的终止
//     （Anthropic 语义），再补 message_delta(结束)+message_stop 会把失败读成正常收尾
//     （与 Responses 的 sawError → response.failed 对齐）。
//   - 否则关闭当前打开块、冲掉缓冲的 tool 块（空流则补一个空 text 块），随后发出
//     message_delta + message_stop，产出完整成功生命周期。
func (s *messagesSink) Done() error {
	if s.sawError {
		return nil
	}
	if err := s.ensureStarted(); err != nil {
		return err
	}
	if err := s.closeBlock(); err != nil {
		return err
	}
	// 冲掉流尾仍缓冲的 tool 块（tools 只在切块/收尾时整块发出，见 flushTools）。
	if err := s.flushTools(); err != nil {
		return err
	}
	if s.nextIndex == 0 {
		// 空流（未打开任何块）：补一个空 text 块，保证 start/stop 成对。
		if err := s.openTextBlock(); err != nil {
			return err
		}
		if err := s.closeBlock(); err != nil {
			return err
		}
	}
	output := int64(numberField(s.usage, "completion_tokens"))
	usage := messagesUsageFields(s.usage)
	usage["output_tokens"] = output
	if err := s.emit("message_delta", map[string]any{
		"delta": map[string]any{"stop_reason": s.stopReason(), "stop_sequence": nil},
		"usage": usage,
	}); err != nil {
		return err
	}
	return s.emit("message_stop", map[string]any{})
}

// stopReason 把上游 finish_reason 映射为 Anthropic stop_reason；未知/空 → end_turn。
func (s *messagesSink) stopReason() string {
	return messagesStopReason(s.finish)
}

// ---------------------------------------------------------------------------
// 非流式转换
// ---------------------------------------------------------------------------

// messagesFromAggregate 把聚合后的 OpenAI chat.completion 转为 Anthropic 消息对象。
// tool_call.arguments 非法 JSON → 返回错误（不伪造 input）。
func messagesFromAggregate(resp map[string]any, meta messagesMeta) (map[string]any, error) {
	content := make([]any, 0, 3)
	// web_search 服务端工具：结果块前置在正文之前（与 Anthropic 非流式消息一致）。
	// 模型主动调用模式按 execs 逐对前插；旧预检路径退回单对。
	if ws := meta.webSearch; ws != nil {
		pairs := ws.execs
		if len(pairs) == 0 {
			pairs = []searchExecRecord{{query: ws.query, results: ws.results, err: ws.err}}
		}
		for _, e := range pairs {
			id := "srvtoolu_" + randHex(16)
			content = append(content,
				map[string]any{
					"type": "server_tool_use", "id": id, "name": "web_search",
					"input": map[string]any{"query": e.query},
				},
				map[string]any{
					"type": "web_search_tool_result", "tool_use_id": id,
					"content": webSearchResultContentOf(e.results, e.err),
				},
			)
		}
	}
	finish := ""
	if choices, ok := resp["choices"].([]any); ok && len(choices) > 0 {
		if cm, ok := choices[0].(map[string]any); ok {
			if fr, ok := cm["finish_reason"].(string); ok {
				finish = fr
			}
		}
	}
	if message, ok := firstMessage(resp); ok {
		// 思维链优先于正文（Anthropic 内容块顺序：thinking → text → tool_use）。
		if rc, ok := message["reasoning_content"].(string); ok && rc != "" {
			content = append(content, map[string]any{
				"type": "thinking", "thinking": rc, "signature": "",
			})
		}
		if s, ok := message["content"].(string); ok && s != "" {
			content = append(content, map[string]any{"type": "text", "text": s})
		}
		for _, tc := range toolCallsOf(message) {
			callID, _ := tc["id"].(string)
			name, args := "", ""
			if fn, ok := tc["function"].(map[string]any); ok {
				name, _ = fn["name"].(string)
				args, _ = fn["arguments"].(string)
			}
			if meta.webSearch != nil && name == searchFunctionName {
				continue
			}
			input, err := parseToolArguments(args)
			if err != nil {
				return nil, fmt.Errorf("upstream returned invalid tool arguments for %q: %v", name, err)
			}
			content = append(content, map[string]any{
				"type": "tool_use", "id": callID, "name": name, "input": input,
			})
		}
	}

	obj := map[string]any{
		"id":            "msg_" + randHex(16),
		"type":          "message",
		"role":          "assistant",
		"model":         meta.model,
		"content":       content,
		"stop_reason":   messagesStopReason(finish),
		"stop_sequence": nil,
		"usage":         messagesUsage(resp),
	}
	return obj, nil
}

// toolCallsOf 取出聚合 message 的 tool_calls。upstream.Aggregate 在内存里存的是
// []map[string]any（见 upstream/sse.go），而 JSON 解码路径是 []any——两种都接受，
// 避免非流式工具调用被静默丢成空 content。
func toolCallsOf(message map[string]any) []map[string]any {
	switch tcs := message["tool_calls"].(type) {
	case []any:
		out := make([]map[string]any, 0, len(tcs))
		for _, tci := range tcs {
			if tc, ok := tci.(map[string]any); ok {
				out = append(out, tc)
			}
		}
		return out
	case []map[string]any:
		return tcs
	default:
		return nil
	}
}

// parseToolArguments 解析 tool_call.arguments。空/空白 → 空对象 {}（无参调用的诚实
// 表示，不伪造数据）；非空但非法 JSON → 错误（调用方回 502）。
func parseToolArguments(args string) (map[string]any, error) {
	if strings.TrimSpace(args) == "" {
		return map[string]any{}, nil
	}
	var input map[string]any
	if err := json.Unmarshal([]byte(args), &input); err != nil {
		return nil, err
	}
	if input == nil {
		input = map[string]any{}
	}
	return input, nil
}

// ---------------------------------------------------------------------------
// usage / stop_reason / 错误映射
// ---------------------------------------------------------------------------

// messagesStopReason 映射 OpenAI finish_reason → Anthropic stop_reason。
// stop_sequence 恒为 null：上游是 OpenAI 形态，无法识别命中了哪条 stop 序列。
func messagesStopReason(finish string) string {
	switch finish {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}

// messagesUsage 把 OpenAI usage 映射为 Anthropic usage。
//
// Anthropic 语义下 input_tokens 是**未命中缓存**的输入（客户端上下文占用 =
// input_tokens + cache_read_input_tokens + cache_creation_input_tokens），而上游
// prompt_tokens 是**含缓存**的总量。因此必须拆分：input_tokens = prompt_tokens
// − 命中 − 写入，否则客户端把整段 prompt 当新输入、缓存维度又缺失，占用率与命中率
// 同时失真。上游未给缓存维度时 cache 字段省略（不伪造 0）。
func messagesUsage(resp map[string]any) map[string]any {
	usage, _ := resp["usage"].(map[string]any)
	return messagesUsageFields(usage)
}

// messagesUsageFields 从上游 usage 拆出 Anthropic 三件套。流式/非流式共用同一口径，
// 避免两条路径的 input_tokens 定义漂移。
func messagesUsageFields(usage map[string]any) map[string]any {
	prompt := int64(numberField(usage, "prompt_tokens"))
	read, hasRead := upstream.UsageCacheHitTokens(usage)
	write, hasWrite := upstream.UsageCacheWriteTokens(usage)

	out := map[string]any{"output_tokens": int64(numberField(usage, "completion_tokens"))}
	noCache := prompt - int64(read) - int64(write)
	if noCache < 0 {
		// 上游字段不自洽（命中+写入 > 总量）：退回整段 prompt，不产出负数。
		noCache = prompt
	}
	out["input_tokens"] = noCache
	if hasRead {
		out["cache_read_input_tokens"] = int64(read)
	}
	if hasWrite {
		out["cache_creation_input_tokens"] = int64(write)
	}
	return out
}

// mapMessagesErrorType 按 HTTP 状态映射 Anthropic error.type。
func mapMessagesErrorType(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusInternalServerError:
		return "api_error"
	case 529:
		return "overloaded_error"
	}
	switch {
	case status >= 500:
		return "api_error"
	case status >= 400:
		return "invalid_request_error"
	default:
		return "api_error"
	}
}

// messagesErrorMessage 把 gateway_hint 折进 message（Anthropic 无 gateway_hint 字段）。
func messagesErrorMessage(msg, hint string) string {
	if hint == "" {
		return msg
	}
	return msg + " (gateway hint: " + hint + ")"
}

// writeMessagesError 写 Anthropic 错误信封：
// {"type":"error","error":{"type":"<...>","message":"..."}}。
func writeMessagesError(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    typ,
			"message": msg,
		},
	})
}
