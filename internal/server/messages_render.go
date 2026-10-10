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
	"net/http"
	"strings"

	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
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
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	fl, _ := w.(http.Flusher)
	sink := newMessagesSink(w, fl, r.meta)
	return upstream.PumpFrames(stats, sink,
		upstream.WithMarkupRepair(repair),
		upstream.WithErrorFrameObserver(onErrorFrame))
}

// renderNonStream 把聚合后的 OpenAI chat.completion 转换为 Anthropic 消息写出。
// 工具参数非法 JSON → 502 api_error（不伪造 input）。
func (r messagesRenderer) renderNonStream(w http.ResponseWriter, resp map[string]any) {
	obj, err := messagesFromAggregate(resp, r.meta)
	if err != nil {
		writeMessagesError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, obj)
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
type messagesToolCall struct {
	openAIIndex int
	blockIndex  int
	id          string
	name        string
	args        strings.Builder
	opened      bool
}

// messagesSink 是 FrameSink 的 Anthropic 实现：拥有事件状态机（message_start 惰性
// 一次、内容块生命周期、单调 index、per-tool 状态、usage 捕获、终态收尾）。
//
// 已知限制（诚实记录，不伪造）：
//   - message_start 的 usage.input_tokens 取「发出时已观测到的 prompt_tokens」。上游
//     通常在**末帧**才给 usage，因此流式下 input_tokens 多为 0；output_tokens 以
//     message_delta 的末帧 usage 为准。
//   - 假设同一 tool_call index 的 arguments 分片连续到达（上游实测如此）；跨 index
//     交错分片不支持。
type messagesSink struct {
	w    io.Writer
	fl   http.Flusher
	meta messagesMeta

	msgID   string
	started bool

	nextIndex int
	open      bool   // 是否有内容块处于打开状态
	openKind  string // "text" | "tool"
	openTool  int    // openKind=="tool" 时的 OpenAI tool index

	textIndex int // 已打开的 text 块 index（-1 = 无）

	toolCalls map[int]*messagesToolCall
	toolOrder []int

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
		w:         w,
		fl:        fl,
		meta:      meta,
		msgID:     "msg_" + randHex(16),
		textIndex: -1,
		toolCalls: map[int]*messagesToolCall{},
	}
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
// prompt_tokens（未见 usage 时为 0，不伪造）。
func (s *messagesSink) ensureStarted() error {
	if s.started {
		return nil
	}
	s.started = true
	input := int64(numberField(s.usage, "prompt_tokens"))
	return s.emit("message_start", map[string]any{
		"message": map[string]any{
			"id":            s.msgID,
			"type":          "message",
			"role":          "assistant",
			"model":         s.meta.model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         map[string]any{"input_tokens": input, "output_tokens": 0},
		},
	})
}

// Chunk 处理一帧归一化后的 OpenAI chunk：捕获 usage、处理 content / tool_calls /
// finish_reason。无可用 choices 的帧只捕获 usage 后返回。
func (s *messagesSink) Chunk(frame map[string]any, _ string) error {
	if u, ok := frame["usage"].(map[string]any); ok && u != nil {
		s.usage = u
	}
	if err := s.ensureStarted(); err != nil {
		return err
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
	// delta.reasoning_content 显式忽略：Anthropic 无 reasoning 内容块形态，
	// 本网关无对应数据，不臆造。
	return nil
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

// openTextBlock 保证当前打开的是 text 块；若当前打开的是别的块（tool_use），先 close
// 再新开 text 块（单调 index）。已在 text 块上时空操作。
func (s *messagesSink) openTextBlock() error {
	if s.open && s.openKind == "text" {
		return nil
	}
	if err := s.closeBlock(); err != nil {
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

// handleToolCall 处理一个 tool_call 分片：首现打开 tool_use 内容块，随后发出
// input_json_delta。切换到不同 index 时先关闭当前块（保证单调 index 与顺序闭合）。
func (s *messagesSink) handleToolCall(tc map[string]any) error {
	idx := 0
	if v, ok := tc["index"].(float64); ok {
		idx = int(v)
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
	}
	// 首次见到该 tool_call：关闭当前块，打开新的 tool_use 块。
	if !st.opened {
		if err := s.closeBlock(); err != nil {
			return err
		}
		st.blockIndex = s.nextIndex
		s.nextIndex++
		st.opened = true
		s.open = true
		s.openKind = "tool"
		s.openTool = idx
		if err := s.emit("content_block_start", map[string]any{
			"index": st.blockIndex,
			"content_block": map[string]any{
				"type": "tool_use", "id": st.id, "name": st.name, "input": map[string]any{},
			},
		}); err != nil {
			return err
		}
	}
	if fn, ok := tc["function"].(map[string]any); ok {
		if args, ok := fn["arguments"].(string); ok && args != "" {
			st.args.WriteString(args)
			// 仅当该 tool 块确实处于打开状态才发 delta（绝不向已停止块写 delta）。
			if s.open && s.openKind == "tool" && s.openTool == idx {
				if err := s.emit("content_block_delta", map[string]any{
					"index": st.blockIndex,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
				}); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// closeBlock 关闭当前打开的内容块（无打开块时空操作）。
func (s *messagesSink) closeBlock() error {
	if !s.open {
		return nil
	}
	idx := s.textIndex
	if s.openKind == "tool" {
		if st := s.toolCalls[s.openTool]; st != nil {
			idx = st.blockIndex
		}
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
//   - 否则关闭所有打开的内容块（空流则补一个空 text 块），随后发出 message_delta
//   - message_stop，产出完整成功生命周期。
func (s *messagesSink) Done() error {
	if s.sawError {
		return nil
	}
	if err := s.ensureStarted(); err != nil {
		return err
	}
	if s.nextIndex == 0 {
		// 空流（未打开任何块）：补一个空 text 块，保证 start/stop 成对。
		if err := s.openTextBlock(); err != nil {
			return err
		}
	}
	if err := s.closeBlock(); err != nil {
		return err
	}
	output := int64(numberField(s.usage, "completion_tokens"))
	if err := s.emit("message_delta", map[string]any{
		"delta": map[string]any{"stop_reason": s.stopReason(), "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": output},
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
	content := make([]any, 0, 1)
	finish := ""
	if choices, ok := resp["choices"].([]any); ok && len(choices) > 0 {
		if cm, ok := choices[0].(map[string]any); ok {
			if fr, ok := cm["finish_reason"].(string); ok {
				finish = fr
			}
		}
	}
	if message, ok := firstMessage(resp); ok {
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

// messagesUsage 把 OpenAI usage 映射为 Anthropic usage（只保留 input_tokens /
// output_tokens，不伪造 cache_creation_input_tokens / cache_read_input_tokens）。
func messagesUsage(resp map[string]any) map[string]any {
	usage, _ := resp["usage"].(map[string]any)
	return map[string]any{
		"input_tokens":  int64(numberField(usage, "prompt_tokens")),
		"output_tokens": int64(numberField(usage, "completion_tokens")),
	}
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
