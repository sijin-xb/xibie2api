// responses_render.go 实现 OpenAI Responses API（/v1/responses）的响应渲染：
//   - responsesRenderer：turnRenderer 的 Responses 实现（流式 / 非流式 / 终态错误）；
//   - responsesSink：upstream.FrameSink 的 Responses 实现，把上游 OpenAI Chat
//     Completions SSE 帧翻译成 Responses SSE 事件序列；
//   - 非流式：把聚合后的 OpenAI chat.completion 转换为 Responses 对象；
//   - writeResponsesError：Responses 错误信封。
//
// 边界纪律：本层只做「协议表示翻译」，不持有账号选择/轮转/惩罚（runTurn 拥有），
// 也不做 usage 记账（runTurn 从 chatStatsReader 读）。
package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// randHex 返回 n 字节随机数的十六进制串（用于 resp_/msg_/fc_ 前缀 id）。
func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		// 极不可能：退化为时间基准（仍保证非空、近唯一）。
		return fmt.Sprintf("%x", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// ---------------------------------------------------------------------------
// turnRenderer 实现
// ---------------------------------------------------------------------------

type responsesRenderer struct {
	meta responsesMeta
}

// renderStream 把已归一化的上游 SSE 流翻译为 Responses SSE 事件序列。
// hint 被忽略：Responses 无 gateway_hint 帧形态（错误以 event: error 表达）。
func (r responsesRenderer) renderStream(w http.ResponseWriter, stats io.Reader,
	_ func() upstream.HintContext, repair *upstream.MarkupRepair, onErrorFrame func(string)) error {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	fl, _ := w.(http.Flusher)
	sink := newResponsesSink(w, fl, r.meta)
	return upstream.PumpFrames(stats, sink,
		upstream.WithMarkupRepair(repair),
		upstream.WithErrorFrameObserver(onErrorFrame))
}

// renderNonStream 把聚合后的 OpenAI chat.completion 转换为 Responses 对象写出。
func (r responsesRenderer) renderNonStream(w http.ResponseWriter, resp map[string]any) {
	writeJSON(w, http.StatusOK, responsesFromAggregate(resp, r.meta))
}

// renderError 写出终态错误（Responses 信封；hint 非空时附 gateway_hint）。
func (r responsesRenderer) renderError(w http.ResponseWriter, status int, code, msg, hint string) {
	writeResponsesError(w, status, mapTypeFromStatus(status), code, msg, "", hint)
}

// ---------------------------------------------------------------------------
// FrameSink 实现
// ---------------------------------------------------------------------------

// responsesToolCall 一个 function_call 输出项的跨帧状态。
type responsesToolCall struct {
	index    int
	outIndex int
	itemID   string
	callID   string
	name     string
	args     strings.Builder
	opened   bool
	closed   bool
}

// responsesSink 是 FrameSink 的 Responses 实现：拥有事件状态机（sequence_number、
// message item 生命周期、per-tool-call 状态、output_index 分配、usage 捕获）。
type responsesSink struct {
	w    io.Writer
	fl   http.Flusher
	meta responsesMeta

	respID    string
	createdAt int64
	seq       int
	started   bool

	msgID     string
	msgIndex  int
	msgOpened bool
	msgClosed bool
	msgText   strings.Builder

	toolCalls map[int]*responsesToolCall
	toolOrder []int

	nextIndex int
	usage     map[string]any
	finish    string

	// 上游 error 帧状态：一旦收到，终态必须是 response.failed（绝不 response.completed），
	// 否则失败请求会被客户端读成成功。
	sawError     bool
	errorCode    string
	errorMessage string
}

func newResponsesSink(w io.Writer, fl http.Flusher, meta responsesMeta) *responsesSink {
	return &responsesSink{
		w:         w,
		fl:        fl,
		meta:      meta,
		respID:    "resp_" + randHex(16),
		createdAt: time.Now().Unix(),
		msgID:     "msg_" + randHex(16),
		toolCalls: map[int]*responsesToolCall{},
	}
}

// emit 写一条 SSE 事件：event: <name>\ndata: <json>\n\n，data 内附 type + 单调
// sequence_number（从 1 起），写完 flush。
func (s *responsesSink) emit(event string, data map[string]any) error {
	s.seq++
	data["type"] = event
	data["sequence_number"] = s.seq
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

// ensureStarted 惰性发出 response.created + response.in_progress（仅一次）。
func (s *responsesSink) ensureStarted() error {
	if s.started {
		return nil
	}
	s.started = true
	if err := s.emit("response.created", map[string]any{"response": s.responseObject("in_progress", false)}); err != nil {
		return err
	}
	return s.emit("response.in_progress", map[string]any{"response": s.responseObject("in_progress", false)})
}

// responseObject 构造 Responses 响应对象；completed=true 时填 completed_at。
func (s *responsesSink) responseObject(status string, completed bool) map[string]any {
	obj := map[string]any{
		"id":                   s.respID,
		"object":               "response",
		"created_at":           s.createdAt,
		"status":               status,
		"completed_at":         nil,
		"model":                s.meta.model,
		"output":               []any{},
		"parallel_tool_calls":  s.meta.parallelToolCallsValue(),
		"tool_choice":          s.meta.toolChoiceValue(),
		"tools":                s.meta.toolsValue(),
		"temperature":          s.meta.temperature,
		"top_p":                s.meta.topP,
		"max_output_tokens":    s.meta.maxOutputTokens,
		"previous_response_id": nil,
		"instructions":         s.meta.instructions,
		"metadata":             s.meta.metadata,
		"error":                nil,
		"incomplete_details":   nil,
		"usage":                nil,
	}
	if completed {
		obj["completed_at"] = time.Now().Unix()
	}
	return obj
}

func (m responsesMeta) toolsValue() any {
	if m.tools == nil {
		return []any{}
	}
	return m.tools
}

func (m responsesMeta) toolChoiceValue() any {
	if m.toolChoice == nil {
		return "auto"
	}
	return m.toolChoice
}

func (m responsesMeta) parallelToolCallsValue() any {
	if m.parallelToolCalls == nil {
		return true
	}
	return m.parallelToolCalls
}

// Chunk 处理一帧归一化后的 OpenAI chunk：捕获 usage、处理 content / tool_calls /
// finish_reason。无可用 choices 的帧只捕获 usage 后返回。
func (s *responsesSink) Chunk(frame map[string]any, _ string) error {
	if u, ok := frame["usage"].(map[string]any); ok && u != nil {
		s.usage = u
	}
	// 首个上游帧即发出 created/in_progress（即使本帧无可渲染 delta），保证空流也有
	// 完整起始事件。
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
	// delta.reasoning_content 显式忽略：Responses 的 reasoning 输出需要独立的
	// reasoning item（含 encrypted_content/摘要），本网关无该数据，不臆造。
	return nil
}

// appendText 累积正文并在需要时打开 message item，随后发出 output_text.delta。
func (s *responsesSink) appendText(txt string) error {
	if !s.msgOpened {
		if err := s.openMessage(); err != nil {
			return err
		}
	}
	s.msgText.WriteString(txt)
	return s.emit("response.output_text.delta", map[string]any{
		"item_id":       s.msgID,
		"output_index":  s.msgIndex,
		"content_index": 0,
		"delta":         txt,
		"logprobs":      []any{},
	})
}

// openMessage 打开 assistant message item（added + content_part.added）。
func (s *responsesSink) openMessage() error {
	s.msgIndex = s.nextOutputIndex()
	s.msgOpened = true
	if err := s.emit("response.output_item.added", map[string]any{
		"output_index": s.msgIndex,
		"item": map[string]any{
			"id":      s.msgID,
			"type":    "message",
			"status":  "in_progress",
			"role":    "assistant",
			"content": []any{},
		},
	}); err != nil {
		return err
	}
	return s.emit("response.content_part.added", map[string]any{
		"item_id":       s.msgID,
		"output_index":  s.msgIndex,
		"content_index": 0,
		"part": map[string]any{
			"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{},
		},
	})
}

// handleToolCall 处理一个 tool_call 分片：首现打开 function_call item，随后发出
// arguments.delta。function_call item 没有 content parts。
func (s *responsesSink) handleToolCall(tc map[string]any) error {
	idx := 0
	if v, ok := tc["index"].(float64); ok {
		idx = int(v)
	}
	st := s.toolCalls[idx]
	if st == nil {
		st = &responsesToolCall{index: idx, outIndex: s.nextOutputIndex(), itemID: "fc_" + randHex(16)}
		s.toolCalls[idx] = st
		s.toolOrder = append(s.toolOrder, idx)
	}
	if id, ok := tc["id"].(string); ok && id != "" && st.callID == "" {
		st.callID = id
	}
	if fn, ok := tc["function"].(map[string]any); ok {
		if name, ok := fn["name"].(string); ok && name != "" && st.name == "" {
			st.name = name
		}
	}
	if !st.opened {
		st.opened = true
		if err := s.emit("response.output_item.added", map[string]any{
			"output_index": st.outIndex,
			"item": map[string]any{
				"type": "function_call", "id": st.itemID, "call_id": st.callID,
				"name": st.name, "arguments": "", "status": "in_progress",
			},
		}); err != nil {
			return err
		}
	}
	if fn, ok := tc["function"].(map[string]any); ok {
		if args, ok := fn["arguments"].(string); ok && args != "" {
			st.args.WriteString(args)
			if err := s.emit("response.function_call_arguments.delta", map[string]any{
				"item_id": st.itemID, "output_index": st.outIndex, "delta": args,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

// Raw 非 JSON 帧无 Responses 等价物 → no-op（不报错）。
func (s *responsesSink) Raw(_ string) error { return nil }

// Comment 注释行无 Responses 等价物 → no-op。
func (s *responsesSink) Comment(_ string) error { return nil }

// ErrorFrame 发出 event: error（type/sequence_number 由 emit 补），code/message 从
// 上游 error 信封派生（解析失败用原文）。同时**记录错误状态**：终态由 Done 据此
// 选择 response.failed（而非 response.completed），失败请求不再被读成成功。
func (s *responsesSink) ErrorFrame(payload string) error {
	if err := s.ensureStarted(); err != nil {
		return err
	}
	code, message := responsesErrorFromFrame(payload)
	s.sawError = true
	s.errorCode = code
	s.errorMessage = message
	return s.emit("error", map[string]any{
		"code": code, "message": message, "param": nil,
	})
}

// Done 收尾：关闭 message item 与所有 function_call item，随后发出唯一终态事件。
//   - 收到过上游 error 帧（含空流兜底帧）→ response.failed（status:"failed" + error），
//     携带已产生的部分 output；绝不发 response.completed。
//   - 否则 → response.completed（status:"completed"）。
//
// 空流（未打开任何 item）时仍补一个空 message item，保证结构完整的终态。
func (s *responsesSink) Done() error {
	if err := s.ensureStarted(); err != nil {
		return err
	}
	if !s.msgOpened && len(s.toolOrder) == 0 {
		// 空流（未打开任何 item）：仍产出一个空 message item，保证结构完整的终态。
		if err := s.openMessage(); err != nil {
			return err
		}
	}
	if s.msgOpened && !s.msgClosed {
		if err := s.closeMessage(); err != nil {
			return err
		}
	}
	for _, idx := range s.toolOrder {
		st := s.toolCalls[idx]
		if st == nil || st.closed {
			continue
		}
		if err := s.emit("response.function_call_arguments.done", map[string]any{
			"item_id": st.itemID, "output_index": st.outIndex, "arguments": st.args.String(),
		}); err != nil {
			return err
		}
		if err := s.emit("response.output_item.done", map[string]any{
			"output_index": st.outIndex,
			"item":         st.completedItem(),
		}); err != nil {
			return err
		}
		st.closed = true
	}
	if s.sawError {
		resp := s.responseObject("failed", false)
		resp["output"] = s.finalOutput()
		resp["usage"] = responsesUsage(s.usage)
		resp["error"] = map[string]any{"code": s.errorCode, "message": s.errorMessage}
		return s.emit("response.failed", map[string]any{"response": resp})
	}
	resp := s.responseObject("completed", true)
	resp["output"] = s.finalOutput()
	resp["usage"] = responsesUsage(s.usage)
	return s.emit("response.completed", map[string]any{"response": resp})
}

// closeMessage 关闭 message item（output_text.done + content_part.done + output_item.done）。
func (s *responsesSink) closeMessage() error {
	text := s.msgText.String()
	if err := s.emit("response.output_text.done", map[string]any{
		"item_id": s.msgID, "output_index": s.msgIndex, "content_index": 0,
		"text": text, "logprobs": []any{},
	}); err != nil {
		return err
	}
	if err := s.emit("response.content_part.done", map[string]any{
		"item_id": s.msgID, "output_index": s.msgIndex, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": text, "annotations": []any{}, "logprobs": []any{}},
	}); err != nil {
		return err
	}
	if err := s.emit("response.output_item.done", map[string]any{
		"output_index": s.msgIndex,
		"item":         s.completedMessageItem(),
	}); err != nil {
		return err
	}
	s.msgClosed = true
	return nil
}

func (s *responsesSink) completedMessageItem() map[string]any {
	return map[string]any{
		"id": s.msgID, "type": "message", "status": "completed", "role": "assistant",
		"content": []any{map[string]any{
			"type": "output_text", "text": s.msgText.String(), "annotations": []any{}, "logprobs": []any{},
		}},
	}
}

func (t *responsesToolCall) completedItem() map[string]any {
	return map[string]any{
		"type": "function_call", "id": t.itemID, "call_id": t.callID,
		"name": t.name, "arguments": t.args.String(), "status": "completed",
	}
}

// finalOutput 按 output_index 升序汇总所有已完成 item。
func (s *responsesSink) finalOutput() []any {
	type entry struct {
		idx  int
		item any
	}
	entries := make([]entry, 0, 1+len(s.toolOrder))
	if s.msgOpened {
		entries = append(entries, entry{s.msgIndex, s.completedMessageItem()})
	}
	for _, idx := range s.toolOrder {
		st := s.toolCalls[idx]
		if st == nil {
			continue
		}
		entries = append(entries, entry{st.outIndex, st.completedItem()})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].idx < entries[j].idx })
	out := make([]any, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.item)
	}
	return out
}

func (s *responsesSink) nextOutputIndex() int {
	i := s.nextIndex
	s.nextIndex++
	return i
}

// ---------------------------------------------------------------------------
// 非流式转换
// ---------------------------------------------------------------------------

// responsesFromAggregate 把聚合后的 OpenAI chat.completion 转为 Responses 对象。
// 文本 → message/output_text item；tool_calls → function_call items。
func responsesFromAggregate(resp map[string]any, meta responsesMeta) map[string]any {
	createdAt := time.Now().Unix()
	if v := numberField(resp, "created"); v > 0 {
		createdAt = int64(v)
	}
	obj := map[string]any{
		"id":                   "resp_" + randHex(16),
		"object":               "response",
		"created_at":           createdAt,
		"status":               "completed",
		"completed_at":         time.Now().Unix(),
		"model":                meta.model,
		"output":               []any{},
		"parallel_tool_calls":  meta.parallelToolCallsValue(),
		"tool_choice":          meta.toolChoiceValue(),
		"tools":                meta.toolsValue(),
		"temperature":          meta.temperature,
		"top_p":                meta.topP,
		"max_output_tokens":    meta.maxOutputTokens,
		"previous_response_id": nil,
		"instructions":         meta.instructions,
		"metadata":             meta.metadata,
		"error":                nil,
		"incomplete_details":   nil,
		"usage":                nil,
	}

	content := ""
	var toolCalls []map[string]any
	if message, ok := firstMessage(resp); ok {
		if s, ok := message["content"].(string); ok {
			content = s
		}
		// upstream.Aggregate 在内存里把 tool_calls 存为 []map[string]any（见
		// upstream/sse.go），JSON 解码路径才是 []any。toolCallsOf 同时接受两种形态，
		// 否则类型断言失败 → 非流式工具调用被静默丢成空 output。
		toolCalls = toolCallsOf(message)
	}

	output := make([]any, 0, 1+len(toolCalls))
	// 纯工具调用回合（content 为空）不产出空 message item；否则始终产出 message item。
	if content != "" || len(toolCalls) == 0 {
		output = append(output, map[string]any{
			"id": "msg_" + randHex(16), "type": "message", "status": "completed", "role": "assistant",
			"content": []any{map[string]any{
				"type": "output_text", "text": content, "annotations": []any{}, "logprobs": []any{},
			}},
		})
	}
	for _, tc := range toolCalls {
		callID, _ := tc["id"].(string)
		name, args := "", ""
		if fn, ok := tc["function"].(map[string]any); ok {
			name, _ = fn["name"].(string)
			args, _ = fn["arguments"].(string)
		}
		output = append(output, map[string]any{
			"type": "function_call", "id": "fc_" + randHex(16), "call_id": callID,
			"name": name, "arguments": args, "status": "completed",
		})
	}
	obj["output"] = output
	if usage, ok := resp["usage"].(map[string]any); ok {
		obj["usage"] = responsesUsage(usage)
	}
	return obj
}

// firstMessage 取聚合响应 choices[0].message。
func firstMessage(resp map[string]any) (map[string]any, bool) {
	choices, _ := resp["choices"].([]any)
	if len(choices) == 0 {
		return nil, false
	}
	c, _ := choices[0].(map[string]any)
	if c == nil {
		return nil, false
	}
	m, _ := c["message"].(map[string]any)
	return m, m != nil
}

// ---------------------------------------------------------------------------
// usage / 错误映射
// ---------------------------------------------------------------------------

// responsesUsage 把 OpenAI usage 映射为 Responses usage 结构。
func responsesUsage(usage map[string]any) any {
	if usage == nil {
		return nil
	}
	input := int64(numberField(usage, "prompt_tokens"))
	output := int64(numberField(usage, "completion_tokens"))
	total := int64(numberField(usage, "total_tokens"))
	if total == 0 && (input != 0 || output != 0) {
		total = input + output
	}
	cached, _ := upstream.UsageCacheHitTokens(usage)
	return map[string]any{
		"input_tokens":          input,
		"input_tokens_details":  map[string]any{"cached_tokens": int64(cached), "cache_write_tokens": 0},
		"output_tokens":         output,
		"output_tokens_details": map[string]any{"reasoning_tokens": 0},
		"total_tokens":          total,
	}
}

// numberField 从 map 读取数值字段（JSON number → float64）；非数值返回 0。
func numberField(m map[string]any, key string) float64 {
	switch n := m[key].(type) {
	case float64:
		return n
	case int64:
		return float64(n)
	case int:
		return float64(n)
	default:
		return 0
	}
}

// responsesErrorFromFrame 从上游 error 帧 payload 派生 (code, message)；
// 解析失败时 code="upstream_error"、message=原文。
func responsesErrorFromFrame(payload string) (code, message string) {
	var obj struct {
		Error struct {
			Message any    `json:"message"`
			Code    any    `json:"code"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(payload), &obj) == nil && obj.Error.Message != nil {
		message = fmt.Sprint(obj.Error.Message)
		code = "upstream_error"
		if obj.Error.Code != nil {
			code = fmt.Sprint(obj.Error.Code)
		} else if obj.Error.Type != "" {
			code = obj.Error.Type
		}
		return code, message
	}
	return "upstream_error", payload
}

// writeResponsesError 写 Responses 错误信封：
// {"error":{"message","type","param","code"}}（param 空串 → null；hint 非空 → gateway_hint）。
func writeResponsesError(w http.ResponseWriter, status int, typ, code, msg, param, hint string) {
	errObj := map[string]any{
		"message": msg,
		"type":    typ,
		"code":    code,
		"param":   nil,
	}
	if param != "" {
		errObj["param"] = param
	}
	if hint != "" {
		errObj["gateway_hint"] = hint
	}
	writeJSON(w, status, map[string]any{"error": errObj})
}

// mapTypeFromStatus 按 HTTP 状态映射 Responses error.type。
func mapTypeFromStatus(status int) string {
	switch {
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status >= 500:
		return "server_error"
	case status >= 400:
		return "invalid_request_error"
	default:
		return "api_error"
	}
}
