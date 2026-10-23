// responses_request.go 把 OpenAI Responses API（/v1/responses）请求归一化为网关
// 既有的 OpenAI Chat Completions 请求体，使两种协议复用同一条执行管线
// （prepareTurn → runTurn：别名 / realm / 提示词 / 脱敏 / 选号 / 出站传输）。
//
// 设计纪律（支持矩阵，见任务书 Stage 4 §3）：
//   - 归一化只做「请求形态翻译」，**不**做别名 / 提示词 / 脱敏（那些在 prepareTurn 里
//     对归一化后的 OpenAI body 执行）。归一化出的 body["model"] 必须是**原始客户端
//     model 字符串**，这样 peek.Model / 别名 / 出站重写与 chat 路径逐字节一致。
//   - 网关是无状态的：任何需要「服务端存储的历史响应/会话」的**请求字段**一律
//     **400 拒绝**（previous_response_id / conversation / truncation:"auto"），
//     不伪造、不假装支持。
//   - input 里的**历史 item 类型**则是另一回事：历史属于客户端（Codex 会回放压缩
//     项 / web_search_call / 各种新 item），对未知类型逐个 400 会让**整个会话无法
//     继续**——与 /v1/messages 路径 dropContentBlock 同一教训。可翻译的翻译
//     （agent_message / additional_tools / reasoning），无法表达的安全丢弃并记日志。
//   - 已支持字段映射见下方 mapXxx；接受但不转发字段在 meta 里原样带回（回显），
//     绝不落盘。
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// responsesMeta 承载 Responses 响应对象回显所需的原始请求字段（归一化后的
// OpenAI body 之外的信息）。渲染层只用它构造 Responses 对象，不参与执行。
type responsesMeta struct {
	model             string // 原始客户端 model（回显）
	instructions      any    // string 或 nil
	tools             []any  // 原始 tools（nil = 未提供）
	toolChoice        any    // 原始 tool_choice（string / object / nil）
	temperature       any    // 原始数值或 nil
	topP              any
	maxOutputTokens   any
	metadata          any // 接受但不转发
	parallelToolCalls any // 接受但不转发
	reasoning         any // 原始 reasoning 对象（effort 已翻译进 body，此处仅回显）
	stream            bool
	// webSearch 非 nil = Codex 声明了 web_search 服务端工具：出站前由网关检索
	// （见 responses 入口），渲染层据此产出 web_search_call item。
	webSearch *messagesWebSearch
	// toolRefs 摊平后的上游 function 名 → Responses 原始工具形态。渲染层据此把
	// 上游 function_call 还原为 Codex 期望的 item（namespace / custom）。nil = 无需还原。
	toolRefs map[string]responsesToolRef
}

// responsesError 归一化/读体阶段的请求级错误（映射为 Responses 错误信封）。
type responsesError struct {
	status  int
	typ     string
	code    string
	message string
	param   string
}

func invalidResponses(param, message string) *responsesError {
	return &responsesError{
		status:  http.StatusBadRequest,
		typ:     "invalid_request_error",
		code:    "invalid_request",
		message: message,
		param:   param,
	}
}

// responses 是 /v1/responses 入口：读体 → 归一化为 OpenAI body → 共享准备层 →
// runTurn。与 chatCompletions 的唯一差别是「读体后先归一化」这一步。
func (h *Handler) responses(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeResponsesError(w, http.StatusBadRequest, "invalid_request_error", "invalid_request",
			"read body: "+err.Error(), "", "")
		return
	}
	openAIBody, meta, rerr := normalizeResponsesRequest(raw)
	if rerr != nil {
		writeResponsesError(w, rerr.status, rerr.typ, rerr.code, rerr.message, rerr.param, "")
		return
	}
	st, p, derr := h.prepareTurn(r, openAIBody, responsesRenderer{meta: meta})
	if derr != nil {
		writeResponsesError(w, derr.status, "invalid_request_error", derr.code, derr.message, derr.param, "")
		return
	}
	defer st.done()
	// web_search 服务端工具（与 /v1/messages 路径同构）：Codex 声明 type:"web_search"
	// 时由网关出站前实际检索，结果作为一条 system 消息注入 prompt，渲染层先产出
	// 一个 web_search_call item 让 UI 可见。放在 prepareTurn 之后以复用选号/realm。
	if meta.webSearch != nil {
		h.runWebSearch(r.Context(), meta.webSearch, p)
		p.body = injectSearchContext(p.body, meta.webSearch.results)
	}
	h.runTurn(r.Context(), w, st, p)
}

// normalizeResponsesRequest 解析并归一化一个 Responses 请求。
// 返回：OpenAI Chat Completions 请求体（JSON）+ 回显 meta + 请求级错误（nil = OK）。
func normalizeResponsesRequest(raw []byte) ([]byte, responsesMeta, *responsesError) {
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, responsesMeta{}, invalidResponses("", "invalid JSON body: "+err.Error())
	}
	if root == nil {
		return nil, responsesMeta{}, invalidResponses("", "request body must be a JSON object")
	}

	// model：必填、非空字符串。
	modelVal, ok := root["model"]
	if !ok || modelVal == nil {
		return nil, responsesMeta{}, invalidResponses("model", "model is required")
	}
	model, ok := modelVal.(string)
	if !ok {
		return nil, responsesMeta{}, invalidResponses("model", "model must be a string")
	}
	if strings.TrimSpace(model) == "" {
		return nil, responsesMeta{}, invalidResponses("model", "model must be a non-empty string")
	}

	// input：必填（string 或 array）。
	inputVal, ok := root["input"]
	if !ok || inputVal == nil {
		return nil, responsesMeta{}, invalidResponses("input", "input is required")
	}

	// stream：可选 bool，缺省 false。
	stream := false
	if v, ok := root["stream"]; ok && v != nil {
		b, ok := v.(bool)
		if !ok {
			return nil, responsesMeta{}, invalidResponses("stream", "stream must be a boolean")
		}
		stream = b
	}

	// instructions：可选 string。
	var instructions any
	if v, ok := root["instructions"]; ok && v != nil {
		s, ok := v.(string)
		if !ok {
			return nil, responsesMeta{}, invalidResponses("instructions", "instructions must be a string")
		}
		instructions = s
	}

	// 数值字段：可选 number。
	temperature, terr := optionalNumber(root, "temperature")
	if terr != nil {
		return nil, responsesMeta{}, terr
	}
	topP, terr := optionalNumber(root, "top_p")
	if terr != nil {
		return nil, responsesMeta{}, terr
	}
	maxOut, terr := optionalNumber(root, "max_output_tokens")
	if terr != nil {
		return nil, responsesMeta{}, terr
	}

	// tools：可选 array。
	var tools []any
	if v, ok := root["tools"]; ok && v != nil {
		arr, ok := v.([]any)
		if !ok {
			return nil, responsesMeta{}, invalidResponses("tools", "tools must be an array")
		}
		tools = arr
	}

	// tool_choice：可选 string / object。
	var toolChoice any
	if v, ok := root["tool_choice"]; ok && v != nil {
		toolChoice = v
	}

	// reasoning：可选对象 {effort, summary}。effort 翻译为上游 reasoning_effort
	// （Codex 把推理档位放在这里；缺翻译时档位静默丢失，上游回落默认档）。
	// summary（detailed/auto）上游无法表达，仅在 meta 回显。整个对象非 object
	// （且非 null）按非法请求拒绝。
	var reasoning any
	if v, ok := root["reasoning"]; ok && v != nil {
		obj, isObj := v.(map[string]any)
		if !isObj {
			return nil, responsesMeta{}, invalidResponses("reasoning", "reasoning must be an object")
		}
		reasoning = v
		_ = obj // effort 的提取在 body 构造处（见下方），这里只做形态校验与回显留存
	}

	// metadata / parallel_tool_calls：metadata 接受但不转发（仅回显）；
	// parallel_tool_calls 透传给上游（Codex 显式携带，语义与上游字段一致）。
	var metadata any
	if v, ok := root["metadata"]; ok {
		metadata = v
	}
	var parallel any
	if v, ok := root["parallel_tool_calls"]; ok && v != nil {
		if _, ok := v.(bool); !ok {
			return nil, responsesMeta{}, invalidResponses("parallel_tool_calls", "parallel_tool_calls must be a boolean")
		}
		parallel = v
	}

	// 无状态网关的拒绝项（需服务端存储的字段一律不伪造）。
	if rerr := rejectUnsupported(root); rerr != nil {
		return nil, responsesMeta{}, rerr
	}

	// messages：instructions 作为首条 system 前置，再拼接 input 归一化结果。
	messages := make([]any, 0, 4)
	if s, _ := instructions.(string); s != "" {
		messages = append(messages, map[string]any{"role": "system", "content": s})
	}
	inMsgs, inState, rerr := normalizeResponsesInput(inputVal)
	if rerr != nil {
		return nil, responsesMeta{}, rerr
	}
	messages = append(messages, inMsgs...)

	// tools 映射（Responses 扁平形态 → OpenAI 嵌套 function 形态）。namespace /
	// custom 工具会被摊平为上游可表达的 function，并记录还原映射。input 里
	// additional_tools item 携带的带内工具声明（Codex 会话中途加载的工具）也在这里
	// 并入同一映射。type:"web_search" 是服务端工具：不进上游 tools，转由网关检索
	// （meta.webSearch），渲染层产出 web_search_call item。
	declaredTools := tools
	if len(inState.extraTools) > 0 {
		declaredTools = append(append([]any{}, tools...), inState.extraTools...)
	}
	oTools, toolRefs, ws, rerr := mapResponsesTools(declaredTools)
	if rerr != nil {
		return nil, responsesMeta{}, rerr
	}
	if ws != nil {
		// 查询串取最后一条 user 消息（与 /v1/messages 路径同构）；runWebSearch 里
		// 会再按 query_mode 用模型提炼。
		ws.query = lastUserText(messages)
	}

	body := map[string]any{
		"model":    model,
		"stream":   stream,
		"messages": messages,
	}
	if temperature != nil {
		body["temperature"] = temperature
	}
	if topP != nil {
		body["top_p"] = topP
	}
	if maxOut != nil {
		body["max_tokens"] = maxOut
	}
	if oTools != nil {
		body["tools"] = oTools
	}
	if toolChoice != nil {
		mapped, rerr := mapResponsesToolChoice(toolChoice)
		if rerr != nil {
			return nil, responsesMeta{}, rerr
		}
		body["tool_choice"] = mapped
	}
	if parallel != nil {
		body["parallel_tool_calls"] = parallel
	}
	if reasoning != nil {
		if effort, ok := reasoningEffortOf(reasoning); ok {
			body["reasoning_effort"] = effort
		}
	}

	out, merr := json.Marshal(body)
	if merr != nil {
		return nil, responsesMeta{}, invalidResponses("", "failed to encode normalized request: "+merr.Error())
	}
	meta := responsesMeta{
		model:             model,
		instructions:      instructions,
		tools:             tools,
		toolChoice:        toolChoice,
		temperature:       temperature,
		topP:              topP,
		maxOutputTokens:   maxOut,
		metadata:          metadata,
		parallelToolCalls: parallel,
		reasoning:         reasoning,
		stream:            stream,
		webSearch:         ws,
		toolRefs:          toolRefs,
	}
	return out, meta, nil
}

// reasoningEffortOf 从 reasoning 对象提取 effort 字段（非空字符串才返回）。
func reasoningEffortOf(reasoning any) (string, bool) {
	obj, ok := reasoning.(map[string]any)
	if !ok {
		return "", false
	}
	e, _ := obj["effort"].(string)
	e = strings.TrimSpace(e)
	return e, e != ""
}

// rejectUnsupported 拒绝一切「需要服务端存储历史/会话」的字段（无状态网关）。
// 接受但不转发的字段（metadata / parallel_tool_calls / include / user / service_tier
// / store / truncation:"disabled"）在此不拒绝。
func rejectUnsupported(root map[string]any) *responsesError {
	if v, ok := root["previous_response_id"]; ok && v != nil {
		// 空串等价于「未提供」；非空（含非字符串）一律拒绝——不做服务端会话存储。
		if s, isStr := v.(string); !isStr || strings.TrimSpace(s) != "" {
			return invalidResponses("previous_response_id",
				"previous_response_id is not supported: this gateway is stateless and does not store prior responses")
		}
	}
	if v, ok := root["conversation"]; ok && v != nil {
		return invalidResponses("conversation",
			"conversation is not supported: this gateway is stateless and does not store conversation state")
	}
	if v, ok := root["background"]; ok && v != nil {
		if b, isBool := v.(bool); isBool && b {
			return invalidResponses("background", "background mode is not supported")
		}
	}
	if v, ok := root["truncation"]; ok && v != nil {
		s, isStr := v.(string)
		if !isStr {
			return invalidResponses("truncation", "truncation must be a string")
		}
		switch s {
		case "disabled":
			// 接受（默认），不转发。
		case "auto":
			return invalidResponses("truncation", `truncation:"auto" is not supported: it requires full conversation state`)
		default:
			return invalidResponses("truncation", `truncation must be "disabled"`)
		}
	}
	return nil
}

// responsesInputState 承载 input 数组归一化的跨 item 状态：
//   - pendingReasoning：最近一个 reasoning item 的摘要文本，挂到**下一条** assistant
//     工具调用消息的 reasoning_content 上（DeepSeek 系多轮工具链要求回传推理痕迹，
//     payload 层 backfillReasoningContent 只补空串，真实内容在这里注入）；
//   - extraTools：additional_tools item 带内声明的工具（Responses 形态）；
//   - dropped：按类型计数被丢弃的 item（日志用，避免每 item 一行刷屏）。
type responsesInputState struct {
	pendingReasoning string
	extraTools       []any
	dropped          map[string]int
}

// maxReasoningBackfill 单个 reasoning 摘要回填的长度上限（字符）。摘要是 Codex 回放的
// 历史推理概要，超长摘要回填只会推高 prompt 体积（实测长会话 196 个 item）。
const maxReasoningBackfill = 4096

// normalizeResponsesInput 把 input 归一化为 OpenAI messages 数组。
//   - string → 单条 user 消息；
//   - array  → 逐项归一化；无法表达的历史 item（压缩项 / web_search_call 等）
//     安全丢弃并计数（绝不 400，见文件头「设计纪律」）。
func normalizeResponsesInput(input any) ([]any, *responsesInputState, *responsesError) {
	state := &responsesInputState{}
	switch v := input.(type) {
	case string:
		return []any{map[string]any{"role": "user", "content": v}}, state, nil
	case []any:
		msgs := make([]any, 0, len(v))
		for i, item := range v {
			m, rerr := normalizeResponsesItem(item, i, state)
			if rerr != nil {
				return nil, nil, rerr
			}
			if m != nil {
				msgs = append(msgs, m)
			}
		}
		for typ, n := range state.dropped {
			log.Printf("INFO: [server] responses: dropped %d history item(s) of type %q (not expressible upstream)", n, typ)
		}
		return msgs, state, nil
	default:
		return nil, nil, invalidResponses("input", "input must be a string or an array of items")
	}
}

// normalizeResponsesItem 归一化单个 input item。
// 返回 (nil, nil) = 该 item 无对应消息（被忽略 / 仅记录状态）。
func normalizeResponsesItem(item any, i int, state *responsesInputState) (any, *responsesError) {
	obj, ok := item.(map[string]any)
	if !ok {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d] must be an object", i))
	}
	typ, _ := obj["type"].(string)
	switch typ {
	case "message":
		return normalizeResponsesMessage(obj, i)
	case "function_call":
		m, rerr := normalizeResponsesFunctionCall(obj, i)
		if rerr != nil {
			return nil, rerr
		}
		attachPendingReasoning(m, state)
		return m, nil
	case "function_call_output":
		return normalizeResponsesFunctionCallOutput(obj, i)
	case "custom_tool_call":
		m, rerr := normalizeResponsesCustomToolCall(obj, i)
		if rerr != nil {
			return nil, rerr
		}
		attachPendingReasoning(m, state)
		return m, nil
	case "custom_tool_call_output":
		return normalizeResponsesCustomToolCallOutput(obj, i)
	case "reasoning":
		// 摘要文本进入 pendingReasoning，挂到下一条 assistant 工具调用消息
		// （DeepSeek 多轮工具链的推理回传）；无后续工具调用时自然丢弃。
		if txt := reasoningSummaryText(obj); txt != "" {
			state.pendingReasoning = txt
		}
		return nil, nil
	case "agent_message":
		// 多智能体消息（author→recipient 的 input_text 列表）：按用户可见文本
		// 转成 user 消息，保留来源标注。
		return normalizeResponsesAgentMessage(obj, i)
	case "additional_tools":
		// 带内工具声明（Codex 会话中途加载的工具）：并入本轮 tools 映射。
		if subs, ok := obj["tools"].([]any); ok && len(subs) > 0 {
			state.extraTools = append(state.extraTools, subs...)
		}
		return nil, nil
	case "web_search_call", "local_shell_call", "image_generation_call",
		"tool_search_call", "tool_search_output",
		"compaction", "compaction_summary", "context_compaction", "compaction_trigger",
		"configuration_update", "ghost_snapshot":
		// 历史动作记录 / 压缩占位 / 会话内配置变更：上游无对应形态，丢弃
		// （压缩摘要本地路径是 user 消息形态，这里只会出现加密占位，无法解出）。
		state.noteDropped(typ)
		return nil, nil
	default:
		// 未知类型（客户端版本演进出的新 item）：丢弃 + WARN。历史 item 属于
		// 客户端，逐个 400 会让整个会话无法继续（messages 路径 dropContentBlock
		// 同一教训）。
		state.noteDropped(typ)
		log.Printf("WARN: [server] responses: input[%d]: unknown item type %q dropped", i, typ)
		return nil, nil
	}
}

// noteDropped 计数被丢弃的 item 类型（惰性建表）。
func (st *responsesInputState) noteDropped(typ string) {
	if st.dropped == nil {
		st.dropped = map[string]int{}
	}
	if typ == "" {
		typ = "(missing)"
	}
	st.dropped[typ]++
}

// attachPendingReasoning 把 pendingReasoning 挂到 assistant 工具调用消息的
// reasoning_content（非空时），并清空 pending（只挂一次）。
func attachPendingReasoning(m any, state *responsesInputState) {
	if state.pendingReasoning == "" {
		return
	}
	if msg, ok := m.(map[string]any); ok {
		msg["reasoning_content"] = state.pendingReasoning
	}
	state.pendingReasoning = ""
}

// reasoningSummaryText 提取 reasoning item 的可读文本：summary[].text 拼接，
// 无 summary 时取 content[].text。截断到 maxReasoningBackfill。
func reasoningSummaryText(obj map[string]any) string {
	var sb strings.Builder
	if parts, ok := obj["summary"].([]any); ok {
		for _, p := range parts {
			part, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if txt, ok := part["text"].(string); ok && txt != "" {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(txt)
			}
		}
	}
	if sb.Len() == 0 {
		if parts, ok := obj["content"].([]any); ok {
			for _, p := range parts {
				part, ok := p.(map[string]any)
				if !ok {
					continue
				}
				if txt, ok := part["text"].(string); ok && txt != "" {
					if sb.Len() > 0 {
						sb.WriteString("\n")
					}
					sb.WriteString(txt)
				}
			}
		}
	}
	txt := strings.TrimSpace(sb.String())
	if len(txt) > maxReasoningBackfill {
		txt = txt[:maxReasoningBackfill]
	}
	return txt
}

// normalizeResponsesAgentMessage 归一化 {type:"agent_message",author,recipient,
// content:[{type:"input_text",text}]} → user 消息（保留来源标注；content 兼容纯字符串）。
func normalizeResponsesAgentMessage(obj map[string]any, i int) (any, *responsesError) {
	author, _ := obj["author"].(string)
	recipient, _ := obj["recipient"].(string)
	var texts []string
	switch c := obj["content"].(type) {
	case string:
		if c != "" {
			texts = append(texts, c)
		}
	case []any:
		for _, p := range c {
			part, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if txt, ok := part["text"].(string); ok && txt != "" {
				texts = append(texts, txt)
			}
		}
	default:
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: agent_message content must be a string or an array", i))
	}
	if len(texts) == 0 {
		return nil, nil
	}
	return map[string]any{
		"role": "user",
		"content": fmt.Sprintf("[message from agent %s to %s]\n%s",
			author, recipient, strings.Join(texts, "\n")),
	}, nil
}

// normalizeResponsesMessage 归一化 {type:"message", role, content}。
// role developer → system；content 支持 string 或 parts 数组。
func normalizeResponsesMessage(obj map[string]any, i int) (any, *responsesError) {
	role, _ := obj["role"].(string)
	switch role {
	case "user", "assistant", "system":
	case "developer":
		role = "system"
	default:
		return nil, invalidResponses("input",
			fmt.Sprintf("input[%d]: message role must be one of user/assistant/system/developer", i))
	}
	content, rerr := normalizeResponsesContent(obj["content"], i)
	if rerr != nil {
		return nil, rerr
	}
	return map[string]any{"role": role, "content": content}, nil
}

// normalizeResponsesContent 归一化 message.content：
//   - string → 原样；
//   - parts  → input_text/output_text → {type:"text",text}；
//     input_image → {type:"image_url",image_url:{url,detail?}}。
func normalizeResponsesContent(content any, i int) (any, *responsesError) {
	switch v := content.(type) {
	case string:
		return v, nil
	case []any:
		parts := make([]any, 0, len(v))
		for j, p := range v {
			part, ok := p.(map[string]any)
			if !ok {
				return nil, invalidResponses("input", fmt.Sprintf("input[%d].content[%d] must be an object", i, j))
			}
			pt, _ := part["type"].(string)
			switch pt {
			case "input_text", "output_text":
				txt, _ := part["text"].(string)
				parts = append(parts, map[string]any{"type": "text", "text": txt})
			case "input_image":
				url, _ := part["image_url"].(string)
				if strings.TrimSpace(url) == "" {
					return nil, invalidResponses("input",
						fmt.Sprintf("input[%d].content[%d]: input_image.image_url is required", i, j))
				}
				img := map[string]any{"url": url}
				if d, ok := part["detail"].(string); ok && d != "" {
					img["detail"] = d
				}
				parts = append(parts, map[string]any{"type": "image_url", "image_url": img})
			default:
				return nil, invalidResponses("input",
					fmt.Sprintf("input[%d].content[%d]: unsupported part type %q", i, j, pt))
			}
		}
		return parts, nil
	default:
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: message content must be a string or an array", i))
	}
}

// normalizeResponsesFunctionCall 归一化 {type:"function_call",call_id,name,arguments}
// → OpenAI assistant 消息（tool_calls 单元素）。
func normalizeResponsesFunctionCall(obj map[string]any, i int) (any, *responsesError) {
	callID, _ := obj["call_id"].(string)
	name, _ := obj["name"].(string)
	args, _ := obj["arguments"].(string)
	if strings.TrimSpace(callID) == "" {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: function_call.call_id is required", i))
	}
	if strings.TrimSpace(name) == "" {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: function_call.name is required", i))
	}
	// 历史回放：namespace 工具在上游侧以 "<ns>__<tool>" 摊平名出现，这里还原成同一
	// 摊平名，保证历史里的工具名与本轮 tools 声明逐字一致。
	if ns, _ := obj["namespace"].(string); strings.TrimSpace(ns) != "" {
		name = flattenNamespaceTool(ns, name)
	}
	if _, isStr := obj["arguments"].(string); !isStr && obj["arguments"] != nil {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: function_call.arguments must be a string", i))
	}
	return map[string]any{
		"role": "assistant",
		"tool_calls": []any{map[string]any{
			"id":   callID,
			"type": "function",
			"function": map[string]any{
				"name":      name,
				"arguments": args,
			},
		}},
	}, nil
}

// normalizeResponsesFunctionCallOutput 归一化
// {type:"function_call_output",call_id,output} → OpenAI tool 消息（content 字符串；
// output 为对象时 JSON 编码）。
func normalizeResponsesFunctionCallOutput(obj map[string]any, i int) (any, *responsesError) {
	callID, _ := obj["call_id"].(string)
	if strings.TrimSpace(callID) == "" {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: function_call_output.call_id is required", i))
	}
	content, rerr := responsesOutputContent(obj["output"], "function_call_output", i)
	if rerr != nil {
		return nil, rerr
	}
	return map[string]any{"role": "tool", "tool_call_id": callID, "content": content}, nil
}

// responsesOutputContent 归一化 *_tool_call_output 的 output 字段。Codex 新版会把
// 工具输出编码为结构化内容项数组（[{type:"output_text"|"input_text"|..., text}]，
// 含图片等非文本部分）：文本部分拼接，非文本部分丢弃记日志；字符串原样；对象
// JSON 编码（既有行为）。
func responsesOutputContent(out any, what string, i int) (string, *responsesError) {
	switch v := out.(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	case []any:
		var texts []string
		dropped := 0
		for _, p := range v {
			part, ok := p.(map[string]any)
			if !ok {
				continue
			}
			switch part["type"].(string) {
			case "output_text", "input_text", "text":
				if txt, ok := part["text"].(string); ok && txt != "" {
					texts = append(texts, txt)
				}
			default:
				dropped++
			}
		}
		if dropped > 0 {
			log.Printf("INFO: [server] responses: input[%d]: %s.output dropped %d non-text content part(s)", i, what, dropped)
		}
		return strings.Join(texts, "\n"), nil
	default:
		b, err := json.Marshal(out)
		if err != nil {
			return "", invalidResponses("input",
				fmt.Sprintf("input[%d]: %s.output could not be encoded", i, what))
		}
		return string(b), nil
	}
}

// namespaceToolSeparator 摊平 namespace 工具时的连接符。上游只认扁平 function 名；
// Codex 自身对 MCP 工具也用 "mcp__server__tool" 约定，故沿用 "__"。
const namespaceToolSeparator = "__"

// flattenNamespaceTool 把 (namespace, tool) 拼成上游可表达的扁平 function 名。
func flattenNamespaceTool(namespace, name string) string {
	return namespace + namespaceToolSeparator + name
}

// responsesToolRef 记录摊平后的上游 function 名 → Responses 原始工具形态。
//   - namespace 非空 = 该 function 由 namespace 容器里的子工具摊平而来；
//   - custom = true = 该 function 由 type:"custom"（freeform）工具翻译而来。
type responsesToolRef struct {
	namespace string
	name      string
	custom    bool
}

// mapResponsesTools 把 Responses 工具定义映射为上游 OpenAI 嵌套 function 形态。
//   - {type:"function"} → 原样嵌套（含已嵌套形态）；
//   - {type:"namespace",name,tools} → 每个子工具摊平为 "<ns>__<tool>" function；
//   - {type:"custom",name,format} → 单参数（input:string）function；
//   - {type:"web_search"/"web_search_preview"} → 服务端工具：网关检索执行
//     （meta.webSearch，入口处 runWebSearch），不进上游 tools；
//   - 其余上游无法表达的内置工具 → 跳过并告警。
//
// 返回值 refs 以摊平后的上游函数名为键，渲染层据此把 function_call 还原为
// Codex 期望的 namespace / custom item 形态；ws 非 nil = 检出了 web_search 工具。
func mapResponsesTools(tools []any) ([]any, map[string]responsesToolRef, *messagesWebSearch, *responsesError) {
	if tools == nil {
		return nil, nil, nil, nil
	}
	out := make([]any, 0, len(tools))
	refs := make(map[string]responsesToolRef)
	var ws *messagesWebSearch
	var skipped []string
	for i, t := range tools {
		tool, ok := t.(map[string]any)
		if !ok {
			return nil, nil, nil, invalidResponses("tools", fmt.Sprintf("tools[%d] must be an object", i))
		}
		typ, _ := tool["type"].(string)
		switch typ {
		case "function":
			fn, rerr := mapResponsesFunctionTool(tool, i)
			if rerr != nil {
				return nil, nil, nil, rerr
			}
			out = append(out, map[string]any{"type": "function", "function": fn})
		case "web_search", "web_search_preview":
			if ws == nil {
				ws = &messagesWebSearch{}
				if filters, ok := tool["filters"].(map[string]any); ok {
					ws.allowedDomains = stringSlice(filters["allowed_domains"])
					ws.blockedDomains = stringSlice(filters["blocked_domains"])
				}
				// 顶层形态兼容（OpenAI 规范历史版本）。
				if len(ws.allowedDomains) == 0 {
					ws.allowedDomains = stringSlice(tool["allowed_domains"])
				}
				if len(ws.blockedDomains) == 0 {
					ws.blockedDomains = stringSlice(tool["blocked_domains"])
				}
			}
		case "namespace":
			ns, _ := tool["name"].(string)
			subs, _ := tool["tools"].([]any)
			if strings.TrimSpace(ns) == "" || len(subs) == 0 {
				skipped = append(skipped, "namespace")
				continue
			}
			nsDesc, _ := tool["description"].(string)
			for j, st := range subs {
				sub, ok := st.(map[string]any)
				if !ok {
					return nil, nil, nil, invalidResponses("tools", fmt.Sprintf("tools[%d].tools[%d] must be an object", i, j))
				}
				if styp, _ := sub["type"].(string); styp != "function" {
					skipped = append(skipped, "namespace/"+styp)
					continue
				}
				fn, rerr := mapResponsesFunctionTool(sub, i)
				if rerr != nil {
					return nil, nil, nil, rerr
				}
				bare, _ := fn["name"].(string)
				flat := flattenNamespaceTool(ns, bare)
				fn["name"] = flat
				// 命名空间描述并入子工具描述：上游只见扁平 function，命名空间语义
				// 只能靠描述携带，否则模型不知道这批工具属于同一容器。
				if nsDesc != "" {
					if d, _ := fn["description"].(string); d != "" {
						fn["description"] = nsDesc + "\n\n" + d
					} else {
						fn["description"] = nsDesc
					}
				}
				out = append(out, map[string]any{"type": "function", "function": fn})
				refs[flat] = responsesToolRef{namespace: ns, name: bare}
			}
		case "custom":
			name, _ := tool["name"].(string)
			if strings.TrimSpace(name) == "" {
				return nil, nil, nil, invalidResponses("tools", fmt.Sprintf("tools[%d]: custom tool name is required", i))
			}
			// freeform（grammar）在上游无法表达，退化为单字符串参数 input：
			// 模型把整段文本塞进该参数，渲染层再还原成 custom_tool_call.input。
			fn := map[string]any{
				"name": name,
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"input": map[string]any{
							"type":        "string",
							"description": "Raw tool input text. Pass the tool's freeform payload verbatim.",
						},
					},
					"required": []any{"input"},
				},
			}
			if desc, ok := tool["description"].(string); ok {
				fn["description"] = desc
			}
			out = append(out, map[string]any{"type": "function", "function": fn})
			refs[name] = responsesToolRef{name: name, custom: true}
		default:
			// 其余上游无法表达的内置工具（local_shell 等）：跳过并告警（不 400），
			// 否则 Codex 整体不可用。
			skipped = append(skipped, typ)
		}
	}
	if len(skipped) > 0 {
		log.Printf("WARN: [server] responses: skipped %d tool(s) the upstream cannot express: %v", len(skipped), skipped)
	}
	return out, refs, ws, nil
}

// mapResponsesFunctionTool 归一化单个 Responses function 工具为上游嵌套 function 形态。
// 同时接受已嵌套的 {type:"function",function:{...}}。
func mapResponsesFunctionTool(tool map[string]any, i int) (map[string]any, *responsesError) {
	if inner, ok := tool["function"].(map[string]any); ok {
		if name, _ := inner["name"].(string); strings.TrimSpace(name) == "" {
			return nil, invalidResponses("tools", fmt.Sprintf("tools[%d]: function name is required", i))
		}
		return inner, nil
	}
	name, _ := tool["name"].(string)
	if strings.TrimSpace(name) == "" {
		return nil, invalidResponses("tools", fmt.Sprintf("tools[%d]: function name is required", i))
	}
	fn := map[string]any{"name": name}
	if desc, ok := tool["description"].(string); ok {
		fn["description"] = desc
	}
	if params, ok := tool["parameters"]; ok && params != nil {
		fn["parameters"] = params
	}
	if strict, ok := tool["strict"]; ok && strict != nil {
		fn["strict"] = strict
	}
	return fn, nil
}

// normalizeResponsesCustomToolCall 归一化 {type:"custom_tool_call",call_id,name,input}
// → OpenAI assistant 消息（tool_calls 单元素，arguments 为 {"input":...}）。
// 与 custom 工具声明时的单参数形态对齐，保证历史与本轮 tools 逐字一致。
func normalizeResponsesCustomToolCall(obj map[string]any, i int) (any, *responsesError) {
	callID, _ := obj["call_id"].(string)
	name, _ := obj["name"].(string)
	input, _ := obj["input"].(string)
	if strings.TrimSpace(callID) == "" {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: custom_tool_call.call_id is required", i))
	}
	if strings.TrimSpace(name) == "" {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: custom_tool_call.name is required", i))
	}
	args, _ := json.Marshal(map[string]any{"input": input})
	return map[string]any{
		"role": "assistant",
		"tool_calls": []any{map[string]any{
			"id":   callID,
			"type": "function",
			"function": map[string]any{
				"name":      name,
				"arguments": string(args),
			},
		}},
	}, nil
}

// normalizeResponsesCustomToolCallOutput 归一化 {type:"custom_tool_call_output",call_id,output}
// → OpenAI tool 消息（与 function_call_output 同形）。
func normalizeResponsesCustomToolCallOutput(obj map[string]any, i int) (any, *responsesError) {
	callID, _ := obj["call_id"].(string)
	if strings.TrimSpace(callID) == "" {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: custom_tool_call_output.call_id is required", i))
	}
	content, rerr := responsesOutputContent(obj["output"], "custom_tool_call_output", i)
	if rerr != nil {
		return nil, rerr
	}
	return map[string]any{"role": "tool", "tool_call_id": callID, "content": content}, nil
}

// mapResponsesToolChoice 映射 tool_choice："auto"/"none"/"required" 原样；
// object（{type:"function",name} 或已嵌套 function 形态）→ OpenAI function 形态。
func mapResponsesToolChoice(tc any) (any, *responsesError) {
	switch v := tc.(type) {
	case string:
		switch v {
		case "auto", "none", "required":
			return v, nil
		default:
			return nil, invalidResponses("tool_choice", fmt.Sprintf("tool_choice %q is not supported", v))
		}
	case map[string]any:
		typ, _ := v["type"].(string)
		if typ != "function" {
			return nil, invalidResponses("tool_choice", `tool_choice object must have type "function"`)
		}
		if fn, ok := v["function"].(map[string]any); ok {
			return map[string]any{"type": "function", "function": fn}, nil
		}
		name, _ := v["name"].(string)
		if strings.TrimSpace(name) == "" {
			return nil, invalidResponses("tool_choice", "tool_choice function name is required")
		}
		return map[string]any{"type": "function", "function": map[string]any{"name": name}}, nil
	default:
		return nil, invalidResponses("tool_choice", "tool_choice must be a string or an object")
	}
}

// optionalNumber 读取可选数值字段：缺省/ null → (nil, nil)；非 number → 错误。
func optionalNumber(root map[string]any, key string) (any, *responsesError) {
	v, ok := root[key]
	if !ok || v == nil {
		return nil, nil
	}
	switch v.(type) {
	case float64:
		return v, nil
	default:
		return nil, invalidResponses(key, key+" must be a number")
	}
}
