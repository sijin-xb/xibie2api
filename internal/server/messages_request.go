// messages_request.go 把 Anthropic Messages API（/v1/messages）请求归一化为网关
// 既有的 OpenAI Chat Completions 请求体，使两种协议复用同一条执行管线
// （prepareTurn → runTurn：别名 / realm / 提示词 / 脱敏 / 选号 / 出站传输）。
//
// 设计纪律（支持矩阵，见任务书 Stage 5 §3）：
//   - 归一化只做「请求形态翻译」，**不**做别名 / 提示词 / 脱敏（那些在 prepareTurn 里
//     对归一化后的 OpenAI body 执行）。归一化出的 body["model"] 必须是**原始客户端
//     model 字符串**，这样 peek.Model / 别名 / 出站重写与 chat 路径逐字节一致。
//   - 网关是无状态的：任何会改变生成语义、而上游 OpenAI 形态又无法表达的字段一律
//     **400 拒绝**（top_k / thinking / output_config / container / mcp_servers /
//     context_management / service_tier），绝不静默丢弃。
//   - metadata（含 metadata.user_id）**接受但不转发**（语义惰性，仅回显需要才留存；
//     本网关不回显，故直接忽略）。
//
// 支持矩阵（→ 出站 OpenAI body）：
//   - model                 → body["model"]（原始客户端串，交给 prepareTurn 解析）
//   - max_tokens (必填, 正整数) → body["max_tokens"]
//   - messages (必填数组)    → body["messages"]（见 normalizeMessagesArray）
//   - system (string | text 块数组) → 前置一条 role:"system"
//   - messages 内的 role:"system"（mid-conversation-system beta）→ 原地一条 role:"system"
//   - stream (缺省 false)    → body["stream"]
//   - temperature / top_p    → 同名转发
//   - stop_sequences         → body["stop"]
//   - tools ({name,description,input_schema}) → {type:"function",function:{...}}
//   - tool_choice ({type:auto|any|none|tool}) → "auto"|"required"|"none"|{function}
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"github.com/linguo2625469/workbuddy2api-panel/internal/httpauth"
)

// messagesMeta 承载 Anthropic 响应对象回显所需的原始请求字段（归一化后的 OpenAI
// body 之外的信息）。渲染层只用它构造 Anthropic 对象，不参与执行。
type messagesMeta struct {
	model  string // 原始客户端 model（回显）
	stream bool
}

// messagesError 归一化/读体阶段的请求级错误（映射为 Anthropic 错误信封）。
type messagesError struct {
	status  int
	typ     string
	message string
}

func invalidMessages(field, message string) *messagesError {
	if field != "" {
		message = field + ": " + message
	}
	return &messagesError{status: http.StatusBadRequest, typ: "invalid_request_error", message: message}
}

// unsupportedMessages 构造「字段被明确拒绝」的错误：命名该字段并说明原因。
// 拒绝而非静默丢弃的理由：这些字段会改变生成语义，而网关无状态、上游 OpenAI
// 形态无法表达——静默丢弃会让客户端以为设置生效（如 top_k 截断、thinking 预算）。
func unsupportedMessages(field string) *messagesError {
	return &messagesError{
		status:  http.StatusBadRequest,
		typ:     "invalid_request_error",
		message: field + " is not supported by this gateway: it changes generation semantics and cannot be represented by the upstream; remove the field and retry",
	}
}

// messages 是 /v1/messages 入口：读体 → 归一化为 OpenAI body → 共享准备层 →
// runTurn。与 chatCompletions 的唯一差别是「读体后先归一化」这一步。
func (h *Handler) messages(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeMessagesError(w, http.StatusBadRequest, "invalid_request_error", "read body: "+err.Error())
		return
	}
	openAIBody, meta, merr := normalizeMessagesRequest(raw)
	if merr != nil {
		writeMessagesError(w, merr.status, merr.typ, merr.message)
		return
	}
	st, p := h.prepareTurn(r, openAIBody, messagesRenderer{meta: meta})
	defer st.done()
	h.runTurn(r.Context(), w, st, p)
}

// withAuthMessages 是 /v1/messages 的鉴权包装：复用 httpauth.VerifyBearer 与热改
// api_key（loadLive），失败时写 **Anthropic** 错误信封（401 authentication_error），
// 与 OpenAI 路径的 withAuth 互不影响。
func (h *Handler) withAuthMessages(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !httpauth.VerifyBearer(r, h.loadLive().APIKey) {
			writeMessagesError(w, http.StatusUnauthorized, "authentication_error", "missing or invalid API key")
			return
		}
		next(w, r)
	}
}

// normalizeMessagesRequest 解析并归一化一个 Anthropic Messages 请求。
// 返回：OpenAI Chat Completions 请求体（JSON）+ 回显 meta + 请求级错误（nil = OK）。
func normalizeMessagesRequest(raw []byte) ([]byte, messagesMeta, *messagesError) {
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, messagesMeta{}, invalidMessages("", "invalid JSON body: "+err.Error())
	}
	if root == nil {
		return nil, messagesMeta{}, invalidMessages("", "request body must be a JSON object")
	}

	// model：必填、非空字符串。
	modelVal, ok := root["model"]
	if !ok || modelVal == nil {
		return nil, messagesMeta{}, invalidMessages("model", "model is required")
	}
	model, ok := modelVal.(string)
	if !ok {
		return nil, messagesMeta{}, invalidMessages("model", "model must be a string")
	}
	if strings.TrimSpace(model) == "" {
		return nil, messagesMeta{}, invalidMessages("model", "model must be a non-empty string")
	}

	// max_tokens：必填、正整数（Anthropic 语义：缺省即错误，非正/非整数一律拒绝）。
	maxTokens, merr := requiredPositiveInt(root, "max_tokens")
	if merr != nil {
		return nil, messagesMeta{}, merr
	}

	// messages：必填数组。
	messagesVal, ok := root["messages"]
	if !ok || messagesVal == nil {
		return nil, messagesMeta{}, invalidMessages("messages", "messages is required")
	}
	messagesArr, ok := messagesVal.([]any)
	if !ok {
		return nil, messagesMeta{}, invalidMessages("messages", "messages must be an array")
	}

	// stream：可选 bool，缺省 false。
	stream := false
	if v, ok := root["stream"]; ok && v != nil {
		b, ok := v.(bool)
		if !ok {
			return nil, messagesMeta{}, invalidMessages("stream", "stream must be a boolean")
		}
		stream = b
	}

	// temperature / top_p：可选 number。
	temperature, merr := messagesOptionalNumber(root, "temperature")
	if merr != nil {
		return nil, messagesMeta{}, merr
	}
	topP, merr := messagesOptionalNumber(root, "top_p")
	if merr != nil {
		return nil, messagesMeta{}, merr
	}

	// stop_sequences：可选 string 数组 → OpenAI stop。
	var stop []any
	if v, ok := root["stop_sequences"]; ok && v != nil {
		arr, ok := v.([]any)
		if !ok {
			return nil, messagesMeta{}, invalidMessages("stop_sequences", "stop_sequences must be an array of strings")
		}
		for i, s := range arr {
			str, ok := s.(string)
			if !ok {
				return nil, messagesMeta{}, invalidMessages(fmt.Sprintf("stop_sequences[%d]", i), "stop sequence must be a string")
			}
			stop = append(stop, str)
		}
	}

	// system：可选 string 或 text 块数组 → 单条 system 文本。
	system, merr := mapAnthropicSystem(root["system"])
	if merr != nil {
		return nil, messagesMeta{}, merr
	}

	// tools：可选数组。
	var tools []any
	if v, ok := root["tools"]; ok && v != nil {
		arr, ok := v.([]any)
		if !ok {
			return nil, messagesMeta{}, invalidMessages("tools", "tools must be an array")
		}
		tools = arr
	}

	// tool_choice：可选对象。
	var toolChoice any
	if v, ok := root["tool_choice"]; ok && v != nil {
		toolChoice = v
	}

	// 明确拒绝项（改变生成语义 / 网关无法表达）：命名字段并 400，绝不静默丢弃。
	for _, field := range []string{
		"top_k", "thinking", "output_config", "container",
		"mcp_servers", "context_management", "service_tier",
	} {
		if v, ok := root[field]; ok && v != nil {
			return nil, messagesMeta{}, unsupportedMessages(field)
		}
	}
	// metadata（含 metadata.user_id）：接受但忽略（不转发、不回显）。

	// messages 归一化（tool_use/tool_result/文本/图片块）。
	oMsgs, merr := normalizeMessagesArray(messagesArr)
	if merr != nil {
		return nil, messagesMeta{}, merr
	}

	// system 前置为第一条 system 消息。
	if system != "" {
		oMsgs = append([]any{map[string]any{"role": "system", "content": system}}, oMsgs...)
	}

	oTools, merr := mapAnthropicTools(tools)
	if merr != nil {
		return nil, messagesMeta{}, merr
	}

	body := map[string]any{
		"model":      model,
		"stream":     stream,
		"messages":   oMsgs,
		"max_tokens": maxTokens,
	}
	if temperature != nil {
		body["temperature"] = temperature
	}
	if topP != nil {
		body["top_p"] = topP
	}
	if len(stop) > 0 {
		body["stop"] = stop
	}
	if oTools != nil {
		body["tools"] = oTools
	}
	if toolChoice != nil {
		mapped, merr := mapAnthropicToolChoice(toolChoice)
		if merr != nil {
			return nil, messagesMeta{}, merr
		}
		body["tool_choice"] = mapped
	}

	out, jerr := json.Marshal(body)
	if jerr != nil {
		return nil, messagesMeta{}, invalidMessages("", "failed to encode normalized request: "+jerr.Error())
	}
	return out, messagesMeta{model: model, stream: stream}, nil
}

// normalizeMessagesArray 把 Anthropic messages 数组逐条归一化为 OpenAI messages。
// 顺序保持：assistant 的 text+tool_use 合成单条 assistant（content + tool_calls）；
// user 的 tool_result 块单独拆成 role:"tool" 消息，绝不当成用户文本。
func normalizeMessagesArray(messages []any) ([]any, *messagesError) {
	out := make([]any, 0, len(messages))
	for i, m := range messages {
		obj, ok := m.(map[string]any)
		if !ok {
			return nil, invalidMessages(fmt.Sprintf("messages[%d]", i), "message must be an object")
		}
		role, _ := obj["role"].(string)
		switch role {
		case "user":
			msgs, merr := normalizeUserMessage(obj, i)
			if merr != nil {
				return nil, merr
			}
			out = append(out, msgs...)
		case "assistant":
			msg, merr := normalizeAssistantMessage(obj, i)
			if merr != nil {
				return nil, merr
			}
			out = append(out, msg)
		case "system":
			msg, merr := normalizeSystemMessage(obj, i)
			if merr != nil {
				return nil, merr
			}
			out = append(out, msg)
		default:
			return nil, invalidMessages(fmt.Sprintf("messages[%d].role", i), `role must be "user", "assistant", or "system"`)
		}
	}
	return out, nil
}

// normalizeUserMessage 归一化一条 user 消息，返回一到多条 OpenAI 消息（顺序保持）：
//   - content 为 string → 单条 {role:"user",content:string}；
//   - content 为数组 → 文本/图片块聚合为 user 消息；tool_result 块在此处「切分」成
//     独立的 role:"tool" 消息（前序未决文本先落盘为一条 user，再落 tool 消息）。
func normalizeUserMessage(obj map[string]any, i int) ([]any, *messagesError) {
	switch v := obj["content"].(type) {
	case string:
		return []any{map[string]any{"role": "user", "content": v}}, nil
	case []any:
		var out []any
		var texts []string
		var parts []any
		hasImage := false
		flush := func() {
			if !hasImage && len(texts) == 0 {
				return
			}
			if hasImage {
				out = append(out, map[string]any{"role": "user", "content": parts})
			} else {
				out = append(out, map[string]any{"role": "user", "content": strings.Join(texts, "")})
			}
			texts = nil
			parts = nil
			hasImage = false
		}
		for j, b := range v {
			block, ok := b.(map[string]any)
			if !ok {
				return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d]", i, j), "content block must be an object")
			}
			typ, _ := block["type"].(string)
			switch typ {
			case "text":
				txt, _ := block["text"].(string)
				texts = append(texts, txt)
				parts = append(parts, map[string]any{"type": "text", "text": txt})
			case "image":
				part, merr := mapAnthropicImage(block, i, j)
				if merr != nil {
					return nil, merr
				}
				hasImage = true
				parts = append(parts, part)
			case "tool_result":
				flush()
				toolMsg, merr := mapAnthropicToolResult(block, i, j)
				if merr != nil {
					return nil, merr
				}
				out = append(out, toolMsg)
			default:
				return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d].type", i, j),
					fmt.Sprintf("unsupported content block type %q", typ))
			}
		}
		flush()
		if len(out) == 0 {
			// 空 content 数组：落一条空 user 消息，保持消息计数（不丢轮次）。
			out = append(out, map[string]any{"role": "user", "content": ""})
		}
		return out, nil
	default:
		return nil, invalidMessages(fmt.Sprintf("messages[%d].content", i), "content must be a string or an array")
	}
}

// normalizeAssistantMessage 归一化一条 assistant 消息：文本块拼成 content，tool_use
// 块汇成 tool_calls（同一 assistant 消息内 text + tool_use 共存）。
func normalizeAssistantMessage(obj map[string]any, i int) (any, *messagesError) {
	switch v := obj["content"].(type) {
	case string:
		return map[string]any{"role": "assistant", "content": v}, nil
	case []any:
		var texts []string
		var toolCalls []any
		for j, b := range v {
			block, ok := b.(map[string]any)
			if !ok {
				return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d]", i, j), "content block must be an object")
			}
			typ, _ := block["type"].(string)
			switch typ {
			case "text":
				txt, _ := block["text"].(string)
				texts = append(texts, txt)
			case "tool_use":
				tc, merr := mapAnthropicToolUse(block, i, j)
				if merr != nil {
					return nil, merr
				}
				toolCalls = append(toolCalls, tc)
			default:
				return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d].type", i, j),
					fmt.Sprintf("unsupported content block type %q", typ))
			}
		}
		msg := map[string]any{"role": "assistant", "content": strings.Join(texts, "")}
		if len(toolCalls) > 0 {
			msg["tool_calls"] = toolCalls
		}
		return msg, nil
	default:
		return nil, invalidMessages(fmt.Sprintf("messages[%d].content", i), "content must be a string or an array")
	}
}

// normalizeSystemMessage 归一化一条 messages 数组内的 role:"system" 消息
// （Anthropic mid-conversation-system beta，Claude Code 用它插入会话中途的提醒）。
// content 为 string 或 text 块数组，按序拼成单条 role:"system" OpenAI 消息，保持
// 其在会话中的原位置——不折叠进顶层 system（语义不同：顶层是会话级规范，中途
// system 是随轮次出现的提醒）。
func normalizeSystemMessage(obj map[string]any, i int) (any, *messagesError) {
	switch v := obj["content"].(type) {
	case string:
		return map[string]any{"role": "system", "content": v}, nil
	case []any:
		var sb strings.Builder
		for j, b := range v {
			block, ok := b.(map[string]any)
			if !ok {
				return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d]", i, j), "content block must be an object")
			}
			typ, _ := block["type"].(string)
			if typ != "text" {
				return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d].type", i, j),
					fmt.Sprintf("unsupported system content block type %q", typ))
			}
			txt, _ := block["text"].(string)
			sb.WriteString(txt)
		}
		return map[string]any{"role": "system", "content": sb.String()}, nil
	default:
		return nil, invalidMessages(fmt.Sprintf("messages[%d].content", i), "content must be a string or an array")
	}
}

// mapAnthropicImage 把 {type:"image",source:{...}} 映射为 OpenAI image_url part。
// 仅用于 user 消息（错误路径前缀按 messages[i].content[j] 生成）。
func mapAnthropicImage(block map[string]any, i, j int) (map[string]any, *messagesError) {
	return mapAnthropicImageAt(block, fmt.Sprintf("messages[%d].content[%d]", i, j))
}

// mapAnthropicImageAt 是 mapAnthropicImage 的实现，path 为错误消息里的字段前缀
// （user 消息用 messages[i].content[j]，tool_result 内嵌图片用更长的路径）。
// base64 源拼 data URI；url 源直接透传；其余 source.type 一律拒绝。
func mapAnthropicImageAt(block map[string]any, path string) (map[string]any, *messagesError) {
	src, ok := block["source"].(map[string]any)
	if !ok {
		return nil, invalidMessages(path+".source", "image.source must be an object")
	}
	stype, _ := src["type"].(string)
	switch stype {
	case "base64":
		mediaType, _ := src["media_type"].(string)
		data, _ := src["data"].(string)
		if strings.TrimSpace(mediaType) == "" || strings.TrimSpace(data) == "" {
			return nil, invalidMessages(path+".source",
				"image.source.media_type and image.source.data are required for base64 source")
		}
		return map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": "data:" + mediaType + ";base64," + data},
		}, nil
	case "url":
		url, _ := src["url"].(string)
		if strings.TrimSpace(url) == "" {
			return nil, invalidMessages(path+".source",
				"image.source.url is required for url source")
		}
		return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}, nil
	default:
		return nil, invalidMessages(path+".source.type",
			fmt.Sprintf("unsupported image source type %q", stype))
	}
}

// mapAnthropicToolResult 把 {type:"tool_result",tool_use_id,content,is_error} 映射为
// OpenAI role:"tool" 消息。content 为数组时支持 text 与 image 块：只含文本 → 拼成
// 字符串（与既有形态一致，兼容性最好）；含图片 → 输出多模态 parts 数组（OpenAI 的
// tool 消息 content 允许是 part 数组，Claude Code 的 Read 工具会返回图片）。
// is_error 不映射（OpenAI 无对应字段，语义由文本表达——不伪造）。
func mapAnthropicToolResult(block map[string]any, i, j int) (map[string]any, *messagesError) {
	id, _ := block["tool_use_id"].(string)
	if strings.TrimSpace(id) == "" {
		return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d].tool_use_id", i, j), "tool_result.tool_use_id is required")
	}
	base := fmt.Sprintf("messages[%d].content[%d]", i, j)
	switch c := block["content"].(type) {
	case nil:
		return map[string]any{"role": "tool", "tool_call_id": id, "content": ""}, nil
	case string:
		return map[string]any{"role": "tool", "tool_call_id": id, "content": c}, nil
	case []any:
		var sb strings.Builder
		parts := make([]any, 0, len(c))
		hasImage := false
		for k, p := range c {
			pm, ok := p.(map[string]any)
			if !ok {
				return nil, invalidMessages(fmt.Sprintf("%s.content[%d]", base, k), "tool_result content block must be an object")
			}
			switch pt, _ := pm["type"].(string); pt {
			case "text":
				txt, _ := pm["text"].(string)
				sb.WriteString(txt)
				parts = append(parts, map[string]any{"type": "text", "text": txt})
			case "image":
				part, merr := mapAnthropicImageAt(pm, fmt.Sprintf("%s.content[%d]", base, k))
				if merr != nil {
					return nil, merr
				}
				hasImage = true
				parts = append(parts, part)
			default:
				return nil, invalidMessages(fmt.Sprintf("%s.content[%d].type", base, k),
					fmt.Sprintf("unsupported tool_result content block type %q", pt))
			}
		}
		if hasImage {
			return map[string]any{"role": "tool", "tool_call_id": id, "content": parts}, nil
		}
		return map[string]any{"role": "tool", "tool_call_id": id, "content": sb.String()}, nil
	default:
		return nil, invalidMessages(base+".content", "tool_result.content must be a string or an array")
	}
}

// mapAnthropicToolUse 把 {type:"tool_use",id,name,input} 映射为 OpenAI tool_call。
// input 缺省 → {}；input 以 JSON 编码为 arguments 字符串（保证 OpenAI 形态一致）。
func mapAnthropicToolUse(block map[string]any, i, j int) (map[string]any, *messagesError) {
	id, _ := block["id"].(string)
	name, _ := block["name"].(string)
	if strings.TrimSpace(id) == "" {
		return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d].id", i, j), "tool_use.id is required")
	}
	if strings.TrimSpace(name) == "" {
		return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d].name", i, j), "tool_use.name is required")
	}
	input := block["input"]
	if input == nil {
		input = map[string]any{}
	}
	args, err := json.Marshal(input)
	if err != nil {
		return nil, invalidMessages(fmt.Sprintf("messages[%d].content[%d].input", i, j), "tool_use.input could not be encoded")
	}
	return map[string]any{
		"id":   id,
		"type": "function",
		"function": map[string]any{
			"name":      name,
			"arguments": string(args),
		},
	}, nil
}

// mapAnthropicSystem 归一化 system：string 原样；数组仅接受 {type:"text",text} 块
// （按序拼接）；缺省 → 空串（不前置 system 消息）。
func mapAnthropicSystem(system any) (string, *messagesError) {
	switch v := system.(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	case []any:
		var sb strings.Builder
		for i, b := range v {
			block, ok := b.(map[string]any)
			if !ok {
				return "", invalidMessages(fmt.Sprintf("system[%d]", i), "system block must be an object")
			}
			typ, _ := block["type"].(string)
			if typ != "text" {
				return "", invalidMessages(fmt.Sprintf("system[%d].type", i), fmt.Sprintf("unsupported system block type %q", typ))
			}
			txt, _ := block["text"].(string)
			sb.WriteString(txt)
		}
		return sb.String(), nil
	default:
		return "", invalidMessages("system", "system must be a string or an array of text blocks")
	}
}

// mapAnthropicTools 把 Anthropic 工具定义映射为 OpenAI 嵌套 function 形态。
func mapAnthropicTools(tools []any) ([]any, *messagesError) {
	if tools == nil {
		return nil, nil
	}
	out := make([]any, 0, len(tools))
	for i, t := range tools {
		tool, ok := t.(map[string]any)
		if !ok {
			return nil, invalidMessages(fmt.Sprintf("tools[%d]", i), "tool must be an object")
		}
		name, _ := tool["name"].(string)
		if strings.TrimSpace(name) == "" {
			return nil, invalidMessages(fmt.Sprintf("tools[%d].name", i), "tool name is required")
		}
		fn := map[string]any{"name": name}
		if desc, ok := tool["description"].(string); ok {
			fn["description"] = desc
		}
		if schema, ok := tool["input_schema"]; ok && schema != nil {
			fn["parameters"] = schema
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out, nil
}

// mapAnthropicToolChoice 映射 tool_choice：
//   - {type:"auto"} → "auto"
//   - {type:"any"}  → "required"
//   - {type:"none"} → "none"
//   - {type:"tool",name} → {type:"function",function:{name}}
func mapAnthropicToolChoice(tc any) (any, *messagesError) {
	obj, ok := tc.(map[string]any)
	if !ok {
		return nil, invalidMessages("tool_choice", "tool_choice must be an object")
	}
	typ, _ := obj["type"].(string)
	switch typ {
	case "auto":
		return "auto", nil
	case "any":
		return "required", nil
	case "none":
		return "none", nil
	case "tool":
		name, _ := obj["name"].(string)
		if strings.TrimSpace(name) == "" {
			return nil, invalidMessages("tool_choice.name", `tool_choice.name is required when type is "tool"`)
		}
		return map[string]any{"type": "function", "function": map[string]any{"name": name}}, nil
	default:
		return nil, invalidMessages("tool_choice.type", fmt.Sprintf("unsupported tool_choice type %q", typ))
	}
}

// requiredPositiveInt 读取必填正整数（Anthropic max_tokens）。
func requiredPositiveInt(root map[string]any, key string) (any, *messagesError) {
	v, ok := root[key]
	if !ok || v == nil {
		return nil, invalidMessages(key, key+" is required")
	}
	n, ok := v.(float64)
	if !ok {
		return nil, invalidMessages(key, key+" must be a positive integer")
	}
	if n <= 0 || n != math.Trunc(n) {
		return nil, invalidMessages(key, key+" must be a positive integer")
	}
	return n, nil
}

// messagesOptionalNumber 读取可选数值字段：缺省 / null → (nil, nil)；非 number → 错误。
func messagesOptionalNumber(root map[string]any, key string) (any, *messagesError) {
	v, ok := root[key]
	if !ok || v == nil {
		return nil, nil
	}
	if _, ok := v.(float64); !ok {
		return nil, invalidMessages(key, key+" must be a number")
	}
	return v, nil
}
