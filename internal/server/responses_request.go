// responses_request.go 把 OpenAI Responses API（/v1/responses）请求归一化为网关
// 既有的 OpenAI Chat Completions 请求体，使两种协议复用同一条执行管线
// （prepareTurn → runTurn：别名 / realm / 提示词 / 脱敏 / 选号 / 出站传输）。
//
// 设计纪律（支持矩阵，见任务书 Stage 4 §3）：
//   - 归一化只做「请求形态翻译」，**不**做别名 / 提示词 / 脱敏（那些在 prepareTurn 里
//     对归一化后的 OpenAI body 执行）。归一化出的 body["model"] 必须是**原始客户端
//     model 字符串**，这样 peek.Model / 别名 / 出站重写与 chat 路径逐字节一致。
//   - 网关是无状态的：任何需要「服务端存储的历史响应/会话」的字段一律 **400 拒绝**，
//     不伪造、不假装支持。已支持字段映射见下方 mapXxx；接受但不转发字段在 meta 里
//     原样带回（回显），绝不落盘。
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
	stream            bool
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

	// metadata / parallel_tool_calls：接受但不转发（仅回显）。
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
	inMsgs, rerr := normalizeResponsesInput(inputVal)
	if rerr != nil {
		return nil, responsesMeta{}, rerr
	}
	messages = append(messages, inMsgs...)

	// tools 映射（Responses 扁平形态 → OpenAI 嵌套 function 形态）。namespace /
	// custom 工具会被摊平为上游可表达的 function，并记录还原映射。
	oTools, toolRefs, rerr := mapResponsesTools(tools)
	if rerr != nil {
		return nil, responsesMeta{}, rerr
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
		stream:            stream,
		toolRefs:          toolRefs,
	}
	return out, meta, nil
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

// normalizeResponsesInput 把 input 归一化为 OpenAI messages 数组。
//   - string → 单条 user 消息；
//   - array  → 逐项归一化（reasoning 项安全忽略，返回 nil 被过滤）。
func normalizeResponsesInput(input any) ([]any, *responsesError) {
	switch v := input.(type) {
	case string:
		return []any{map[string]any{"role": "user", "content": v}}, nil
	case []any:
		msgs := make([]any, 0, len(v))
		for i, item := range v {
			m, rerr := normalizeResponsesItem(item, i)
			if rerr != nil {
				return nil, rerr
			}
			if m != nil {
				msgs = append(msgs, m)
			}
		}
		return msgs, nil
	default:
		return nil, invalidResponses("input", "input must be a string or an array of items")
	}
}

// normalizeResponsesItem 归一化单个 input item；reasoning 项返回 (nil, nil) 表示忽略。
func normalizeResponsesItem(item any, i int) (any, *responsesError) {
	obj, ok := item.(map[string]any)
	if !ok {
		return nil, invalidResponses("input", fmt.Sprintf("input[%d] must be an object", i))
	}
	typ, _ := obj["type"].(string)
	switch typ {
	case "message":
		return normalizeResponsesMessage(obj, i)
	case "function_call":
		return normalizeResponsesFunctionCall(obj, i)
	case "function_call_output":
		return normalizeResponsesFunctionCallOutput(obj, i)
	case "custom_tool_call":
		return normalizeResponsesCustomToolCall(obj, i)
	case "custom_tool_call_output":
		return normalizeResponsesCustomToolCallOutput(obj, i)
	case "reasoning":
		// 安全忽略：本网关无 reasoning item 的加密内容/摘要数据，不臆造、不转发。
		return nil, nil
	default:
		return nil, invalidResponses("input", fmt.Sprintf("input[%d]: unsupported item type %q", i, typ))
	}
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
	var content string
	switch out := obj["output"].(type) {
	case nil:
		content = ""
	case string:
		content = out
	default:
		b, err := json.Marshal(out)
		if err != nil {
			return nil, invalidResponses("input",
				fmt.Sprintf("input[%d]: function_call_output.output could not be encoded", i))
		}
		content = string(b)
	}
	return map[string]any{"role": "tool", "tool_call_id": callID, "content": content}, nil
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
//   - 其余（type:"web_search" 等上游无法表达的内置工具）→ 跳过并告警。
//
// 返回值 refs 以摊平后的上游函数名为键，渲染层据此把 function_call 还原为
// Codex 期望的 namespace / custom item 形态。
func mapResponsesTools(tools []any) ([]any, map[string]responsesToolRef, *responsesError) {
	if tools == nil {
		return nil, nil, nil
	}
	out := make([]any, 0, len(tools))
	refs := make(map[string]responsesToolRef)
	var skipped []string
	for i, t := range tools {
		tool, ok := t.(map[string]any)
		if !ok {
			return nil, nil, invalidResponses("tools", fmt.Sprintf("tools[%d] must be an object", i))
		}
		typ, _ := tool["type"].(string)
		switch typ {
		case "function":
			fn, rerr := mapResponsesFunctionTool(tool, i)
			if rerr != nil {
				return nil, nil, rerr
			}
			out = append(out, map[string]any{"type": "function", "function": fn})
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
					return nil, nil, invalidResponses("tools", fmt.Sprintf("tools[%d].tools[%d] must be an object", i, j))
				}
				if styp, _ := sub["type"].(string); styp != "function" {
					skipped = append(skipped, "namespace/"+styp)
					continue
				}
				fn, rerr := mapResponsesFunctionTool(sub, i)
				if rerr != nil {
					return nil, nil, rerr
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
				return nil, nil, invalidResponses("tools", fmt.Sprintf("tools[%d]: custom tool name is required", i))
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
			// type:"web_search" 等上游无法表达的内置工具：搜索由服务端执行器完成，
			// 网关无法替代。跳过并告警（不 400），否则 Codex 整体不可用。
			skipped = append(skipped, typ)
		}
	}
	if len(skipped) > 0 {
		log.Printf("WARN: [server] responses: skipped %d tool(s) the upstream cannot express: %v", len(skipped), skipped)
	}
	return out, refs, nil
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
	var content string
	switch out := obj["output"].(type) {
	case nil:
		content = ""
	case string:
		content = out
	default:
		b, err := json.Marshal(out)
		if err != nil {
			return nil, invalidResponses("input",
				fmt.Sprintf("input[%d]: custom_tool_call_output.output could not be encoded", i))
		}
		content = string(b)
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
