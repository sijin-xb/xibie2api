package server

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/prompt"
	"github.com/linguo2625469/workbuddy2api-panel/internal/session"
	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// prepareTurn 是协议无关的「请求准备层」：把入站请求体（始终为 OpenAI Chat
// Completions 形态）归一化为单轮执行所需的 chatStat + turnParams，然后交由
// runTurn 执行。它不写任何响应，也不 defer st.done()——调用方负责 defer 与
// runTurn 调用。
//
// 抽取自原 chatCompletions（Stage 4）：/v1/chat/completions、/v1/responses 与
// /v1/messages 共用同一条准备管线（peek → 别名 → realm 解析 → 统计/粘性/轮级键
// → 提示词改写 → 出站模型名改写 → 会话头族 → turnParams）。responses / messages
// 入口先把各自协议请求归一化成 OpenAI body，再调用本方法，因此别名 / realm /
// 提示词 / 脱敏 / 选号 / 出站传输全部沿用既有语义，三种协议不会漂移。
//
// 边界纪律：
//   - 本方法只读 r 的 Header/Context（ExtractClientIP / requestTrace / 会话头），
//     不读 r.Body（body 由调用方读入后传入，读错误由调用方就地 400）。
//   - 返回值 st 的 done() 由调用方 defer；p.renderer 由调用方注入。
func (h *Handler) prepareTurn(r *http.Request, body []byte, renderer turnRenderer) (*chatStat, *turnParams) {
	// 客户端 IP 提取（按请求传递到 ChatStream，不透传时 upstream 侧忽略）；
	// 消除早年共享字段方案的并发交叉污染（issue：ClientIP 竞态）。
	clientIP := upstream.ExtractClientIP(r)

	var peek struct {
		Stream bool   `json:"stream"`
		Model  string `json:"model"`
	}
	_ = json.Unmarshal(body, &peek)

	// 工具名名单：上游在没有 tools 的请求里会把工具调用吐成原生标记文本
	// （见 upstream/dsml.go），修复层用它做严格判定。名单来自「本请求声明的 tools」
	// 与「会话历史里出现过的工具名」两个来源——实测 tools 声明会在中转环节丢失，
	// 只认前者会让修复层在最需要它的场景下失效。
	// 在提示词/模型名改写之前取（那些改写不动 tools / messages[].tool_calls 子树）。
	declaredTools := upstream.ToolNameAllowlist(body)

	// 模型别名（#5）：先对客户端原始模型名做精确匹配，命中则替换为网关目标模型名，
	// 再走既有 realm 前缀解析。未配置别名（nil 表）时 effectiveModel==peek.Model，
	// 行为与引入前逐字节一致。别名只决定"用哪个模型"，后续选号 / 账号策略 / 脱敏 /
	// 提示词改写 / 出站传输全部沿用既有管线，不绕过任何安全处理。
	effectiveModel := peek.Model
	if h.cfg.ModelAliases != nil {
		effectiveModel = h.cfg.ModelAliases.Apply(peek.Model)
	}
	// realm 前缀解析（D6）：model 名可能带 "[realm:]" 前缀。剥出 realm + bareModel，
	// bareModel 用于选号/粘性/出站 body 重写（前缀是网关侧路由协议，上游只认裸名）。
	// 裸名 → ("cn", 原串)，CN 现状零回归。
	realm, bareModel := resolveModel(effectiveModel)
	modelRate := ""
	if h.cfg.Upstream != nil {
		modelRate = h.cfg.Upstream.ModelRate(realm, bareModel)
	}

	// 请求级统计：出口即打一行表格日志（任何路径都会走到）。
	st := newChatStat(time.Now(), body, peek.Stream)
	if tr := requestTraceFrom(r); tr != nil {
		tr.stat = st
		// 来源在 ServeHTTP 入口采集（此时才知道开关与请求头），此处转交给统计对象，
		// 让 stdout 流水行与归档事件共用同一份来源值，两处不会漂移。
		st.clientIP, st.userAgent = tr.clientIP, tr.userAgent
	}

	// 会话粘性：从请求体提取会话键并解析绑定号（找不到/无效则 stickyUID 为空，走普通轮换）。
	// ExtractKey 与粘性开关解耦（issue #35 侧）：关闭粘性时会话头族的聚合主键仍按
	// 会话级（RequestIDForKey(sessKey)），不悄悄退化成轮级——提取本身与粘性无关。
	sessKey := session.ExtractKey(body)
	stickyUID := ""
	if h.cfg.Session != nil && sessKey != "" {
		// 按模型解析：绑定号在**当前模型**被 6004 限额时视为不可用 → 重新分配，
		// 而不是钉在限额号上反复失败（"限额后换不动号"的正解）。
		if uid, ok := h.cfg.Session.ResolveForModel(sessKey, effectiveModel); ok {
			stickyUID = uid
		}
	}

	// 轮级聚合键：按 body 里最后一条 user 消息派生（同轮内所有上游调用同键，
	// 换 user 消息换键）。#170 起带会话键的客户端也统一走轮级（对齐官方桌面 CLI
	// 的 X-Conversation-Request-ID 轮级语义——TraceStartHook 每次 USER_PROMPT_SUBMIT
	// 清空重生成），故不再限 sessKey=="" 才计算；sessKey 由下方派生处以复合键方式
	// 入键（防不同会话同轮文本互撞）。
	// 必须在下方 prompt.Rewrite 之前取——改写会动 messages 内容，之后取会让键漂移。
	turnKey := session.TurnKey(body)

	// gateway_hint 判定所需的请求形态（image_url part）：在改写前取（与 turnKey
	// 同理）。11133「模型不支持图片」指向的前提。
	reqHasImage := hasImagePart(body)

	// 系统提示词改写（出站前、轮转前；每个请求一次）。
	//   - custom：用自有提示词替换客户端 system/developer（从源头消灭 system 指纹误报）。
	//   - append：开头连续 system/developer 块后插自有提示词，既有消息逐字不动
	//     （客户端项目规范/工具约定与网关提示词并用，issue #129）。
	//   - passthrough + 降级期：换 Degraded 中性提示词直达，不再先撞 400。
	//   - passthrough / append 非降级期：透传客户端原始 system（append 则再插一条网关 system）。
	// 降级裁决：append 在降级期退化为 replace（Rewrite(Degraded)）——append 带
	// 指纹原文重试是确定性再撞墙，replace 是一次性最小抢救（issue #129 设计 §4）。
	degradedApplied := false
	if h.cfg.PromptMode == "custom" && h.cfg.PromptText != "" {
		body = prompt.Rewrite(body, h.cfg.PromptText)
	} else if h.cfg.PromptMode == "append" && h.cfg.PromptText != "" && !h.degrade.Active() {
		body = prompt.Append(body, h.cfg.PromptText)
	} else if (h.cfg.PromptMode == "passthrough" || h.cfg.PromptMode == "append") && h.degrade.Active() {
		body = prompt.Rewrite(body, prompt.Degraded)
		degradedApplied = true
	}

	// outbound model 名重写为 bareModel（D6）：realm 前缀是网关侧路由协议，
	// 上游不认前缀（global 账号也请求裸模型名）。peek.Model 保持客户端原始名，
	// 故别名命中或带前缀时 bareModel != peek.Model，都会把 body 的 model 改写为
	// 目标裸名；无别名 + 裸名时二者相等，恒等不改写（零回归）。
	if bareModel != peek.Model {
		body = rewriteModel(body, bareModel)
	}

	// 会话头族（issue #35）：后台按 X-Conversation-Request-ID（对话轮级）聚合请求，
	// 官方客户端一次 user send 内所有 tool call/重试/换号复用同一个 ID。此处**轮转
	// 循环外**生成一次，循环内每次出站原样复用 → 换号/重试/降级全部同 ID，后台不再
	// 碎片化（此前网关一个都不发，上游按 HTTP 请求逐条记账，同一对话几十上百个
	// RequestID）。
	//   - conversationID：body 提取（透传客户端原值）；无 conversation_id 时回退会话键
	//     （见下方说明），两者都空则仍为空串——不伪造；
	//   - conversationRequestID：入站 X-Conversation-Request-ID 透传优先，否则按
	//     粘性 key 进程内稳定生成；粘性 key 也空时走轮级兜底（TurnKey/TurnRequestID），
	//     无 user 消息时退化成本请求级随机——轮转内捕获一次即共享；
	//   - messageID 在 ChatHeaders 内每条消息生成（消息级独立，无需外部可见）。
	// ConversationID 透传客户端对话 id。没有 conversation_id 的客户端（Anthropic /
	// Claude Code 流量）回退到会话键：否则 InjectPromptCacheKey 注入的 prompt_cache_key
	// 会退化成「每账号一个常量」，同账号下所有客户端（Claude Code 会话、桌面端等）
	// 共用一个不断被淘汰的前缀缓存槽，互相覆盖，前缀命中不稳定、常整段 miss。
	// 只影响 prompt_cache_key 取值：带 conversation_id 的客户端行为不变，sessKey 为空
	// 时行为也不变。（上游 20254d8；随本重构一并移植到此新位置。）
	convID := session.ResolveConversationID(body)
	if convID == "" {
		convID = sessKey
	}
	chatMeta := upstream.ChatMeta{ConversationID: convID}
	if v := r.Header.Get("X-Conversation-Request-ID"); v != "" {
		chatMeta.ConversationRequestID = v
	} else if turnKey != "" && sessKey != "" {
		// 轮级复合键：sessKey 入键防跨会话同轮文本互撞（#170 统一轮级）。
		chatMeta.ConversationRequestID = session.TurnRequestID(sessKey + ":" + turnKey)
	} else if turnKey != "" {
		// 无会话键客户端：纯轮级键（既有兜底语义不变，存量会话键值零漂移）。
		chatMeta.ConversationRequestID = session.TurnRequestID(turnKey)
	} else if sessKey != "" {
		// 残留空态兜底（无 user 消息/无可签名内容）：会话级聚合，好于请求级随机。
		chatMeta.ConversationRequestID = session.RequestIDForKey(sessKey)
	} else {
		// 无会话键也无轮级键：请求级随机（轮转内捕获一次即共享）。
		chatMeta.ConversationRequestID = session.TurnRequestID("")
	}
	chatMeta.TraceID = r.Header.Get("X-Trace-ID")

	p := &turnParams{
		stream:        peek.Stream,
		rawModel:      peek.Model,
		bareModel:     bareModel,
		realm:         realm,
		modelRate:     modelRate,
		clientIP:      clientIP,
		body:          body,
		declaredTools: declaredTools,
		sessKey:       sessKey,
		stickyUID:     stickyUID,
		reqHasImage:   reqHasImage,
		chatMeta:      chatMeta,
		degraded:      degradedApplied,
		renderer:      renderer,
	}
	return st, p
}
