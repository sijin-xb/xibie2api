package server

import (
	"io"
	"net/http"

	"github.com/linguo2625469/workbuddy2api-panel/internal/upstream"
)

// turnRenderer 是单轮执行的「协议相关响应渲染」边界：把执行结果按目标协议的
// 线上格式写出。执行语义（选号 / 轮转 / 账号策略 / 统计记账）**不在**此层。
//
// 实现有三个（按入口选择）：openAIRenderer（/v1/chat/completions）、
// responsesRenderer（/v1/responses，见 responses_render.go）、messagesRenderer
// （/v1/messages，见 messages_render.go）。各入口构造对应 renderer 传入 runTurn，
// runTurn 的执行/轮转路径不变。
//
// 边界现状：帧归一化 + DSML 修复已抽成协议无关泵 upstream/frames.go（PumpFrames），
// 各协议经 FrameSink 实现（openAISSEWriter / responsesSink / messagesSink）只负责
// 目标协议的编码与写出。renderStream 仍把 MarkupRepair / HintContext 透传给
// PumpFrames（由它统一做归一化与修复）。
type turnRenderer interface {
	// renderStream 将已归一化的上游 SSE 流写出为流式响应。返回 StreamHint 的错误；
	// 空流 / 写失败的**语义判定**仍由调用方（执行层）负责。
	renderStream(w http.ResponseWriter, stats io.Reader, hint func() upstream.HintContext,
		repair *upstream.MarkupRepair, onErrorFrame func(string)) error

	// renderNonStream 将聚合后的上游响应写出为非流式响应。返回 false 表示转换失败、
	// 已写出错误信封（调用方据此**不得**把该跳记成账号成功）。
	renderNonStream(w http.ResponseWriter, resp map[string]any) bool

	// renderError 写出终态错误响应（hint 为空时不附加 gateway_hint）。
	renderError(w http.ResponseWriter, status int, code, msg, hint string)
}

// openAIRenderer 是 OpenAI Chat Completions 的渲染实现，逐字节复用既有写出函数。
type openAIRenderer struct{}

func (openAIRenderer) renderStream(w http.ResponseWriter, stats io.Reader,
	hint func() upstream.HintContext, repair *upstream.MarkupRepair, onErrorFrame func(string)) error {
	// 推理静默期补 SSE 注释心跳（见 keepalive.go）：与 responses/messages 路径同
	// 语义，避免客户端在长思考回合按空闲超时腰斩。先设头再启心跳，避免心跳在
	// StreamHint 设头之前以默认 Content-Type 提前提交响应头。
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	k := newSSEKeepalive(w, sseKeepaliveInterval)
	defer k.Close()
	return upstream.StreamHint(k, stats, upstream.FrameHintFunc(hint), streamOpts(repair, onErrorFrame)...)
}

// loopGuardEnabled 输出复读检测开关（features.loop_guard，缺省 true）。
//
// 进程级而非每请求：与 chatLogEnabled 同一口径——它是进程配置，不是请求属性。
// 热生效：main 在配置保存时调 SetLoopGuardEnabled，下一次请求即用新值。
var loopGuardEnabled = true

// SetLoopGuardEnabled 设置复读检测开关（main 从 config 注入）。
func SetLoopGuardEnabled(v bool) { loopGuardEnabled = v }

// streamOpts 组装三条协议共用的流式开关：错误帧观察者 + 标记修复 + （开启时）复读检测。
//
// 抽成一处是为了让三条协议不会各自漂移——复读检测若漏挂某一条协议，那条协议
// （例如 Claude Code 走的 /v1/messages）就等于完全不受保护，而这类"只挂了两条"
// 的漏洞在代码里看不出来。
func streamOpts(repair *upstream.MarkupRepair, onErrorFrame func(string)) []upstream.StreamOption {
	opts := []upstream.StreamOption{
		upstream.WithErrorFrameObserver(onErrorFrame),
		upstream.WithMarkupRepair(repair),
	}
	if loopGuardEnabled {
		opts = append(opts, upstream.WithLoopGuard())
	}
	return opts
}

func (openAIRenderer) renderNonStream(w http.ResponseWriter, resp map[string]any) bool {
	writeJSON(w, http.StatusOK, resp)
	return true
}

func (openAIRenderer) renderError(w http.ResponseWriter, status int, code, msg, hint string) {
	writeOpenAIErrorHint(w, status, code, msg, hint)
}
