// keepalive.go 给流式响应加周期性 SSE 注释心跳，消除「推理静默期被客户端空闲
// 超时误判断流」这一类问题。
//
// 背景（实测）：上游在长推理阶段可能连续数分钟一个 data 帧都不发
// （deepseek-v4.1-flash 实测静默 >300s，见 2026-10-10 的 stream_error 归档）。
// 客户端按**字节空闲**判定流断开——Codex 的 stream_idle_timeout_ms 属于
// ModelProviderInfo（HTTP 传输层参数，与 request_max_retries 同级），报
// "idle timeout waiting for SSE" 后重试整个回合，长回合被腰斩。
//
// 方案：网关在写流的同时按固定周期补一帧标准 SSE 注释（": keepalive"）。注释行
// 被所有标准 SSE 解析器忽略（对事件序列零影响），但提供字节活动，客户端空闲计时
// 器因此不会走满。
//
// 边界：本心跳只解决**客户端**侧超时。网关自己掐上游读的
// upstream.idle_timeout_seconds 是另一层——上游真静默超过该值时网关仍会主动断流，
// 需按模型最长静默时间把该配置调大（推荐 1800s，见 README 的提示词/超时章节）。
package server

import (
	"io"
	"net/http"
	"sync"
	"time"
)

// sseKeepaliveInterval 心跳间隔。需明显小于常见客户端空闲阈值（Codex 默认约
// 300s），确保静默期不会触发客户端断流重试。变量形式便于测试注入短间隔。
var sseKeepaliveInterval = 15 * time.Second

// sseKeepalive 包装底层 ResponseWriter：
//   - Write/Flush 与心跳写共用一把锁，保证完整 SSE 帧与心跳不交错写坏；
//   - 后台 goroutine 按 interval 发一帧 ": keepalive\n\n" 并 flush；
//   - Close 停掉后台 goroutine 并等待其退出（渲染返回时必须调用）。
//
// 实现 http.ResponseWriter + http.Flusher，可直接交给 upstream.StreamHint 或各
// 协议的 FrameSink 使用。
type sseKeepalive struct {
	mu   sync.Mutex
	w    http.ResponseWriter
	stop chan struct{}
	done chan struct{}
	once sync.Once
}

func newSSEKeepalive(w http.ResponseWriter, interval time.Duration) *sseKeepalive {
	k := &sseKeepalive{w: w, stop: make(chan struct{}), done: make(chan struct{})}
	if interval <= 0 {
		close(k.done)
		return k
	}
	go func() {
		defer close(k.done)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-k.stop:
				return
			case <-t.C:
				if err := k.writeComment(); err != nil {
					return
				}
			}
		}
	}()
	return k
}

func (k *sseKeepalive) writeComment() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if _, err := io.WriteString(k.w, ": keepalive\n\n"); err != nil {
		return err
	}
	if fl, ok := k.w.(http.Flusher); ok {
		fl.Flush()
	}
	return nil
}

// Header / WriteHeader 透传底层，使本类型满足 http.ResponseWriter。
func (k *sseKeepalive) Header() http.Header { return k.w.Header() }

func (k *sseKeepalive) WriteHeader(status int) { k.w.WriteHeader(status) }

func (k *sseKeepalive) Write(p []byte) (int, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.w.Write(p)
}

// Flush 实现 http.Flusher：与心跳写互斥，避免与心跳交错。
func (k *sseKeepalive) Flush() {
	k.mu.Lock()
	defer k.mu.Unlock()
	if fl, ok := k.w.(http.Flusher); ok {
		fl.Flush()
	}
}

// Close 停掉后台 goroutine 并等待其退出：返回后保证不再有新的心跳写。
func (k *sseKeepalive) Close() {
	k.once.Do(func() { close(k.stop) })
	<-k.done
}
