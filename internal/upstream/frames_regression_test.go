// frames_regression_test.go 是 SSE 帧管线与 DSML 工具调用修复的**回归安全网**（仅测试）。
//
// 目的：为「把 PumpFrames 从 StreamHint 抽出」的重构（internal/upstream/sse.go）提供
// 逐字节行为基线。所有断言都钉死在当前实现的**精确输出字节/JSON**上（而非「无错误」或
// 「函数被调用」），任何字段顺序、默认值、id 续传、name 收敛、标记修复语义、usage 别名
// 规整的变化都会被检出。
//
// 约束：本文件不触碰任何生产代码；不重复 sse_test.go / dsml_test.go / tool_pairing_test.go
// 已有的断言，只在其之上补足「逐字节」口径。夹具复用同包既有构造器
// （upstreamSSEWithContent / mkBlock / mkInvoke / mkParam / testTools / viewSSE ...）。
//
// 离线不可覆盖处：标记修复合成的 tool_call id 由 time.Now().UnixNano() 生成
// （dsml.go markupCallID），非确定，故含合成 id 的帧先经 scrubCallIDs 归一后再逐字节比对。
package upstream

import (
	"encoding/json"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
)

// dataFrame 拼一个 SSE 数据帧（与实现自产的帧格式一致）。
func dataFrame(payload string) string { return "data: " + payload + "\n\n" }

// runStreamRaw 经 Stream/StreamHint 处理 raw，返回写出的完整字节与错误。
// 无 opts 时走 Stream（= StreamHint(nil)），有 opts 时走 StreamHint。
func runStreamRaw(t *testing.T, raw string, opts ...StreamOption) (string, error) {
	t.Helper()
	rec := httptest.NewRecorder()
	var err error
	if len(opts) == 0 {
		err = Stream(rec, strings.NewReader(raw))
	} else {
		err = StreamHint(rec, strings.NewReader(raw), nil, opts...)
	}
	return rec.Body.String(), err
}

// framePayloads 取出响应体里所有数据帧的 payload（跳过 [DONE] 与非 data 行）。
func framePayloads(t *testing.T, body string) []string {
	t.Helper()
	var out []string
	for _, ln := range strings.Split(body, "\n") {
		ln = strings.TrimSpace(ln)
		if !strings.HasPrefix(ln, "data: ") {
			continue
		}
		p := strings.TrimPrefix(ln, "data: ")
		if p == "[DONE]" {
			continue
		}
		out = append(out, p)
	}
	return out
}

// synthCallIDRe 匹配标记修复合成的 tool_call id：call_<nano>_<idx>（dsml.go markupCallID）。
// 只匹配两段纯数字，不会误伤上游夹具里的 call_a / call_1。
var synthCallIDRe = regexp.MustCompile(`call_[0-9]+_[0-9]+`)

// scrubCallIDs 把非确定的合成 tool_call id 归一为 call_ID，便于逐字节比对。
func scrubCallIDs(s string) string { return synthCallIDRe.ReplaceAllString(s, "call_ID") }

// --- 1. SSE 帧规范化 + 结束事件，逐字节 --------------------------------------

// TestFramesStreamByteExactNormalization 固定上游流（多 data 帧 + 非 data 注释行 +
// [DONE]）经 Stream 后逐字节断言：帧顺序、字段白名单、id 续传、finish_reason 空串→null、
// usage 缺失→null、恰好一个 [DONE]。
//
// 注意当前实现的行为：非 data 且非空的行（此处为 SSE 注释 `: keepalive`）**原样透传**
// （sse.go readLoop 的 `trimmed != ""` 分支），并非丢弃；被吞掉的是帧分隔用的空行。
func TestFramesStreamByteExactNormalization(t *testing.T) {
	raw := dataFrame(`{"id":"up-1","object":"chat.completion.chunk","created":111,"model":"m1","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"}}]}`) +
		": keepalive\n\n" +
		dataFrame(`{"id":"up-1","choices":[{"index":0,"delta":{"content":"，世界"}}]}`) +
		dataFrame(`{"id":"up-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`) +
		dataFrame("[DONE]")

	body, err := runStreamRaw(t, raw)
	if err != nil {
		t.Fatal(err)
	}
	want := dataFrame(`{"choices":[{"delta":{"content":"你好","role":"assistant"},"finish_reason":null,"index":0}],"created":111,"id":"up-1","model":"m1","object":"chat.completion.chunk","usage":null}`) +
		": keepalive\n" +
		dataFrame(`{"choices":[{"delta":{"content":"，世界"},"finish_reason":null,"index":0}],"id":"up-1","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{},"finish_reason":"stop","index":0}],"id":"up-1","object":"chat.completion.chunk","usage":{"completion_tokens":2,"prompt_tokens":5,"total_tokens":7}}`) +
		dataFrame("[DONE]")
	if body != want {
		t.Errorf("Stream 输出非逐字节预期\n got: %q\nwant: %q", body, want)
	}
}

// TestFramesStreamEmptyAndNoDoneFallbackByteExact 覆盖：
//   - 无 [DONE] 的 EOF 兜底：有效帧原样透出 + 恰好补一个 [DONE]，err=nil；
//   - 空流（0 有效帧）：写本地 error 帧 + [DONE]，返回 errEmptyStream。
func TestFramesStreamEmptyAndNoDoneFallbackByteExact(t *testing.T) {
	// 无 DONE：上游 EOF，网关补 [DONE]。
	rawNoDone := dataFrame(`{"id":"x1","choices":[{"index":0,"delta":{"content":"hi"}}]}`)
	body, err := runStreamRaw(t, rawNoDone)
	if err != nil {
		t.Fatalf("无 DONE 的正常流不应报错: %v", err)
	}
	wantNoDone := dataFrame(`{"choices":[{"delta":{"content":"hi"},"finish_reason":null,"index":0}],"id":"x1","object":"chat.completion.chunk","usage":null}`) +
		dataFrame("[DONE]")
	if body != wantNoDone {
		t.Errorf("无 DONE 兜底非逐字节预期\n got: %q\nwant: %q", body, wantNoDone)
	}

	// 空流兜底：本地 error 帧（error 字段不被白名单剥离）+ [DONE]，且返回空流哨兵。
	// 注释行在当前实现里先被原样透传（sse.go `trimmed != ""` 分支），再写空流兜底。
	const emptyErrPayload = `{"error":{"message":"empty upstream stream","type":"upstream_error","code":"upstream_parse"}}`
	errFrame := dataFrame(emptyErrPayload)
	for _, c := range []struct {
		name string
		raw  string
		want string
	}{
		{"空输入", "", errFrame + dataFrame("[DONE]")},
		{"只有注释行", ": comment\n\n", ": comment\n" + errFrame + dataFrame("[DONE]")},
		{"只有 DONE", dataFrame("[DONE]"), errFrame + dataFrame("[DONE]")},
	} {
		t.Run(c.name, func(t *testing.T) {
			body, err := runStreamRaw(t, c.raw)
			if err == nil {
				t.Fatalf("空流必须返回错误")
			}
			if !IsEmptyStreamError(err) {
				t.Errorf("IsEmptyStreamError(err)=false，err=%v", err)
			}
			if body != c.want {
				t.Errorf("空流兜底非逐字节预期\n got: %q\nwant: %q", body, c.want)
			}
		})
	}
}

// TestFramesStreamGarbageAfterDoneByteExact [DONE] 之后的一切（垃圾帧、注释、空行）
// 一律不再透传；整条输出恰好一个 [DONE]。逐字节断言。
func TestFramesStreamGarbageAfterDoneByteExact(t *testing.T) {
	raw := dataFrame(`{"id":"x1","choices":[{"index":0,"delta":{"content":"hello"}}]}`) +
		dataFrame("[DONE]") +
		": trailing comment\n\n" +
		dataFrame(`{"should":"not appear"}`) +
		dataFrame("[DONE]")

	body, err := runStreamRaw(t, raw)
	if err != nil {
		t.Fatal(err)
	}
	want := dataFrame(`{"choices":[{"delta":{"content":"hello"},"finish_reason":null,"index":0}],"id":"x1","object":"chat.completion.chunk","usage":null}`) +
		dataFrame("[DONE]")
	if body != want {
		t.Errorf("DONE 之后内容泄漏\n got: %q\nwant: %q", body, want)
	}
	if n := strings.Count(body, "data: [DONE]"); n != 1 {
		t.Errorf("[DONE] count=%d want 1", n)
	}
}

// --- 2. 首帧 id 透传 + 无 id 哨兵，逐字节 -----------------------------------

// TestFramesFirstIDAndSentinelByteExact 首帧真实 id 续传到所有缺 id/空 id 帧；
// 全流无任何真实 id 时才出现哨兵 chatcmpl-wb2api。逐字节断言两种情形。
func TestFramesFirstIDAndSentinelByteExact(t *testing.T) {
	raw := dataFrame(`{"id":"chatcmpl-up-7","object":"chat.completion.chunk","created":5,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"A"}}]}`) +
		dataFrame(`{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"B"}}]}`) +
		dataFrame(`{"id":"","choices":[{"index":0,"delta":{"content":"C"}}]}`) +
		dataFrame("[DONE]")
	body, err := runStreamRaw(t, raw)
	if err != nil {
		t.Fatal(err)
	}
	want := dataFrame(`{"choices":[{"delta":{"content":"A","role":"assistant"},"finish_reason":null,"index":0}],"created":5,"id":"chatcmpl-up-7","model":"m","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{"content":"B"},"finish_reason":null,"index":0}],"id":"chatcmpl-up-7","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{"content":"C"},"finish_reason":null,"index":0}],"id":"chatcmpl-up-7","object":"chat.completion.chunk","usage":null}`) +
		dataFrame("[DONE]")
	if body != want {
		t.Errorf("首帧 id 续传非逐字节预期\n got: %q\nwant: %q", body, want)
	}

	// 全流无真实 id → 哨兵。
	rawNoID := dataFrame(`{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}`) +
		dataFrame(`{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`) +
		dataFrame("[DONE]")
	body2, err := runStreamRaw(t, rawNoID)
	if err != nil {
		t.Fatal(err)
	}
	want2 := dataFrame(`{"choices":[{"delta":{"content":"hi"},"finish_reason":null,"index":0}],"id":"chatcmpl-wb2api","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{},"finish_reason":"stop","index":0}],"id":"chatcmpl-wb2api","object":"chat.completion.chunk","usage":null}`) +
		dataFrame("[DONE]")
	if body2 != want2 {
		t.Errorf("无 id 哨兵非逐字节预期\n got: %q\nwant: %q", body2, want2)
	}
}

// --- 3. tool_call 跨帧增量合并，逐字节 ---------------------------------------

// TestFramesAggregateToolCallMergeByteExact 两个并行 tool_call（index 0/1）交错到达、
// 各自 function.arguments 分片：聚合后 name 各出现一次、arguments 精确拼接。
// 对整个聚合响应 json.Marshal 逐字节断言。
func TestFramesAggregateToolCallMergeByteExact(t *testing.T) {
	raw := dataFrame(`{"id":"c1","model":"m","created":1,"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":"}}]}}]}`) +
		dataFrame(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"write_stdin","arguments":"{\"session\":"}}]}}]}`) +
		dataFrame(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls -la\"}"}}]}}]}`) +
		dataFrame(`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"s1\"}"}}]}}]}`) +
		dataFrame(`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"total_tokens":9}}`) +
		dataFrame("[DONE]")

	resp, err := Aggregate(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"cmd\":\"ls -la\"}","name":"exec_command"},"id":"call_a","index":0,"type":"function"},{"function":{"arguments":"{\"session\":\"s1\"}","name":"write_stdin"},"id":"call_b","index":1,"type":"function"}]}}],"created":1,"id":"c1","model":"m","object":"chat.completion","usage":{"total_tokens":9}}`
	if string(got) != want {
		t.Errorf("聚合 tool_calls 非逐字节预期\n got: %s\nwant: %s", got, want)
	}
}

// --- 4. DSML 修复跨 chunk 边界，逐字节 --------------------------------------

// TestFramesDSMLRepairByteExactAcrossBoundary 标记块在**起始标记内部**被切成两片
// （rune 边界），经 StreamHint + WithMarkupRepair 后逐字节断言：正文帧、被摘空 content
// 的过渡帧、还原出的 delta.tool_calls 帧、收尾 tool_calls 帧、[DONE]。
// 合成 id 非确定，先经 scrubCallIDs 归一。
func TestFramesDSMLRepairByteExactAcrossBoundary(t *testing.T) {
	content := "前言\n" + mkBlock(mkInvoke("exec_command", mkParam("cmd", "ls -la")))
	runes := []rune(content)
	// "前言\n" 占前 3 个 rune，起始标记从第 3 个 rune 起；第 5 个 rune 落在两个全角竖线之间。
	const split = 5
	if split >= len(runes) {
		t.Fatalf("夹具过短: %d runes", len(runes))
	}
	chunks := []string{string(runes[:split]), string(runes[split:])}
	raw := upstreamSSEWithContent(chunks, "stop")

	repair := NewMarkupRepair(testTools, false)
	body, err := runStreamRaw(t, raw, WithMarkupRepair(repair))
	if err != nil {
		t.Fatal(err)
	}
	if repair.Converted() != 1 {
		t.Fatalf("Converted=%d want 1", repair.Converted())
	}
	got := scrubCallIDs(body)
	want := dataFrame(`{"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{"content":"前言\n"},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"cmd\":\"ls -la\"}","name":"exec_command"},"id":"call_ID","index":0,"type":"function"}]},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame("[DONE]")
	if got != want {
		t.Errorf("DSML 修复输出非逐字节预期\n got: %q\nwant: %q", got, want)
	}
}

// TestFramesDSMLRepairArbitraryBoundaries 多个任意切分点（含起始标记内部、invoke 头内部、
// 参数内部）都必须产出同一份：正文拼接 = 去掉标记块后的原文；恰好一个还原调用（name/
// arguments 精确）；恰好一个 finish_reason: tool_calls 收尾帧，且排在调用帧之后。
func TestFramesDSMLRepairArbitraryBoundaries(t *testing.T) {
	content := "前言\n" + mkBlock(mkInvoke("exec_command", mkParam("cmd", "ls -la"))) + "\n后记"
	const wantText = "前言\n\n后记"
	wantCall := `{"function":{"arguments":"{\"cmd\":\"ls -la\"}","name":"exec_command"},"id":"call_ID","index":0,"type":"function"}`
	wantCallFrame := `{"choices":[{"delta":{"tool_calls":[` + wantCall + `]},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`
	wantFinishFrame := `{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`

	for _, split := range []int{3, 5, 9, 20, 40} {
		runes := []rune(content)
		if split >= len(runes) {
			continue
		}
		raw := upstreamSSEWithContent([]string{string(runes[:split]), string(runes[split:])}, "stop")
		repair := NewMarkupRepair(testTools, false)
		body, err := runStreamRaw(t, raw, WithMarkupRepair(repair))
		if err != nil {
			t.Fatalf("split=%d: %v", split, err)
		}
		if repair.Converted() != 1 {
			t.Fatalf("split=%d: Converted=%d want 1", split, repair.Converted())
		}

		var text strings.Builder
		callFrameCount, finishCount := 0, 0
		lastCallIdx, finishIdx := -1, -1
		for i, payload := range framePayloads(t, body) {
			scrubbed := scrubCallIDs(payload)
			if scrubbed == wantCallFrame {
				callFrameCount++
				lastCallIdx = i
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(payload), &obj); err != nil {
				t.Fatalf("split=%d: 非法帧 %q: %v", split, payload, err)
			}
			chs, _ := obj["choices"].([]any)
			if len(chs) == 0 {
				continue
			}
			ch, _ := chs[0].(map[string]any)
			if d, ok := ch["delta"].(map[string]any); ok {
				if c, _ := d["content"].(string); c != "" {
					text.WriteString(c)
				}
			}
			if fr, _ := ch["finish_reason"].(string); fr != "" {
				if fr != "tool_calls" {
					t.Fatalf("split=%d: finish_reason=%q want tool_calls", split, fr)
				}
				if scrubbed != wantFinishFrame {
					t.Fatalf("split=%d: 收尾帧非逐字节预期\n got: %q\nwant: %q", split, scrubbed, wantFinishFrame)
				}
				finishCount++
				finishIdx = i
			}
		}
		if text.String() != wantText {
			t.Errorf("split=%d: 正文拼接=%q want %q", split, text.String(), wantText)
		}
		if callFrameCount != 1 {
			t.Errorf("split=%d: 调用帧数=%d want 1", split, callFrameCount)
		}
		if finishCount != 1 {
			t.Errorf("split=%d: 收尾帧数=%d want 1", split, finishCount)
		}
		if finishIdx < lastCallIdx {
			t.Errorf("split=%d: 收尾帧(%d)必须排在调用帧(%d)之后", split, finishIdx, lastCallIdx)
		}
	}
}

// TestFramesDSMLUnclosedFlushedVerbatimByteExact 未闭合的块在流尾连起始标记一起原样
// 回吐（不吞字节），不产出 tool_call，finish_reason 保持上游原值。整条输出逐字节断言
// （本用例无合成 id，完全确定）。
func TestFramesDSMLUnclosedFlushedVerbatimByteExact(t *testing.T) {
	tail := markupOpen + "\n" + mkInvoke("exec_command", mkParam("cmd", "pwd"))
	content := "说明\n" + tail
	raw := upstreamSSEWithContent([]string{content}, "stop")

	repair := NewMarkupRepair(testTools, false)
	body, err := runStreamRaw(t, raw, WithMarkupRepair(repair))
	if err != nil {
		t.Fatal(err)
	}
	if repair.Converted() != 0 {
		t.Fatalf("未闭合块不得转换，Converted=%d", repair.Converted())
	}
	tailJSON, err := json.Marshal(tail)
	if err != nil {
		t.Fatal(err)
	}
	want := dataFrame(`{"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{"content":"说明\n"},"finish_reason":null,"index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":1,"id":"resp-1","model":"global:deepseek-v4.1-flash","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(`{"choices":[{"delta":{"content":`+string(tailJSON)+`},"finish_reason":null,"index":0}],"id":"resp-1","object":"chat.completion.chunk","usage":null}`) +
		dataFrame("[DONE]")
	if body != want {
		t.Errorf("未闭合块回吐非逐字节预期\n got: %q\nwant: %q", body, want)
	}
}

// --- 5. Aggregate 非流式输出，逐字节 ----------------------------------------

// TestFramesAggregateNonStreamByteExact 聚合 content + tool_calls + usage（显式 total）
// 后，对整个 map json.Marshal 逐字节断言。
func TestFramesAggregateNonStreamByteExact(t *testing.T) {
	raw := dataFrame(`{"id":"chatcmpl-agg","object":"chat.completion.chunk","created":42,"model":"global:deepseek-v4.1-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"答案"}}]}`) +
		dataFrame(`{"id":"chatcmpl-agg","choices":[{"index":0,"delta":{"content":"是 42","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`) +
		dataFrame(`{"id":"chatcmpl-agg","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]}}]}`) +
		dataFrame(`{"id":"chatcmpl-agg","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}`) +
		dataFrame("[DONE]")

	resp, err := Aggregate(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"content":"答案是 42","role":"assistant","tool_calls":[{"function":{"arguments":"{\"q\":\"x\"}","name":"lookup"},"id":"call_1","index":0,"type":"function"}]}}],"created":42,"id":"chatcmpl-agg","model":"global:deepseek-v4.1-flash","object":"chat.completion","usage":{"completion_tokens":3,"prompt_tokens":10,"total_tokens":13}}`
	if string(got) != want {
		t.Errorf("非流式聚合非逐字节预期\n got: %s\nwant: %s", got, want)
	}
}

// TestFramesAggregateEnsuresUsageTotal 上游末帧 usage 缺 total_tokens 但 prompt/completion
// 都在 → 网关合成 total=prompt+completion。逐字节断言 usage 子对象。
func TestFramesAggregateEnsuresUsageTotal(t *testing.T) {
	raw := dataFrame(`{"id":"c1","model":"m","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`) +
		dataFrame(`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}`) +
		dataFrame("[DONE]")

	resp, err := Aggregate(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(resp["usage"])
	if err != nil {
		t.Fatal(err)
	}
	want := `{"completion_tokens":3,"prompt_tokens":10,"total_tokens":13}`
	if string(got) != want {
		t.Errorf("usage total 合成非逐字节预期\n got: %s\nwant: %s", got, want)
	}
}

// --- 6. error 帧 / 上游异常，逐字节 -----------------------------------------

// TestFramesErrorFrameObserverByteExact 上游 error 帧：观察者收到**原始 payload**（精确），
// 且该帧以原文透传（不走 normalizeFrame 白名单，error 字段存活）。逐字节断言完整输出。
// 另覆盖「只有 error 帧」的流：计为有效帧，不触发空流兜底。
func TestFramesErrorFrameObserverByteExact(t *testing.T) {
	const errPayload = `{"error":{"code":6004,"message":"限流"}}`
	raw := dataFrame(`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`) +
		dataFrame(errPayload) +
		dataFrame("[DONE]")

	var observed []string
	body, err := runStreamRaw(t, raw, WithErrorFrameObserver(func(p string) { observed = append(observed, p) }))
	if err != nil {
		t.Fatal(err)
	}
	if len(observed) != 1 || observed[0] != errPayload {
		t.Fatalf("观察者回调=%v want [%s]", observed, errPayload)
	}
	want := dataFrame(`{"choices":[{"delta":{"content":"hi"},"finish_reason":null,"index":0}],"created":1,"id":"x1","model":"m","object":"chat.completion.chunk","usage":null}`) +
		dataFrame(errPayload) +
		dataFrame("[DONE]")
	if body != want {
		t.Errorf("error 帧透传非逐字节预期\n got: %q\nwant: %q", body, want)
	}

	// 只有 error 帧：有效帧计数=1，无空流兜底帧，err=nil。
	onlyErr := dataFrame(`{"error":{"code":1,"message":"boom"}}`) + dataFrame("[DONE]")
	var observed2 []string
	body2, err := runStreamRaw(t, onlyErr, WithErrorFrameObserver(func(p string) { observed2 = append(observed2, p) }))
	if err != nil {
		t.Fatalf("仅 error 帧不应报错: %v", err)
	}
	if len(observed2) != 1 || observed2[0] != `{"error":{"code":1,"message":"boom"}}` {
		t.Fatalf("观察者回调=%v", observed2)
	}
	want2 := dataFrame(`{"error":{"code":1,"message":"boom"}}`) + dataFrame("[DONE]")
	if body2 != want2 {
		t.Errorf("仅 error 帧输出非逐字节预期\n got: %q\nwant: %q", body2, want2)
	}
}

// --- 7. usage/cache 别名（经 normalizeFrame），逐字节 ------------------------

// TestFramesNormalizeFrameCacheAliases normalizeFrame 出口必须把嵌套命中镜像到各别名
// （顶层 cache_read_input_tokens / cached_tokens / prompt_cache_hit_tokens、
// prompt_tokens_details.cached_tokens，以及已存在的 input_tokens_details.cached_tokens）。
// 与既有 TestNormalizeUsageCacheAliasesMirrorsNestedHit 互补：此处走 normalizeFrame 出口，
// 且覆盖 input_tokens_details 分支。逐字节断言 marshal 结果。
func TestFramesNormalizeFrameCacheAliases(t *testing.T) {
	in := map[string]any{
		"id": "x",
		"choices": []any{
			map[string]any{"index": 0, "delta": map[string]any{"content": "hi"}, "finish_reason": "stop"},
		},
		"usage": map[string]any{
			"prompt_tokens":           100.0,
			"completion_tokens":       10.0,
			"total_tokens":            110.0,
			"cache_read_input_tokens": 0.0,
			"cached_tokens":           0.0,
			"prompt_cache_hit_tokens": 0.0,
			"prompt_tokens_details":   map[string]any{"cached_tokens": 20864.0},
			"input_tokens_details":    map[string]any{"cached_tokens": 0.0},
		},
	}
	got, err := json.Marshal(normalizeFrame(in))
	if err != nil {
		t.Fatal(err)
	}
	want := `{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop","index":0}],"id":"x","object":"chat.completion.chunk","usage":{"cache_read_input_tokens":20864,"cached_tokens":20864,"completion_tokens":10,"input_tokens_details":{"cached_tokens":20864},"prompt_cache_hit_tokens":20864,"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":20864},"total_tokens":110}}`
	if string(got) != want {
		t.Errorf("normalizeFrame 缓存别名非逐字节预期\n got: %s\nwant: %s", got, want)
	}
}
