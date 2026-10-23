#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys


def find_rollout(thread_id):
    home = os.path.expanduser("~")
    pattern = os.path.join(home, ".codex", "sessions", "**", f"rollout-*-{thread_id}.jsonl")
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[-1] if matches else None


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def analyze(events):
    r = {"meta": {}, "counts": {}, "tools": {}, "phases": {}, "tokens": [],
         "searches": [], "messages": [], "task_events": 0, "fail_outputs": 0}
    for ev in events:
        t = ev.get("type")
        p = ev.get("payload", {})
        if t == "session_meta":
            r["meta"] = {"cli": p.get("cli_version"), "originator": p.get("originator"),
                         "started": ev.get("timestamp")}
        elif t == "turn_context":
            r["meta"]["model"] = p.get("model")
            r["meta"]["effort"] = p.get("effort")
        elif t == "event_msg":
            if p.get("type") in ("task_started", "task_complete"):
                r["task_events"] += 1
        elif t == "token_usage_record":
            u = p.get("usage", {})
            r["tokens"].append({"in": u.get("input_tokens", 0),
                                "cached": u.get("cached_input_tokens", 0),
                                "out": u.get("output_tokens", 0)})
        elif t == "response_item":
            pt = p.get("type")
            r["counts"][pt] = r["counts"].get(pt, 0) + 1
            if pt in ("function_call", "custom_tool_call"):
                name = p.get("name", "?")
                r["tools"][f"{pt}:{name}"] = r["tools"].get(f"{pt}:{name}", 0) + 1
            elif pt == "web_search_call":
                action = p.get("action") or {}
                r["searches"].append(action.get("query", "(no query)"))
            elif pt == "message":
                role = p.get("role", "?")
                text = ""
                for part in p.get("content") or []:
                    if isinstance(part, dict) and part.get("text"):
                        text = part["text"]
                        break
                r["messages"].append({"role": role, "phase": p.get("phase"), "text": text})
            elif pt == "function_call_output":
                out = p.get("output")
                if isinstance(out, str):
                    try:
                        obj = json.loads(out)
                        if isinstance(obj, dict) and obj.get("exit_code") not in (None, 0):
                            r["fail_outputs"] += 1
                    except json.JSONDecodeError:
                        pass
    return r


def token_summary(tokens):
    if not tokens:
        return "0 次上游请求"
    first, last = tokens[0], tokens[-1]
    total_in = sum(t["in"] for t in tokens)
    total_cached = sum(t["cached"] for t in tokens)
    total_out = sum(t["out"] for t in tokens)
    hit = total_cached / total_in * 100 if total_in else 0.0
    return (f"{len(tokens)} 次上游请求 | input {first['in']}→{last['in']}"
            f"（合计 {total_in:,}，缓存命中 {hit:.0f}%）| output 合计 {total_out:,}")


def verdict(r):
    problems = []
    asst = [m for m in r["messages"] if m["role"] == "assistant"]
    if asst and asst[-1]["phase"] == "commentary":
        problems.append(f"回合以过程播报收尾（phase=commentary）: {asst[-1]['text'][:60]}")
    if asst and asst[-1]["phase"] is None:
        problems.append("最后一条助手消息缺 phase（网关回放路径应补齐）")
    if r["fail_outputs"]:
        problems.append(f"{r['fail_outputs']} 次工具输出非零退出码")
    return problems


def main():
    ap = argparse.ArgumentParser(description="复盘一个 Codex 会话 thread：工具分布 / phase / token 曲线 / 搜索 / 收尾健康度")
    ap.add_argument("thread", help="codex://threads/<uuid> 或裸 uuid")
    ap.add_argument("--messages", action="store_true", help="列出全部助手消息（phase + 摘要）")
    args = ap.parse_args()

    tid = args.thread.split("/")[-1].strip()
    path = find_rollout(tid)
    if not path:
        sys.exit(f"找不到 thread {tid} 的 rollout 文件（~/.codex/sessions/**/rollout-*-{tid}.jsonl）")

    r = analyze(load_events(path))
    m = r["meta"]
    print(f"== 会话 {tid}")
    print(f"   {m.get('cli', '?')} · {m.get('originator', '?')} · 模型 {m.get('model', '?')} · effort {m.get('effort', '?')}")
    print(f"   {token_summary(r['tokens'])}")
    print(f"   task 生命周期事件 {r['task_events']} 次（2 的倍数 = 正常完成一轮）")

    print("== item 分布")
    for k, v in sorted(r["counts"].items(), key=lambda x: -x[1]):
        print(f"   {k:<24} {v}")
    if r["tools"]:
        print("== 工具调用")
        for k, v in sorted(r["tools"].items(), key=lambda x: -x[1]):
            print(f"   {k:<40} {v}")
    if r["searches"]:
        print("== 联网搜索（网关服务端 web_search）")
        for q in r["searches"]:
            print(f"   - {q[:90]}")

    phases = {}
    for msg in r["messages"]:
        if msg["role"] == "assistant":
            key = msg["phase"] if msg["phase"] is not None else "(null)"
            phases[key] = phases.get(key, 0) + 1
    if phases:
        print(f"== 助手消息 phase 分布 {phases}")
    if args.messages:
        for msg in r["messages"]:
            if msg["role"] == "assistant":
                print(f"   [{msg['phase'] or '-'}] {msg['text'][:70]}")

    problems = verdict(r)
    print("== 结论")
    if problems:
        for p in problems:
            print(f"   ⚠ {p}")
    else:
        print("   ✓ 无异常：以最终回答收尾，无失败输出")


if __name__ == "__main__":
    main()
