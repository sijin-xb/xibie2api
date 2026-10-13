#!/usr/bin/env bash
# enable-zcode-websearch.sh —— 把 ZCode 的 workbuddy provider 切到 anthropic-messages
# 并给其模型打开 supportsNativeWebSearch，使内置 WebSearch（provider-native 服务端
# 工具）可用。配合网关的 /v1/messages web_search 支持。
#
# 为什么需要：
#   - ZCode 的 WebSearch 是 provider-native 工具，只在 provider kind = anthropic
#     （api.type=anthropic-messages）或 openai（openai-responses）时才挂载；
#     openai-chat-completions 永远拿不到。
#   - 还需要模型属性 supportsNativeWebSearch=true。
#
# 幂等：可重复运行。改完需**重启 ZCode** 才生效。
set -euo pipefail

CFG="${1:-$HOME/.zcode/v2/provider_config.json}"
[ -f "$CFG" ] || { echo "找不到 $CFG" >&2; exit 1; }

cp -p "$CFG" "$CFG.bak-websearch-$(date +%Y%m%d-%H%M%S)"

python3 - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
cfg = d["config"]
pr = cfg["providerConfigRules"]["providerRules"][0]
pr["config"]["api"]["type"] = "anthropic-messages"
ids = pr["config"].get("personalModelIds", [])
cfg.setdefault("modelConfigRules", {})
cfg["modelConfigRules"]["providerModelRules"] = [
    {"providerId": pr["providerId"], "modelId": m,
     "config": {"enabled": True, "properties": {"supportsNativeWebSearch": True}}}
    for m in ids
]
cfg["modelConfigRules"].setdefault("manualProviderModelRules", [])
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print(f"api.type={pr['config']['api']['type']}; 已为 {len(ids)} 个模型打开 supportsNativeWebSearch")
PY

echo "完成。请重启 ZCode 生效。还原：cp <上面打印的 .bak-websearch-*> $CFG"
