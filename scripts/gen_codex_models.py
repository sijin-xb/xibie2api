#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:8787/v1/models"
DEFAULT_OUTPUT = Path.home() / ".codex" / "workbuddy_models.json"
CONTEXT_WINDOW = 1_000_000
VALID_REALMS = ("cn", "global")

REASONING_DESCRIPTIONS = {
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth for everyday tasks",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra high reasoning depth for complex problems",
    "max": "Maximum reasoning depth for the hardest problems",
}

DEFAULT_REASONING_LEVELS = [
    {"effort": effort, "description": description}
    for effort, description in REASONING_DESCRIPTIONS.items()
]

CREDIT_PATTERN = re.compile(r"(?i)(?:x\s*\d+(?:\.\d+)?|free)")


class CatalogError(RuntimeError):
    """可读的目录生成错误。"""


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="从 workbuddy2api /v1/models 生成 Codex 模型目录"
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="模型列表接口 URL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录 JSON")
    parser.add_argument(
        "--config",
        type=Path,
        help="包含 api_key 或 api_keys 的 workbuddy2api 配置文件",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP 超时秒数")
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="同时在终端打印完整 unified diff（默认只打印摘要和 diff 文件路径）",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def read_json(path: Path, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise CatalogError(f"{label}不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(
            f"{label}不是合法 JSON: {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise CatalogError(f"读取{label}失败: {path}: {exc}") from exc


def config_candidates(repo_root: Path, explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())

    env_config = os.environ.get("WORKBUDDY_CONFIG", "").strip()
    if env_config:
        candidates.append(Path(env_config).expanduser())

    candidates.extend(
        [
            repo_root / "config.json",
            Path.home() / ".local" / "share" / "workbuddy2api-panel" / "config.json",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def key_from_config(path: Path) -> str:
    config = read_json(path, "配置文件")
    if not isinstance(config, dict):
        raise CatalogError(f"配置文件顶层必须是 JSON 对象: {path}")

    api_keys = config.get("api_keys")
    # 网关配置了 api_keys 后，顶层 api_key 只保留给面板/管理接口，不能再访问
    # /v1/models；因此这里必须优先选择一条启用的 api_keys 条目。
    if isinstance(api_keys, list) and api_keys:
        for entry in api_keys:
            if not isinstance(entry, dict):
                continue
            if entry.get("enabled") is False:
                continue
            key = entry.get("key")
            if isinstance(key, str) and key.strip():
                return key.strip()
        return ""

    api_key = config.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return ""


def resolve_api_key(repo_root: Path, explicit_config: Path | None) -> tuple[str, str]:
    for name in ("WORKBUDDY_API_KEY", "WB2A_API_KEY"):
        key = os.environ.get(name, "").strip()
        if key:
            return key, f"环境变量 {name}"

    for path in config_candidates(repo_root, explicit_config):
        if not path.is_file():
            if explicit_config is not None and path == explicit_config.expanduser():
                raise CatalogError(f"指定的配置文件不存在: {path}")
            continue
        key = key_from_config(path)
        if key:
            return key, str(path)

    searched = ", ".join(str(path) for path in config_candidates(repo_root, explicit_config))
    raise CatalogError(
        "未找到 API key。请设置 WORKBUDDY_API_KEY，或用 --config 指定包含 "
        f"api_key/api_keys 的配置文件。已查找: {searched}"
    )


def fetch_models(endpoint: str, api_key: str, timeout: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        raise CatalogError(
            f"请求 {endpoint} 失败: HTTP {exc.code}: {body.strip()}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CatalogError(f"请求 {endpoint} 失败: {exc.reason}") from exc
    except TimeoutError as exc:
        raise CatalogError(f"请求 {endpoint} 超时（{timeout:g}s）") from exc

    if status != 200:
        raise CatalogError(f"请求 {endpoint} 返回 HTTP {status}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{endpoint} 返回的不是合法 JSON: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise CatalogError(f"{endpoint} 响应缺少 data 数组")

    models = payload["data"]
    if not models:
        raise CatalogError(f"{endpoint} 返回的模型列表为空")
    if any(not isinstance(model, dict) for model in models):
        raise CatalogError(f"{endpoint} 的 data 数组包含非对象条目")
    return models


def validate_catalog(catalog: Any, path: Path) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        raise CatalogError(f"现有模型目录顶层必须是 JSON 对象: {path}")
    models = catalog.get("models")
    if not isinstance(models, list) or not models:
        raise CatalogError(f"现有模型目录缺少非空 models 数组: {path}")
    if any(not isinstance(model, dict) for model in models):
        raise CatalogError(f"现有模型目录的 models 数组包含非对象条目: {path}")
    if any(not isinstance(model.get("slug"), str) or not model["slug"] for model in models):
        raise CatalogError(f"现有模型目录存在空 slug: {path}")
    return catalog


def normalized_credit(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    match = CREDIT_PATTERN.search(value)
    if match is None:
        return ""
    credit = re.sub(r"\s+", "", match.group(0)).lower()
    if credit.startswith("x"):
        try:
            if float(credit[1:]) == 0:
                return "free"
        except ValueError:
            pass
    return credit


def display_name(model: dict[str, Any]) -> str:
    slug = model["id"]
    realm = slug.split(":", 1)[0]
    region = "CN" if realm == "cn" else "Global"
    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        name = slug.split(":", 1)[1]
    credit = normalized_credit(model.get("credits"))
    suffix = region if not credit else f"{region}, {credit}"
    return f"{name.strip()} ({suffix})"


def reasoning_levels(
    model: dict[str, Any], template: dict[str, Any]
) -> tuple[list[dict[str, str]], str]:
    remote = model.get("reasoning_supported_efforts")
    levels: list[dict[str, str]] = []
    if isinstance(remote, list):
        seen: set[str] = set()
        for value in remote:
            if not isinstance(value, str):
                continue
            effort = value.strip()
            if not effort or effort in seen:
                continue
            seen.add(effort)
            levels.append(
                {
                    "effort": effort,
                    "description": REASONING_DESCRIPTIONS.get(
                        effort, f"Reasoning effort: {effort}"
                    ),
                }
            )

    if not levels:
        existing = template.get("supported_reasoning_levels")
        if isinstance(existing, list) and existing:
            levels = copy.deepcopy(existing)
        else:
            levels = copy.deepcopy(DEFAULT_REASONING_LEVELS)

    efforts = {
        item.get("effort")
        for item in levels
        if isinstance(item, dict) and isinstance(item.get("effort"), str)
    }
    candidates = [
        model.get("reasoning_default_effort"),
        template.get("default_reasoning_level"),
        "medium",
        levels[0].get("effort") if isinstance(levels[0], dict) else None,
    ]
    default = next(
        (candidate for candidate in candidates if isinstance(candidate, str) and candidate in efforts),
        "medium",
    )
    return levels, default


def validate_remote_model(model: dict[str, Any], index: int) -> None:
    slug = model.get("id")
    if not isinstance(slug, str) or not slug.strip():
        raise CatalogError(f"/v1/models data[{index}] 缺少 id")
    realm, separator, bare = slug.partition(":")
    if separator != ":" or realm not in VALID_REALMS or not bare:
        raise CatalogError(
            f"/v1/models data[{index}] 的 id 必须以 cn: 或 global: 开头: {slug!r}"
        )


def build_catalog(
    existing: dict[str, Any], remote_models: list[dict[str, Any]]
) -> dict[str, Any]:
    existing_models = existing["models"]
    by_slug = {model["slug"]: model for model in existing_models}
    base_template = existing_models[0]

    seen: set[str] = set()
    generated: list[dict[str, Any]] = []
    for index, remote in enumerate(remote_models):
        validate_remote_model(remote, index)
        slug = remote["id"].strip()
        if slug in seen:
            raise CatalogError(f"/v1/models 返回重复模型 id: {slug}")
        seen.add(slug)

        template = by_slug.get(slug, base_template)
        entry = copy.deepcopy(template)
        name = display_name(remote)
        levels, default_reasoning = reasoning_levels(remote, template)

        entry["slug"] = slug
        entry["display_name"] = name
        entry["description"] = name
        entry["default_reasoning_level"] = default_reasoning
        entry["supported_reasoning_levels"] = levels
        entry["context_window"] = CONTEXT_WINDOW
        entry["priority"] = index + 1
        generated.append(entry)

    result = copy.deepcopy(existing)
    result["models"] = generated
    return result


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def backup_path(output: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    candidate = output.with_name(f"{output.name}.bak.{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = output.with_name(f"{output.name}.bak.{stamp}.{suffix}")
        suffix += 1
    return candidate


def diff_path(output: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    candidate = output.with_name(f"{output.name}.diff.{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = output.with_name(f"{output.name}.diff.{stamp}.{suffix}")
        suffix += 1
    return candidate


def atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def changed_fields(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    fields = sorted(set(old) | set(new))
    return [field for field in fields if old.get(field) != new.get(field)]


def summarize_changes(
    old_catalog: dict[str, Any], new_catalog: dict[str, Any]
) -> tuple[list[str], list[str], list[tuple[str, list[str]]]]:
    old_by_slug = {model["slug"]: model for model in old_catalog["models"]}
    new_by_slug = {model["slug"]: model for model in new_catalog["models"]}
    old_slugs = set(old_by_slug)
    new_slugs = set(new_by_slug)
    added = [model["slug"] for model in new_catalog["models"] if model["slug"] not in old_slugs]
    removed = [model["slug"] for model in old_catalog["models"] if model["slug"] not in new_slugs]
    changed: list[tuple[str, list[str]]] = []
    for model in new_catalog["models"]:
        slug = model["slug"]
        if slug not in old_by_slug:
            continue
        fields = changed_fields(old_by_slug[slug], model)
        if fields:
            changed.append((slug, fields))
    return added, removed, changed


def print_summary(
    old_catalog: dict[str, Any],
    new_catalog: dict[str, Any],
    added: list[str],
    removed: list[str],
    changed: list[tuple[str, list[str]]],
) -> None:
    print(
        f"模型数量: {len(old_catalog['models'])} -> {len(new_catalog['models'])}"
    )
    print(f"新增: {len(added)}" + (f" ({', '.join(added)})" if added else ""))
    print(f"移除: {len(removed)}" + (f" ({', '.join(removed)})" if removed else ""))
    print(f"字段变更: {len(changed)} 个已有模型")
    for slug, fields in changed:
        print(f"  {slug}: {', '.join(fields)}")


def verify_generated(
    path: Path, remote_models: list[dict[str, Any]], expected_slugs: list[str]
) -> None:
    catalog = validate_catalog(read_json(path, "生成后的模型目录"), path)
    models = catalog["models"]
    if len(models) != len(remote_models):
        raise CatalogError(
            f"生成后模型数量 {len(models)} 与 /v1/models 返回数量 "
            f"{len(remote_models)} 不一致"
        )

    slugs = [model["slug"] for model in models]
    if slugs != expected_slugs:
        raise CatalogError("生成后的模型 slug 顺序与 /v1/models 不一致")
    if len(set(slugs)) != len(slugs):
        raise CatalogError("生成后的模型 slug 存在重复")
    if any(model.get("context_window") != CONTEXT_WINDOW for model in models):
        raise CatalogError(f"生成后的模型 context_window 未统一为 {CONTEXT_WINDOW}")


def main() -> int:
    args = parse_args()
    output = args.output.expanduser()
    repo_root = args.repo_root.expanduser()

    existing_text = ""
    try:
        existing_text = output.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CatalogError(f"现有模型目录不存在: {output}") from exc
    except OSError as exc:
        raise CatalogError(f"读取现有模型目录失败: {output}: {exc}") from exc

    try:
        existing = validate_catalog(json.loads(existing_text), output)
    except json.JSONDecodeError as exc:
        raise CatalogError(
            f"现有模型目录不是合法 JSON: {output}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc

    api_key, key_source = resolve_api_key(repo_root, args.config)
    print(f"请求模型目录: {args.endpoint}")
    print(f"API key 来源: {key_source}")
    remote_models = fetch_models(args.endpoint, api_key, args.timeout)
    new_catalog = build_catalog(existing, remote_models)
    new_text = json_text(new_catalog)

    old_stat = output.stat()
    backup = backup_path(output)
    try:
        shutil.copy2(output, backup)
    except OSError as exc:
        raise CatalogError(f"备份旧目录失败: {backup}: {exc}") from exc
    print(f"旧目录备份: {backup}")

    unified = list(
        difflib.unified_diff(
            existing_text.splitlines(),
            new_text.splitlines(),
            fromfile=str(output),
            tofile=f"{output} (generated)",
            lineterm="",
        )
    )
    diff_file = diff_path(output)
    atomic_write(diff_file, "\n".join(unified) + ("\n" if unified else ""))
    print(f"diff 文件: {diff_file}")

    added, removed, changed = summarize_changes(existing, new_catalog)
    print_summary(existing, new_catalog, added, removed, changed)
    if args.show_diff and unified:
        print("\n".join(unified))

    atomic_write(output, new_text, mode=old_stat.st_mode & 0o777)
    expected_slugs = [model["id"].strip() for model in remote_models]
    verify_generated(output, remote_models, expected_slugs)

    print(
        f"生成完成: {output} | 模型 {len(remote_models)} | "
        f"context_window={CONTEXT_WINDOW} | JSON 校验通过"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CatalogError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
