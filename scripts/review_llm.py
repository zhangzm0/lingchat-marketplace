#!/usr/bin/env python3
"""LLM 语义审查：机器检查通过后，对代码/内容做语义级判断。

用法：
    python3 scripts/review_llm.py registry/ machine_report.json [--json-out review.json]

环境变量：
    LLM_API_KEY     必填。OpenAI 兼容 API Key（默认智谱开放平台；魔搭 ModelScope 也可用）
    LLM_BASE_URL    默认 https://open.bigmodel.cn/api/paas/v4
    LLM_MODEL       默认 glm-4.7-flash（智谱免费模型，200K 上下文）

退出码：
    0 = verdict=approve / changes（交给门禁决策）
    1 = verdict=reject
    2 = 无法判定（API 失败/输出解析失败）→ fail-closed，PR 挂起等人工
"""

import argparse
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-4.7-flash"

SYSTEM_PROMPT = """你是开源软件市场的代码与内容审核员。你的职责是判断提交上架的插件/内容是否存在以下问题：

1. 恶意意图：代码行为与描述不符，窃取用户数据、破坏数据、隐藏后门
2. 数据外泄：把用户数据（聊天记录、记忆、配置）发送到网络
3. 权限越界：代码实际行为超出 manifest 声明的权限（network 白名单、permissions.tools）
4. 工具承诺欺骗：工具 description 声称的功能与实际代码行为不一致
5. 内容合规：违规、成人、版权侵权、垃圾广告、政治敏感内容

注意：机器检查已处理确定性规则（禁用模块、URL 白名单、大文件等），你只负责语义判断，不要重复机器检查的工作。

输出必须是合法 JSON（不要 markdown 代码块，不要注释），格式：
{
  "verdict": "approve" | "changes" | "reject",
  "risk_level": "low" | "medium" | "high",
  "summary": "一句话结论",
  "findings": [
    {
      "severity": "error" | "warn" | "info",
      "file": "文件名",
      "line": 行号或0,
      "category": "malicious" | "privacy" | "permission" | "policy" | "quality",
      "detail": "具体说明"
    }
  ]
}

verdict 含义：
- approve：可以上架
- changes：有小问题需要作者修改（不致命）
- reject：存在恶意或严重违规，禁止上架"""


def load_material(registry_dir: Path, only: set[str] | None = None) -> str:
    """收集 manifest + 全部源码文本，供 LLM 审查。"""
    parts = []
    for pkg in sorted(registry_dir.iterdir()):
        if not pkg.is_dir():
            continue
        if only is not None and pkg.name not in only:
            continue
        mf = pkg / "manifest.toml"
        if mf.exists():
            parts.append(f"=== {pkg.name}/manifest.toml ===\n{mf.read_text(encoding='utf-8', errors='replace')}")
        for f in sorted(pkg.rglob("*")):
            if not f.is_file() or f.name == "manifest.toml":
                continue
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".onnx", ".bin", ".model"}:
                parts.append(f"=== {f.relative_to(registry_dir)} ===\n[二进制资源，大小 {f.stat().st_size} 字节，跳过内容审查]")
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > 200_000:
                text = text[:200_000] + "\n...[截断]"
            parts.append(f"=== {f.relative_to(registry_dir)} ===\n{text}")
    return "\n\n".join(parts)


def call_llm(material: str, machine_report: dict, base_url: str, api_key: str, model: str) -> str | None:
    """调用 OpenAI 兼容 chat completions，返回响应文本；失败返回 None。

    带指数退避重试（最多 3 次）：对抗免费模型（如智谱 glm-4.7-flash）
    的共享速率限制（HTTP 429 / code 1302）。4xx 认证错误不重试。
    """
    import time as _time

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"# 机器检查报告\n{json.dumps(machine_report, ensure_ascii=False)}\n\n# 提交内容\n{material}\n\n请输出审核 JSON。"},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        # 魔搭/部分厂商默认开启思考模式（输出进 reasoning_content，content 为空），显式关闭
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }

    def attempt() -> tuple[bool, str | None]:
        """返回 (是否终结, 文本)。终结=拿到文本或不可重试错误。"""
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                print(f"[review_llm] 空 choices（可能思考模式被拒）: {str(body)[:200]}", file=sys.stderr)
                return False, None
            msg = choices[0].get("message", {})
            text = msg.get("content") or msg.get("reasoning_content") or ""
            return True, text
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 429 or "1302" in detail or e.code >= 500:
                print(f"[review_llm] 限流/服务端错误 {e.code}（第 {attempt.n} 次）: {detail[:150]}", file=sys.stderr)
                return False, None
            print(f"[review_llm] HTTP {e.code} 不可重试: {detail[:150]}", file=sys.stderr)
            return True, None
        except Exception as e:
            print(f"[review_llm] LLM 调用失败: {e}", file=sys.stderr)
            return False, None

    attempt.n = 0
    for delay in (5, 20, 60):  # 指数退避：5s → 20s → 60s
        attempt.n += 1
        done, text = attempt()
        if done:
            return text
        _time.sleep(delay)
    return None


def parse_verdict(text: str) -> dict | None:
    """解析并校验 LLM 输出 JSON；非法返回 None。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 容错：剥离可能的 markdown 代码块围栏
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if data.get("verdict") not in ("approve", "changes", "reject"):
        return None
    if data.get("risk_level") not in ("low", "medium", "high"):
        return None
    if not isinstance(data.get("findings"), list):
        data["findings"] = []
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 语义审查")
    ap.add_argument("registry_dir")
    ap.add_argument("machine_report", help="check_pr.py 输出的机器检查报告 JSON")
    ap.add_argument("--pkgs", default=None, help="只审查这些包目录名（空格分隔），缺省全量")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        print(json.dumps({"verdict": "pending", "reason": "未配置 LLM_API_KEY（fail-closed）"}, ensure_ascii=False))
        return 2

    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    try:
        machine_report = json.loads(Path(args.machine_report).read_text(encoding="utf-8"))
    except Exception:
        machine_report = {"verdict": "unknown"}

    material = load_material(Path(args.registry_dir), set(args.pkgs.split()) if args.pkgs else None)

    result = None
    for attempt in (1, 2):  # 重试一次
        text = call_llm(material, machine_report, base_url, api_key, model)
        if text is None:
            continue
        result = parse_verdict(text)
        if result is not None:
            break
        print(f"[review_llm] 第 {attempt} 次输出解析失败", file=sys.stderr)

    if result is None:
        result = {
            "verdict": "pending",
            "risk_level": "unknown",
            "summary": "LLM 审查无法完成（API 失败或输出不可解析）——fail-closed，挂起等人工",
            "findings": [],
        }
        rc = 2
    else:
        rc = {"approve": 0, "changes": 0, "reject": 1}.get(result["verdict"], 2)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
