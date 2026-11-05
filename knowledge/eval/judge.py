"""LLM 判官：算 faithfulness 与标准要点覆盖率。

- faithfulness 用 RAGAS 口径：把答案拆成原子陈述，逐条判断是否被检索上下文支持；
- 要点覆盖率按人工标注的 key_points 判断：是否被上下文覆盖（context_recall）、
  是否在答案里出现且有上下文支撑（answer_coverage）。
- 所有判定结果按 (模型 + prompt 版本 + 输入哈希) 落盘缓存：同样输入重跑 100% 复现，
  不会因为模型抖动让表格数字漂移。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "judge"
PROMPT_VERSION = "judge-v2"

# 判官只看最终上下文的前 N 篇、每篇截断 M 字：
# 全量上下文会让单次判定从 ~15s 涨到几分钟，而判定质量几乎不变（要点都集中在 top4）。
MAX_CONTEXTS = 4
MAX_CONTEXT_CHARS = 700

_SYSTEM_PROMPT = (
    "你是一个严格的中文评测判官。只依据给定的【上下文】判断，不要使用你自己的常识补充事实。"
    "必须输出 JSON，不要输出任何解释性文字。"
)

_USER_TEMPLATE = """请按下面步骤评测一个问答系统的输出。

【用户问题】
{query}

【检索到的上下文】
{context}

【系统答案】
{answer}

【人工标注的标准要点】
{key_points}

要求：
1. 把【系统答案】拆成若干条原子陈述（一条只讲一个事实，疑问/寒暄不算陈述）。
   对每条陈述判断是否能在【上下文】中找到直接支持，supported=true/false。
   如果答案为空，claims 返回空数组。
2. 对每条【人工标注的标准要点】判断：
   - in_context：该要点是否能在【上下文】中找到依据；
   - in_answer：该要点是否在【系统答案】中被明确说出来（且被上下文支持）。
3. 只输出如下 JSON：
{{"claims": [{{"text": "陈述原文", "supported": true}}], "key_points": [{{"point": "要点原文", "in_context": true, "in_answer": false}}]}}
"""


class JudgeUnavailable(Exception):
    """判官不可用（没配模型 / 调用失败），此时 faithfulness 记为 n/a。"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _extract_json(raw: str) -> Dict[str, Any]:
    """从模型输出里抠出 JSON（容忍 ```json 围栏和前后废话）。"""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"判官输出不是 JSON: {text[:120]}")
    return json.loads(text[start:end + 1])


class LLMJudge:
    def __init__(self, model: Optional[str] = None, use_cache: bool = True,
                 cache_dir: Path = CACHE_DIR, timeout: float = 90.0):
        self.model = model or os.getenv("LLM_DEFAULT_MODEL", "")
        if not self.model:
            raise JudgeUnavailable("未配置 LLM_DEFAULT_MODEL，无法启用 LLM 判官")
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self._client = None
        self.call_count = 0
        self.cache_hits = 0

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")
            if not api_key or not base_url:
                raise JudgeUnavailable("缺少 OPENAI_API_KEY / OPENAI_API_BASE")
            self._client = OpenAI(api_key=api_key, base_url=base_url,
                                  timeout=self.timeout, max_retries=2)
        return self._client

    def _call(self, prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:
            # 有些兼容端点不支持 response_format，退化成普通调用
            resp = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0,
            )
        content = (resp.choices[0].message.content or "")
        self.call_count += 1
        return _extract_json(content)

    # ------------------------------------------------------------------
    def judge(self, query: str, answer: str, contexts: List[Dict[str, Any]],
              key_points: List[str]) -> Dict[str, Any]:
        context_text = "\n\n".join(
            f"[{i}] {(c.get('content') or '')[:MAX_CONTEXT_CHARS]}"
            for i, c in enumerate(contexts[:MAX_CONTEXTS], 1)
        ) or "（无检索结果）"
        point_text = "\n".join(f"- {p}" for p in key_points) or "（无标注要点）"

        prompt = _USER_TEMPLATE.format(
            query=query,
            context=context_text,
            answer=answer or "（空）",
            key_points=point_text,
        )

        key = _sha("|".join([
            PROMPT_VERSION, self.model, query, answer or "",
            ",".join(str(c.get("chunk_id")) for c in contexts), point_text,
        ]))
        cache_path = self.cache_dir / f"{key}.json"
        if self.use_cache and cache_path.exists():
            self.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))

        try:
            result = self._call(prompt)
        except JudgeUnavailable:
            raise
        except Exception as e:
            raise JudgeUnavailable(f"判官调用失败: {type(e).__name__}: {e}") from e

        claims = [
            {"text": str(c.get("text", ""))[:400], "supported": bool(c.get("supported"))}
            for c in (result.get("claims") or []) if isinstance(c, dict)
        ]
        points = [
            {"point": str(p.get("point", ""))[:200],
             "in_context": bool(p.get("in_context")),
             "in_answer": bool(p.get("in_answer"))}
            for p in (result.get("key_points") or []) if isinstance(p, dict)
        ]
        out = {"claims": claims, "key_points": points, "model": self.model,
               "prompt_version": PROMPT_VERSION}

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
