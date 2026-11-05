"""标注集加载 / 校验 / gold 文本解析。

标注文件（jsonl，一行一条 query）字段：
    id            必填，唯一，如 "Q01"
    query         必填，用户原问题
    item_names    可选，预置商品名。填了会跳过「商品名澄清节点」，用于隔离检索质量
    rewritten_query 可选，预置检索用 query；不填则走 LLM 改写
    gold          必填，标准答案 chunk 列表，每项：
                    {"chunk_id": 470252164696000478, "grade": 2}
                  或（推荐给人工标注用，避免抄 id）：
                    {"snippet": "将功能转盘置于V DC的位置", "grade": 2}
                  grade 缺省为 1，2 表示核心 chunk、1 表示支撑 chunk
    key_points    必填，标准答案要点（字符串列表），用于算要点覆盖率

snippet 会在 chunk 快照里做「归一化包含」匹配，命中唯一则回填 chunk_id，
匹配不唯一/匹配不到会直接报错并列出候选，避免标注错误被静默吞掉。
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DatasetError(Exception):
    """标注集不合法。"""


@dataclass
class GoldItem:
    chunk_id: Optional[int] = None
    snippet: Optional[str] = None
    grade: int = 1


@dataclass
class QueryItem:
    id: str
    query: str
    key_points: List[str]
    gold: List[GoldItem] = field(default_factory=list)
    item_names: Optional[List[str]] = None
    rewritten_query: Optional[str] = None

    @property
    def gold_ids(self) -> List[int]:
        return [g.chunk_id for g in self.gold if g.chunk_id is not None]

    @property
    def gold_grades(self) -> Dict[int, int]:
        return {g.chunk_id: g.grade for g in self.gold if g.chunk_id is not None}


# ----------------------------------------------------------------------
# 文本归一化
# ----------------------------------------------------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[，。、；：！？“”‘’（）《》【】—…·,.;:!?\"'()\[\]<>]")


def normalize(text: str) -> str:
    """去空白 + 去中英标点，用于 snippet 匹配（标注时常漏标点）。"""
    text = _PUNCT.sub("", text or "")
    return _WS.sub("", text).lower()


# ----------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise DatasetError(f"标注文件不存在: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path.name} 第 {lineno} 行不是合法 JSON: {e}") from e
    if not rows:
        raise DatasetError(f"{path} 没有任何标注数据")
    return rows


def load_snapshot(path: Path) -> List[Dict[str, Any]]:
    """chunk 快照（由 export_chunks.py 生成）。"""
    if not path.exists():
        raise DatasetError(
            f"chunk 快照不存在: {path}\n先跑: python -m knowledge.eval.export_chunks")
    return load_jsonl(path)


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def _match_snippet(snippet: str, snapshot: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    """把 snippet 解析成 chunk_id。返回 (chunk_id, 说明)。"""
    needle = normalize(snippet)
    if not needle:
        return None, "snippet 为空"

    hits = [row for row in snapshot if needle in normalize(row.get("content", ""))]
    if len(hits) == 1:
        return int(hits[0]["chunk_id"]), "unique"
    if len(hits) > 1:
        ids = [h["chunk_id"] for h in hits]
        return None, f"snippet 命中多个 chunk {ids}，请改用 chunk_id 标注"

    # 兜底：模糊匹配，阈值刻意收紧（0.75），并要求与次优拉开差距
    scored = sorted(
        ((difflib.SequenceMatcher(None, needle, normalize(row.get("content", ""))).ratio(), row)
         for row in snapshot),
        key=lambda x: x[0], reverse=True,
    )
    if scored and scored[0][0] >= 0.75 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
        return int(scored[0][1]["chunk_id"]), f"fuzzy({scored[0][0]:.2f})"
    return None, "snippet 在快照中匹配不到"


def resolve_dataset(rows: List[Dict[str, Any]],
                    snapshot: List[Dict[str, Any]]) -> Tuple[List[QueryItem], List[str]]:
    """校验并解析标注集。返回 (items, warnings)。有致命错误时抛 DatasetError。"""
    errors: List[str] = []
    warnings: List[str] = []
    items: List[QueryItem] = []
    seen_ids = set()
    valid_chunk_ids = {int(r["chunk_id"]) for r in snapshot}

    for idx, row in enumerate(rows, 1):
        qid = str(row.get("id") or f"row{idx}")
        tag = f"[{qid}]"

        if qid in seen_ids:
            errors.append(f"{tag} id 重复")
        seen_ids.add(qid)

        query = (row.get("query") or "").strip()
        if not query:
            errors.append(f"{tag} query 为空")

        key_points = [p.strip() for p in (row.get("key_points") or []) if str(p).strip()]
        if not key_points:
            errors.append(f"{tag} key_points 为空，faithfulness 与要点覆盖率会无法计算")

        raw_gold = row.get("gold") or []
        if not raw_gold:
            errors.append(f"{tag} gold 为空，检索指标会全部为 n/a")

        gold_items: List[GoldItem] = []
        for g_idx, g in enumerate(raw_gold, 1):
            if not isinstance(g, dict):
                errors.append(f"{tag} gold[{g_idx}] 必须是对象")
                continue
            grade = int(g.get("grade", 1))
            if grade not in (1, 2):
                warnings.append(f"{tag} gold[{g_idx}] grade={grade} 非 1/2，已按二值处理")
                grade = 1

            if g.get("chunk_id") is not None:
                try:
                    cid = int(g["chunk_id"])
                except (TypeError, ValueError):
                    errors.append(f"{tag} gold[{g_idx}] chunk_id 不是整数")
                    continue
                if cid not in valid_chunk_ids:
                    errors.append(f"{tag} gold[{g_idx}] chunk_id={cid} 不在当前快照中（快照过期？）")
                    continue
                gold_items.append(GoldItem(chunk_id=cid, grade=grade))
            elif g.get("snippet"):
                cid, note = _match_snippet(str(g["snippet"]), snapshot)
                if cid is None:
                    errors.append(f"{tag} gold[{g_idx}] {note}：{str(g['snippet'])[:40]}")
                    continue
                if note.startswith("fuzzy"):
                    warnings.append(f"{tag} gold[{g_idx}] 模糊匹配到 chunk {cid}（{note}），建议人工复核")
                gold_items.append(GoldItem(chunk_id=cid, snippet=str(g["snippet"]), grade=grade))
            else:
                errors.append(f"{tag} gold[{g_idx}] 需要 chunk_id 或 snippet 之一")

        if gold_items and len({g.chunk_id for g in gold_items}) != len(gold_items):
            errors.append(f"{tag} gold 中有重复 chunk_id")

        item_names = row.get("item_names")
        if item_names is not None:
            if not isinstance(item_names, list) or not all(isinstance(x, str) for x in item_names):
                errors.append(f"{tag} item_names 必须是字符串列表")
                item_names = None

        items.append(QueryItem(
            id=qid,
            query=query,
            key_points=key_points,
            gold=gold_items,
            item_names=item_names,
            rewritten_query=(row.get("rewritten_query") or None),
        ))

    if errors:
        raise DatasetError("标注集校验失败：\n  - " + "\n  - ".join(errors))
    return items, warnings


def write_resolved(items: List[QueryItem], path: Path) -> None:
    """把解析结果写回（chunk_id 已回填），便于人工复核后再提交。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            row = {
                "id": it.id,
                "query": it.query,
                "item_names": it.item_names,
                "rewritten_query": it.rewritten_query,
                "gold": [{"chunk_id": g.chunk_id, "grade": g.grade,
                          **({"snippet": g.snippet} if g.snippet else {})} for g in it.gold],
                "key_points": it.key_points,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
