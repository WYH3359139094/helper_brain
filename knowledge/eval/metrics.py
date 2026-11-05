"""检索 / 生成评测的纯函数指标实现。

设计约束：
1. 不 import 项目内任何模块，不依赖 numpy / torch，方便单元测试和复用；
2. 所有函数对「空 gold」返回 None 而不是 0，避免把「没标注」算成「没检索到」；
3. nDCG 支持多级相关度（grade），默认二值。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Union

Relevance = Union[Sequence[int], Dict[int, int]]


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------
def _as_relevance_map(gold: Relevance, relevance: Optional[Relevance] = None) -> Dict[int, int]:
    """把 gold 统一成 {chunk_id: grade}。

    gold 可以是 [id1, id2] 或 {id1: 2, id2: 1}；
    relevance 参数可额外覆盖等级（gold 为 list 时用它给等级）。
    """
    if isinstance(gold, dict):
        return {int(k): int(v) for k, v in gold.items()}
    grades = {int(cid): 1 for cid in gold}
    if relevance:
        if isinstance(relevance, dict):
            for k, v in relevance.items():
                if int(k) in grades:
                    grades[int(k)] = int(v)
        else:
            for cid in relevance:
                if int(cid) in grades:
                    grades[int(cid)] = 2
    return grades


def _to_int_list(ranked: Iterable) -> List[int]:
    """把检索结果里的 id 转成 int 列表，保持顺序，跳过无效值。"""
    out: List[int] = []
    for item in ranked:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


# ----------------------------------------------------------------------
# 检索指标
# ----------------------------------------------------------------------
def recall_at_k(ranked: Iterable, gold: Relevance, k: int = 5) -> Optional[float]:
    """Recall@k = |gold ∩ topk| / |gold|。

    gold 为空时返回 None（该条不计入均值）。
    """
    gold_map = _as_relevance_map(gold)
    if not gold_map:
        return None
    topk = set(_to_int_list(ranked)[:k])
    hit = len(topk & set(gold_map.keys()))
    return hit / len(gold_map)


def ndcg_at_k(ranked: Iterable, gold: Relevance, k: int = 10,
              relevance: Optional[Relevance] = None) -> Optional[float]:
    """nDCG@k，二值或多级相关度（grade 越大越相关）。"""
    gold_map = _as_relevance_map(gold, relevance)
    if not gold_map:
        return None

    ranked_ids = _to_int_list(ranked)[:k]

    dcg = 0.0
    for idx, cid in enumerate(ranked_ids):
        grade = gold_map.get(cid, 0)
        if grade:
            dcg += (2 ** grade - 1) / math.log2(idx + 2)

    ideal_grades = sorted(gold_map.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    if idcg == 0:
        return None
    return dcg / idcg


def hit_at_k(ranked: Iterable, gold: Relevance, k: int = 5) -> Optional[float]:
    """Hit@k：topk 内是否命中任一 gold（多 gold 场景下比 recall 宽松）。"""
    gold_map = _as_relevance_map(gold)
    if not gold_map:
        return None
    return 1.0 if set(_to_int_list(ranked)[:k]) & set(gold_map.keys()) else 0.0


def mrr(ranked: Iterable, gold: Relevance) -> Optional[float]:
    """MRR：第一个相关结果位次的倒数。语料小、Recall@5 打满时靠它拉开差异。"""
    gold_map = _as_relevance_map(gold)
    if not gold_map:
        return None
    for idx, cid in enumerate(_to_int_list(ranked), 1):
        if cid in gold_map:
            return 1.0 / idx
    return 0.0


# ----------------------------------------------------------------------
# 生成指标
# ----------------------------------------------------------------------
def ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    """安全除法：分母为 0 / None 时返回 None。"""
    if not denominator:
        return None
    if numerator is None:
        return None
    return numerator / denominator


def faithfulness(supported_claims: Optional[int], total_claims: Optional[int]) -> Optional[float]:
    """faithfulness = 被上下文支持的原子陈述数 / 答案中的原子陈述总数（RAGAS 口径）。"""
    return ratio(supported_claims, total_claims)


def key_point_coverage(covered: Optional[int], total: Optional[int]) -> Optional[float]:
    """要点覆盖率 = 命中的标准答案要点数 / 该条标注要点总数。"""
    return ratio(covered, total)


# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """忽略 None 求均值；全为 None 时返回 None。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(value: Optional[float], digits: int = 3) -> str:
    """统一格式化，None 输出为 n/a。"""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"
