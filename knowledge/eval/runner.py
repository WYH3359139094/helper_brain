"""评测执行器：跑一次「澄清 → 混合检索 → HyDE → RRF → 精排 → 生成答案」。

设计取舍（写在这里是为了复跑时能解释清楚）：
1. 直接复用项目里的节点类，而不是重写检索逻辑——评测要测的是线上真实行为；
2. 默认**不调用** Web MCP 节点：联网结果每次不同，会破坏可复跑性（--with-web 可开）；
3. 评测期间把 Mongo 的历史读写打桩成空：避免污染线上会话，也保证同一 query 每次输入一致；
4. 每条 query 的原始产出（各阶段排序、上下文、答案）落盘缓存，改指标口径时不必重跑模型。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge.eval.dataset import QueryItem

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "runs"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunRecord:
    qid: str
    query: str
    stage: str                      # ok / clarify / error
    rankings: Dict[str, List[int]] = field(default_factory=dict)
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    rewritten_query: str = ""
    item_names: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid, "query": self.query, "stage": self.stage,
            "rankings": self.rankings, "contexts": self.contexts,
            "answer": self.answer, "rewritten_query": self.rewritten_query,
            "item_names": self.item_names, "timings": self.timings, "notes": self.notes,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        return cls(**d)


class EvalRunner:
    """串起评测用的最小 pipeline。节点实例只创建一次，模型只加载一次。"""

    def __init__(self, with_web: bool = False, use_cache: bool = True,
                 cache_dir: Path = CACHE_DIR):
        self.with_web = with_web
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self._nodes: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _patch_side_effects(self) -> None:
        """评测期间隔离 Mongo 历史读写，保证单次可复跑。"""
        import knowledge.processor.query_processor.nodes.answer_output_node as aon
        import knowledge.processor.query_processor.nodes.item_name_confirmed_node as icn

        aon.save_chat_message = lambda **kwargs: None          # 不写历史
        icn.get_recent_messages = lambda **kwargs: []          # 不读历史

    def _ensure_nodes(self) -> Dict[str, Any]:
        if self._nodes is not None:
            return self._nodes

        self._patch_side_effects()

        from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutPutNode
        from knowledge.processor.query_processor.nodes.hybrid_vector_search_node import HybridVectorSearch
        from knowledge.processor.query_processor.nodes.hyde_vector_search_node import HyDeVectorSearchNode
        from knowledge.processor.query_processor.nodes.item_name_confirmed_node import ItemNameConfirmedNode
        from knowledge.processor.query_processor.nodes.reranker_node import RerankerNode
        from knowledge.processor.query_processor.nodes.rrf_merge_node import RrfMergeNode

        nodes = {
            "item_name": ItemNameConfirmedNode(),
            "hybrid": HybridVectorSearch(),
            "hyde": HyDeVectorSearchNode(),
            "rrf": RrfMergeNode(),
            "rerank": RerankerNode(),
            "answer": AnswerOutPutNode(),
        }
        if self.with_web:
            from knowledge.processor.query_processor.nodes.web_mcp_search_node import WebMcpSearchNode
            nodes["web"] = WebMcpSearchNode()

        self._nodes = nodes
        return nodes

    # ------------------------------------------------------------------
    # 单条执行
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(state: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """节点可能返回完整 state，也可能只返回增量 dict。"""
        if isinstance(result, dict) and result is not state:
            state.update(result)
        return state

    def run(self, item: QueryItem) -> RunRecord:
        cache_key = _sha("|".join([
            item.id, item.query, item.rewritten_query or "",
            ",".join(item.item_names or []), str(self.with_web),
        ]))
        cache_path = self.cache_dir / f"{cache_key}.json"
        if self.use_cache and cache_path.exists():
            record = RunRecord.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
            record.cached = True
            return record

        record = self._run_fresh(item)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        return record

    def _run_fresh(self, item: QueryItem) -> RunRecord:
        from knowledge.processor.query_processor.state import create_default_state

        nodes = self._ensure_nodes()
        state = create_default_state(
            session_id=f"eval::{item.id}",
            task_id=f"eval::{item.id}",
            original_query=item.query,
            is_stream=False,
        )
        record = RunRecord(qid=item.id, query=item.query, stage="ok")
        timings: Dict[str, float] = {}

        try:
            # ---- 1. 商品名澄清（标注里预置了 item_names 就跳过，隔离澄清策略的影响）
            if item.item_names:
                state["item_names"] = list(item.item_names)
                state["rewritten_query"] = item.rewritten_query or item.query
            else:
                t0 = time.time()
                self._merge(state, nodes["item_name"].process(state))
                timings["item_name"] = round(time.time() - t0, 2)
                if state.get("answer"):                       # 走了澄清/抱歉分支
                    record.stage = "clarify"
                    record.answer = state.get("answer", "")
                    record.item_names = state.get("item_names") or []
                    record.rewritten_query = state.get("rewritten_query") or ""
                    record.timings = timings
                    return record

            if item.rewritten_query:
                state["rewritten_query"] = item.rewritten_query

            # ---- 2. 混合检索（dense + sparse）
            t0 = time.time()
            self._merge(state, nodes["hybrid"].process(state))
            timings["hybrid"] = round(time.time() - t0, 2)

            # ---- 3. HyDE 检索
            t0 = time.time()
            self._merge(state, nodes["hyde"].process(state))
            timings["hyde"] = round(time.time() - t0, 2)

            # ---- 3.5 可选：联网检索
            if "web" in nodes:
                t0 = time.time()
                self._merge(state, nodes["web"].process(state))
                timings["web"] = round(time.time() - t0, 2)

            # ---- 4. RRF 融合
            t0 = time.time()
            self._merge(state, nodes["rrf"].process(state))
            timings["rrf"] = round(time.time() - t0, 2)

            # ---- 5. 精排
            t0 = time.time()
            self._merge(state, nodes["rerank"].process(state))
            timings["rerank"] = round(time.time() - t0, 2)

            # ---- 6. 生成答案
            t0 = time.time()
            self._merge(state, nodes["answer"].process(state))
            timings["answer"] = round(time.time() - t0, 2)

        except Exception as e:                                  # 节点抛错 → 记录，不中断整批
            record.stage = "error"
            record.notes.append(f"{type(e).__name__}: {e}")
            record.answer = state.get("answer", "") or ""
            record.timings = timings
            return record

        # ---- 收集各阶段排序
        hybrid_hits = state.get("embedding_chunks") or []
        hyde_hits = state.get("hyde_embedding_chunks") or []
        record.rankings = {
            "hybrid": [self._hit_id(h) for h in hybrid_hits if self._hit_id(h) is not None],
            "hyde": [self._hit_id(h) for h in hyde_hits if self._hit_id(h) is not None],
            "rrf": [c.get("chunk_id") for c in (state.get("rrf_chunks") or []) if c.get("chunk_id")],
            "rerank": [d.get("chunk_id") for d in (state.get("reranked_docs") or []) if d.get("chunk_id")],
        }
        record.contexts = [
            {
                "chunk_id": d.get("chunk_id"),
                "title": d.get("title", ""),
                "source": d.get("source", ""),
                "score": round(float(d["score"]), 6) if d.get("score") is not None else None,
                "content": d.get("content", ""),
            }
            for d in (state.get("reranked_docs") or [])
        ]
        record.answer = state.get("answer", "") or ""
        record.rewritten_query = state.get("rewritten_query", "") or ""
        record.item_names = state.get("item_names") or []
        record.timings = timings

        # 降级信号：任一阶段空结果都在报告里显式标出来，不要让它悄悄变成"答得不好"
        if not record.rankings["hybrid"]:
            record.notes.append("hybrid_empty")
        if not record.rankings["hyde"]:
            record.notes.append("hyde_empty")
        if not record.rankings["rerank"]:
            record.notes.append("rerank_empty")
        return record

    @staticmethod
    def _hit_id(hit: Any) -> Optional[int]:
        """Milvus hit → chunk_id（兼容 {"id":..} 与 {"entity":{...}} 两种形态）。"""
        if not isinstance(hit, dict):
            return None
        entity = hit.get("entity") or {}
        for raw in (entity.get("chunk_id"), hit.get("chunk_id"), hit.get("id")):
            try:
                if raw is not None:
                    return int(raw)
            except (TypeError, ValueError):
                continue
        return None
