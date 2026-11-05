"""一键跑评测：读标注集 → 跑检索+生成 → 算指标 → 输出可复跑的表。

用法（工作目录必须是项目根）：
    cd D:/flaskkk111/shopkeeper_brain

    # 1) 先导出 chunk 快照（标注 gold 时用）
    knowledge\\.venv\\Scripts\\python.exe -m knowledge.eval.export_chunks

    # 2) 跑评测（默认 20 条全量、LLM 判官开启、命中缓存就直接复用）
    knowledge\\.venv\\Scripts\\python.exe -m knowledge.eval.run_eval

常用参数：
    --judge none          关掉 LLM 判官，只算检索指标（不花钱、秒级）
    --limit 5             先跑前 5 条冒烟
    --no-cache            忽略缓存强制重跑（复核一致性时用）
    --with-web            打开 Web MCP 联网检索（默认关：联网会让结果不可复跑）
    --tag before-rerank   给这次结果打标签，输出到 results/<时间>_<tag>/

产物：
    results/<时间>_<tag>/report.md      markdown 表格（含汇总与分阶段对比）
    results/<时间>_<tag>/per_query.csv  逐条明细（utf-8-sig，Excel 可直接开）
    results/<时间>_<tag>/summary.json   机器可读汇总，方便接 CI 卡阈值
    results/<时间>_<tag>/meta.json      本次运行的参数/模型/数据指纹
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from knowledge.eval import metrics as M
from knowledge.eval.dataset import (DatasetError, load_jsonl, load_snapshot,
                                    resolve_dataset, write_resolved)
from knowledge.eval.judge import JudgeUnavailable, LLMJudge
from knowledge.eval.runner import EvalRunner

PKG_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_ROOT = PKG_ROOT.parent


def _load_env() -> None:
    load_dotenv(KNOWLEDGE_ROOT / ".env", override=False)


# ----------------------------------------------------------------------
# 指标汇总
# ----------------------------------------------------------------------
STAGES = ("hybrid", "rrf", "rerank")


def score_one(item, record, k: int, ndcg_k: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": item.id,
        "query": item.query,
        "stage": record.stage,
        "gold_n": len(item.gold_ids),
        "first_hit": None,
        "answer_chars": len(record.answer or ""),
        "notes": ",".join(record.notes),
        "cached": record.cached,
    }

    rerank_rank = record.rankings.get("rerank") or []
    for cid in rerank_rank:
        if cid in set(item.gold_ids):
            row["first_hit"] = rerank_rank.index(cid) + 1
            break

    for stage in STAGES:
        ranked = record.rankings.get(stage) or []
        row[f"recall@{k}_{stage}"] = M.recall_at_k(ranked, item.gold_grades, k)
        row[f"ndcg@{ndcg_k}_{stage}"] = M.ndcg_at_k(ranked, item.gold_grades, ndcg_k)

    # 主口径：以精排后的最终排序为准
    row[f"recall@{k}"] = row[f"recall@{k}_rerank"]
    row[f"ndcg@{ndcg_k}"] = row[f"ndcg@{ndcg_k}_rerank"]
    row["recall@1"] = M.recall_at_k(rerank_rank, item.gold_grades, 1)
    row["mrr"] = M.mrr(rerank_rank, item.gold_grades)
    return row


def add_judge_metrics(row: Dict[str, Any], judged: Optional[Dict[str, Any]],
                      key_points: List[str]) -> None:
    if judged is None:
        row["faithfulness"] = None
        row["key_point_recall"] = None
        row["context_recall"] = None
        return

    claims = judged.get("claims") or []
    supported = sum(1 for c in claims if c.get("supported"))
    row["claims"] = len(claims)
    row["faithfulness"] = M.faithfulness(supported, len(claims)) if claims else None

    points = judged.get("key_points") or []
    total = len(key_points) or len(points)
    in_ctx = sum(1 for p in points if p.get("in_context"))
    in_ans = sum(1 for p in points if p.get("in_answer"))
    row["key_point_recall"] = M.key_point_coverage(in_ans, total) if total else None
    row["context_recall"] = M.key_point_coverage(in_ctx, total) if total else None


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------
def _cell(value: Optional[float]) -> str:
    return M.fmt(value)


def render_markdown(rows: List[Dict[str, Any]], k: int, ndcg_k: int,
                    meta: Dict[str, Any], judge_on: bool) -> str:
    lines: List[str] = []
    lines.append(f"# 检索 / 生成评测报告\n")
    lines.append(f"- 运行时间：{meta['run_at']}")
    lines.append(f"- 标注集：`{meta['dataset']}`（{len(rows)} 条，sha256:{meta['dataset_sha'][:12]}）")
    lines.append(f"- chunk 快照：{meta['chunk_count']} 个（sha256:{meta['snapshot_sha'][:12]}）")
    lines.append(f"- 判官：{'LLM（' + meta['judge_model'] + '）' if judge_on else '关闭（仅检索指标）'}")
    lines.append(f"- 参数：recall@{k}、nDCG@{ndcg_k}、RRF k={meta['rrf_k']}、"
                 f"rerank top_k={meta['rerank_min_top_k']}~{meta['rerank_max_top_k']}")
    lines.append("")

    # ---- 逐条明细
    header = ["#", "id", "query", f"R@{k}", f"nDCG@{ndcg_k}", "R@1", "MRR",
              "首命中", "ctx_recall", "faith", "备注"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for idx, row in enumerate(rows, 1):
        notes = row.get("notes") or ""
        if row["stage"] == "clarify":
            notes = (notes + ",澄清分支").strip(",")
        elif row["stage"] == "error":
            notes = (notes + ",执行失败").strip(",")
        lines.append("| " + " | ".join([
            str(idx), row["id"], row["query"].replace("|", "\\|")[:26],
            _cell(row[f"recall@{k}"]), _cell(row[f"ndcg@{ndcg_k}"]),
            _cell(row.get("recall@1")), _cell(row.get("mrr")),
            str(row["first_hit"] or "-"), _cell(row.get("context_recall")),
            _cell(row.get("faithfulness")), notes or "",
        ]) + " |")

    # ---- 汇总
    stage_rows = []
    for stage in STAGES:
        vals_r = [r[f"recall@{k}_{stage}"] for r in rows]
        vals_n = [r[f"ndcg@{ndcg_k}_{stage}"] for r in rows]
        stage_rows.append((stage, M.mean(vals_r), M.mean(vals_n),
                           sum(1 for v in vals_r if v is not None)))

    total = len(rows)
    clarify = sum(1 for r in rows if r["stage"] == "clarify")
    errors = sum(1 for r in rows if r["stage"] == "error")

    lines.append("")
    lines.append("## 汇总（最终排序口径）")
    lines.append("")
    lines.append("| 指标 | 均值 | 有效条数 |")
    lines.append("|---|---|---|")
    lines.append(f"| Recall@{k} | {_cell(M.mean([r[f'recall@{k}'] for r in rows]))} | {total - clarify - errors} |")
    lines.append(f"| nDCG@{ndcg_k} | {_cell(M.mean([r[f'ndcg@{ndcg_k}'] for r in rows]))} | {total - clarify - errors} |")
    lines.append(f"| Recall@1 | {_cell(M.mean([r.get('recall@1') for r in rows]))} | {total - clarify - errors} |")
    lines.append(f"| MRR | {_cell(M.mean([r.get('mrr') for r in rows]))} | {total - clarify - errors} |")
    if judge_on:
        lines.append(f"| 上下文要点覆盖（context_recall） | {_cell(M.mean([r.get('context_recall') for r in rows]))} | {total} |")
        lines.append(f"| 答案要点覆盖 | {_cell(M.mean([r.get('key_point_recall') for r in rows]))} | {total} |")
        lines.append(f"| faithfulness | {_cell(M.mean([r.get('faithfulness') for r in rows]))} | "
                     f"{sum(1 for r in rows if r.get('faithfulness') is not None)} |")
    lines.append("")
    lines.append(f"澄清分支 {clarify} 条、执行失败 {errors} 条、总条数 {total}。")
    lines.append("")

    # ---- 分阶段对比
    lines.append("## 分阶段对比（看 rerank 到底有没有用）")
    lines.append("")
    lines.append(f"| 阶段 | Recall@{k} | nDCG@{ndcg_k} | 有效条数 |")
    lines.append("|---|---|---|---|")
    name_map = {"hybrid": "混合检索(向量)", "rrf": "RRF 融合", "rerank": "精排后(最终)"}
    for stage, r, n, cnt in stage_rows:
        lines.append(f"| {name_map[stage]} | {_cell(r)} | {_cell(n)} | {cnt} |")
    lines.append("")

    # ---- 指标饱和提示（小语料下 R@5 很容易打满，别被数字骗了）
    main_recall = M.mean([r[f"recall@{k}"] for r in rows])
    if main_recall is not None and main_recall >= 0.95:
        lines.append(f"> ⚠️ Recall@{k} 已接近饱和（{main_recall:.3f}）。当前语料仅 "
                     f"{meta['chunk_count']} 个 chunk，随机命中的概率都不低；"
                     f"这份表更适合用来看 R@1 / MRR / 分阶段差异，扩语料后再看 R@{k} 才有区分度。")
        lines.append("")
    return "\n".join(lines)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def build_meta(args, snapshot: List[Dict[str, Any]], dataset_sha: str, judge_model: str) -> Dict[str, Any]:
    from knowledge.processor.query_processor.config import get_config

    cfg = get_config()
    versions = {}
    for pkg in ("pymilvus", "langgraph", "openai", "sentence-transformers", "fastapi", "python-dotenv"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = "n/a"

    return {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(args.dataset),
        "dataset_sha": dataset_sha,
        "snapshot_sha": _file_sha(Path(args.snapshot)),
        "chunk_count": len(snapshot),
        "judge_model": judge_model,
        "llm_model": os.getenv("LLM_DEFAULT_MODEL", ""),
        "embedding_model": os.getenv("BGE_M3_PATH", ""),
        "reranker_model": os.getenv("BGE_RERANKER_LARGE", ""),
        "milvus_collection": os.getenv("CHUNKS_COLLECTION", ""),
        "rrf_k": cfg.rrf_k,
        "rrf_max_results": cfg.rrf_max_results,
        "rerank_min_top_k": cfg.rerank_min_top_k,
        "rerank_max_top_k": cfg.rerank_max_top_k,
        "with_web": args.with_web,
        "judge": args.judge,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="知识库检索/生成离线评测")
    parser.add_argument("--dataset", default=str(PKG_ROOT / "data" / "queries.jsonl"))
    parser.add_argument("--snapshot", default=str(PKG_ROOT / "data" / "chunks_snapshot.jsonl"))
    parser.add_argument("--out", default=str(PKG_ROOT / "results"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--judge", choices=["llm", "none"], default="llm")
    parser.add_argument("--workers", type=int, default=4, help="判官并发数（纯网络 IO）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全量")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存强制重跑")
    parser.add_argument("--with-web", action="store_true", help="打开联网检索（会破坏可复跑性）")
    parser.add_argument("--write-resolved", action="store_true", default=True,
                        help="把 snippet 解析出的 chunk_id 回写到 queries.resolved.jsonl")
    args = parser.parse_args(argv)

    _load_env()

    dataset_path = Path(args.dataset)
    snapshot_path = Path(args.snapshot)

    # ---- 1. 读数据
    try:
        snapshot = load_snapshot(snapshot_path)
        items, warnings = resolve_dataset(load_jsonl(dataset_path), snapshot)
    except DatasetError as e:
        print(f"[标注集错误] {e}", file=sys.stderr)
        print("\n提示：先运行 python -m knowledge.eval.export_chunks 生成快照", file=sys.stderr)
        return 2

    for w in warnings:
        print(f"[warn] {w}")
    if args.limit:
        items = items[: args.limit]

    if args.write_resolved:
        resolved_path = dataset_path.with_name(dataset_path.stem + ".resolved.jsonl")
        write_resolved(items, resolved_path)
        print(f"[i] chunk_id 已回填，人工复核可看：{resolved_path}")

    # ---- 2. 跑 pipeline
    runner = EvalRunner(with_web=args.with_web, use_cache=not args.no_cache)
    judge: Optional[LLMJudge] = None
    judge_error = ""
    if args.judge == "llm":
        try:
            judge = LLMJudge(use_cache=not args.no_cache)
        except JudgeUnavailable as e:
            judge_error = str(e)
            print(f"[warn] LLM 判官不可用，faithfulness 记为 n/a：{judge_error}")

    rows: List[Dict[str, Any]] = []
    t_start = time.time()

    # ---- 第一阶段：跑检索 + 生成（模型在 GPU 上，串行跑更快也更稳）
    records: List[Any] = []
    for idx, item in enumerate(items, 1):
        print(f"[retrieval {idx}/{len(items)}] {item.id} {item.query[:30]} ...", flush=True)
        t_item = time.time()
        record = runner.run(item)
        records.append(record)
        row = score_one(item, record, args.k, args.ndcg_k)
        rows.append(row)
        print(f"       -> {record.stage}  R@{args.k}={M.fmt(row[f'recall@{args.k}'])}  "
              f"nDCG@{args.ndcg_k}={M.fmt(row[f'ndcg@{args.ndcg_k}'])}  "
              f"rerank={len(record.rankings.get('rerank') or [])}条  "
              f"{time.time() - t_item:.1f}s", flush=True)

    # ---- 第二阶段：LLM 判官（纯网络 IO，并发跑，--workers 控制并发数）
    if judge is not None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _judge_one(pair):
            idx, item, record = pair
            return idx, judge.judge(item.query, record.answer, record.contexts, item.key_points)

        print(f"\n[judge] 并发 {args.workers} 路开始判定 {len(items)} 条 ...", flush=True)
        t_judge = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(_judge_one, (i, it, records[i])): i
                       for i, it in enumerate(items)}
            for done, fut in enumerate(as_completed(futures), 1):
                idx = futures[fut]
                try:
                    _, judged = fut.result()
                except JudgeUnavailable as e:
                    judge_error = str(e)
                    judged = None
                    print(f"[warn] 判官不可用：{e}", flush=True)
                add_judge_metrics(rows[idx], judged, items[idx].key_points)
                print(f"[judge {done}/{len(items)}] {items[idx].id} "
                      f"faith={M.fmt(rows[idx].get('faithfulness'))}", flush=True)
        print(f"[judge] 完成，用时 {time.time() - t_judge:.1f}s\n", flush=True)
    else:
        for row in rows:
            add_judge_metrics(row, None, [])

    elapsed = time.time() - t_start

    # ---- 3. 输出
    dataset_sha = _file_sha(dataset_path)
    judge_model = (judge.model if judge else "") or "关闭"
    meta = build_meta(args, snapshot, dataset_sha, judge_model)
    meta["elapsed_sec"] = round(elapsed, 1)
    meta["judge_calls"] = judge.call_count if judge else 0
    meta["judge_cache_hits"] = judge.cache_hits if judge else 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / (f"{stamp}_{args.tag}" if args.tag else stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = render_markdown(rows, args.k, args.ndcg_k, meta, judge is not None)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    write_csv(rows, out_dir / "per_query.csv")

    summary = {
        "n": len(rows),
        f"recall@{args.k}": M.mean([r[f"recall@{args.k}"] for r in rows]),
        f"ndcg@{args.ndcg_k}": M.mean([r[f"ndcg@{args.ndcg_k}"] for r in rows]),
        "context_recall": M.mean([r.get("context_recall") for r in rows]),
        "key_point_recall": M.mean([r.get("key_point_recall") for r in rows]),
        "faithfulness": M.mean([r.get("faithfulness") for r in rows]),
        "clarify": sum(1 for r in rows if r["stage"] == "clarify"),
        "error": sum(1 for r in rows if r["stage"] == "error"),
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(report)
    print(f"\n结果目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
