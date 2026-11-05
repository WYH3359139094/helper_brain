"""导出 Milvus 里的 chunk 快照，供人工标注 gold chunk_id。

用法：
    cd D:/flaskkk111/shopkeeper_brain
    knowledge\\.venv\\Scripts\\python.exe -m knowledge.eval.export_chunks

产物：knowledge/eval/data/chunks_snapshot.jsonl
    每行 {chunk_id, title, parent_title, file_title, item_name, content, content_sha1, char_len}
另外打印一份"人工标注速查表"（id + 标题 + 前 60 字），标注时直接抄 snippet 即可。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "chunks_snapshot.jsonl"

OUTPUT_FIELDS = ["chunk_id", "title", "parent_title", "file_title", "item_name", "content"]


def _load_env() -> None:
    # 显式指定路径，避免依赖启动时的工作目录
    load_dotenv(KNOWLEDGE_ROOT / ".env", override=False)


def fetch_chunks(collection: Optional[str] = None) -> List[Dict[str, Any]]:
    from pymilvus import MilvusClient  # 延后 import，--help 时不必装依赖

    uri = os.getenv("MILVUS_URL")
    if not uri:
        raise SystemExit("缺少环境变量 MILVUS_URL，检查 knowledge/.env")
    collection = collection or os.getenv("CHUNKS_COLLECTION", "kb_chunks_v1")

    client = MilvusClient(uri=uri)
    if not client.has_collection(collection):
        raise SystemExit(f"集合不存在: {collection}")

    rows: List[Dict[str, Any]] = []
    iterator = client.query_iterator(
        collection_name=collection,
        batch_size=1000,
        filter="chunk_id > 0",
        output_fields=OUTPUT_FIELDS,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows.extend(batch)
    finally:
        iterator.close()

    rows.sort(key=lambda r: int(r["chunk_id"]))
    for r in rows:
        content = r.get("content", "") or ""
        r["content_sha1"] = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        r["char_len"] = len(content)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Milvus chunk 快照")
    parser.add_argument("--collection", default=None, help="默认取 .env 的 CHUNKS_COLLECTION")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--print-limit", type=int, default=200, help="速查表最多打印多少行")
    args = parser.parse_args()

    _load_env()
    rows = fetch_chunks(args.collection)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_doc: Dict[str, int] = {}
    for r in rows:
        by_doc[r.get("file_title", "")] = by_doc.get(r.get("file_title", ""), 0) + 1

    print(f"已导出 {len(rows)} 个 chunk -> {out_path}")
    for doc, cnt in by_doc.items():
        print(f"  - {doc}: {cnt} 个")
    print("\n人工标注速查表（chunk_id | 标题 | 内容前 60 字）：")
    for r in rows[: args.print_limit]:
        preview = (r.get("content") or "").replace("\n", " ")[:60]
        print(f"  {r['chunk_id']} | {r.get('title', '')[:24]:24s} | {preview}")


if __name__ == "__main__":
    main()
