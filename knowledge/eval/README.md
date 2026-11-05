# knowledge/eval — 检索与生成离线评测

给知识库问答做一套**可复跑**的评测：20 条人工标注 query，量 Recall@5 / nDCG@10 / faithfulness，
输出一张直接能贴进汇报的 markdown 表 + 一份 Excel 可开的明细 CSV。

## 目录

```
knowledge/eval/
├── data/
│   ├── queries.jsonl            # 标注集（人写，提交进版本库）
│   ├── queries.resolved.jsonl   # 脚本回填 chunk_id 后的副本（复核用，自动生成）
│   └── chunks_snapshot.jsonl    # Milvus chunk 快照（自动生成，别手改）
├── cache/runs/                  # 每条 query 的原始产出缓存（阶段排序/答案）
├── cache/judge/                 # 判官判定缓存
├── results/<时间>_<tag>/         # 每次运行的 report.md / per_query.csv / summary.json / meta.json
├── export_chunks.py             # 导出 chunk 快照
├── dataset.py                   # 标注集校验 + snippet→chunk_id 解析
├── runner.py                    # 跑 pipeline（澄清→检索→RRF→精排→生成）
├── judge.py                     # LLM 判官（faithfulness / 要点覆盖）
├── metrics.py                   # 纯函数指标（Recall / nDCG / MRR）
├── test_metrics.py              # 指标单测
└── run_eval.py                  # 一键入口
```

## 三步跑起来

```bat
cd /d D:\flaskkk111\shopkeeper_brain
knowledge\.venv\Scripts\activate.bat

:: 1) 导出 chunk 快照（换语料/重导入后必须重跑，chunk_id 会变）
python -m knowledge.eval.export_chunks

:: 2) 跑全量 20 条（LLM 判官默认开）
python -m knowledge.eval.run_eval --tag baseline

:: 3) 只算检索指标，秒级、不花钱
python -m knowledge.eval.run_eval --judge none --tag retrieval-only
```

常用参数：`--limit 5` 冒烟、`--no-cache` 强制重跑、`--with-web` 打开联网检索（默认关）。
指标单测：`python -m knowledge.eval.test_metrics`（有 pytest 也可 `pytest knowledge/eval/test_metrics.py -q`）。

## 标注规范（queries.jsonl，一行一条）

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | ✅ | 唯一，如 `Q01` |
| `query` | ✅ | 用户原问题，尽量口语化，别照抄标题 |
| `item_names` | 建议 | 预置商品名。填了会**跳过澄清节点**，隔离「澄清策略」对检索指标的干扰 |
| `rewritten_query` | 可选 | 人工指定检索 query；不填则用 LLM 改写结果 |
| `gold` | ✅ | 标准答案 chunk 列表，每项 `{"chunk_id": 123, "grade": 2}` 或 `{"snippet": "原文片段", "grade": 2}` |
| `key_points` | ✅ | 标准答案要点（2~4 条），算要点覆盖率与 faithfulness 用 |

- `grade`：`2` = 核心 chunk（有它就能答对），`1` = 支撑 chunk。缺省 1。
- **优先写 `snippet`**：从原文里摘 8~20 字，重导语料后仍可自动重新对齐；写 `chunk_id` 更精确但换语料就失效。
- snippet 走「去标点去空白」的归一化包含匹配：命中必须唯一，否则直接报错并列出候选，不会静默标错。

扩到 40/100 条时按主题分组即可（安全类 / 测量类 / 按键类 / 规格类 / 维护类），保证每组比例与线上问题分布接近。

## 指标口径

| 指标 | 定义 | 看什么 |
|---|---|---|
| Recall@5 | top5 命中的 gold 数 / 该条 gold 总数 | 召回够不够，能不能把答案捞回来 |
| nDCG@10 | 按 grade 加权的排序质量（IDCG 用理想排序归一） | 排序好不好，核心 chunk 有没有被压下去 |
| Recall@1 / MRR | 首位命中率 / 首个相关结果倒数位次 | 小语料下 R@5 饱和时，这两个才有区分度 |
| context_recall | 标准要点中「能在检索上下文找到依据」的比例 | 检索侧的上限 |
| key_point_recall | 标准要点中「在答案里被说且被上下文支持」的比例 | 生成侧的召回 |
| faithfulness | 答案的原子陈述中「被上下文支持」的比例（RAGAS 口径） | 有没有编，幻觉多不多 |

注意：`context_recall` / `key_point_recall` / `faithfulness` 由 LLM 判官给出，本身有误差，
关键结论建议人工抽检 20%~30%；判官输入全部落盘在 `cache/judge/`，随时可复查单条的判定理由来源。

## 可复跑是怎么保证的

1. **输入固定**：评测期间 Mongo 历史读写被打桩为空，session 按 query 隔离，同一 query 每次输入完全一致。
2. **默认不联网**：Web MCP 结果每次不同，会破坏可比性，只有显式 `--with-web` 才启用。
3. **温度 0 + 缓存**：LLM 判定按 `(prompt 版本 + 模型 + 输入哈希)` 缓存，重跑数字不会漂移；要复核一致性用 `--no-cache` 跑两遍对比。
4. **指纹留痕**：`meta.json` 记录标注集 sha256、快照 sha256、模型名、RRF k、rerank 区间、依赖版本。
5. **分层落盘**：`cache/runs/*.json` 存各阶段排序，改指标口径（比如换 k）时无需重跑 GPU。

## 已知局限（写在报告里，别藏着）

- 当前语料只有 **21 个 chunk**，Recall@5 很容易饱和；表里的 ⚠️ 提示就是在说这件事。要真正评估检索质量，建议先扩到 300+ chunk、20+ 文档。
- nDCG 只用人工标注的 grade，标注本身有主观性；扩充时建议两人独立标注、算一次标注一致性。
- 判官与被测系统可能用同一个模型，存在「自己判自己」的偏置；有条件就换一个模型当裁判。
- 每条 query 只跑 1 次，没有做随机性方差评估；模型有抖动的场景建议跑 3 次取均值。
