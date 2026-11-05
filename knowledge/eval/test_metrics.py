"""metrics 的单元测试：只测纯函数，不碰模型/数据库。

跑法：
    cd D:/flaskkk111/shopkeeper_brain
    knowledge\\.venv\\Scripts\\python.exe -m pytest knowledge/eval/test_metrics.py -q
"""
from knowledge.eval import metrics as M


def test_recall_basic():
    assert M.recall_at_k([1, 2, 3, 4, 5], [3], 5) == 1.0
    assert M.recall_at_k([1, 2, 3, 4, 5], [9], 5) == 0.0
    # 两个 gold 只命中一个
    assert M.recall_at_k([1, 2, 3], [2, 9], 5) == 0.5
    # 空 gold 返回 None，不能算 0
    assert M.recall_at_k([1, 2], [], 5) is None


def test_recall_cutoff():
    # 第 6 位命中，R@5 不算
    assert M.recall_at_k([1, 2, 3, 4, 5, 6], [6], 5) == 0.0
    assert M.recall_at_k([1, 2, 3, 4, 5, 6], [6], 10) == 1.0


def test_ndcg_perfect_and_order():
    # 唯一 gold 排第 1 → nDCG = 1
    assert abs(M.ndcg_at_k([7, 1, 2], [7], 10) - 1.0) < 1e-9
    # gold 排第 2 → 1/log2(3)
    assert abs(M.ndcg_at_k([1, 7], [7], 10) - (1 / 1.5849625007211562)) < 1e-9
    # 排在 top-k 之外 → 0
    assert M.ndcg_at_k([1, 2, 3], [7], 2) == 0.0


def test_ndcg_graded():
    # 高相关度排在前面，得分应高于排在后面
    good = M.ndcg_at_k([1, 2], {1: 2, 2: 1}, 10)
    bad = M.ndcg_at_k([2, 1], {1: 2, 2: 1}, 10)
    assert good > bad


def test_mrr():
    assert M.mrr([9, 3, 4], [3]) == 0.5
    assert M.mrr([9, 8], [3]) == 0.0
    assert M.mrr([1], []) is None


def test_mean_ignores_none():
    assert M.mean([1.0, None, 0.0]) == 0.5
    assert M.mean([None, None]) is None


def test_ratio_and_faithfulness():
    assert M.faithfulness(3, 4) == 0.75
    assert M.faithfulness(0, 0) is None
    assert M.key_point_coverage(2, 3) == 2 / 3


if __name__ == "__main__":
    # 没装 pytest 时也能跑：python -m knowledge.eval.test_metrics
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'全部通过' if not failed else f'{failed} 个用例失败'}")
    raise SystemExit(1 if failed else 0)
