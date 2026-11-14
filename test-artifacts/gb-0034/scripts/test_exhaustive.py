"""小序列穷举校验：独立纯 Python 枚举全部状态路径，对照两套引擎输出。

不依赖 numpy/scipy 实现参考解，避免与被测代码共用数值套路：
- 枚举 K^T 条路径的精确（float 逐步累加）联合概率；
- 序列 logL、逐时刻边际后验、联合 MAP 及字典序平局规则逐项对照；
- 覆盖 2..5 隐态、2..6 观测码、零转移、不可能序列、随机参数多组用例。
"""
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend')))

from src.services.parameters import normalize_parameters  # noqa: E402
from src.services.hmm import infer_sequence  # noqa: E402
from src.services.engine import infer_streaming, MemorySink  # noqa: E402

TOL = 1e-10


def _safe_log(value):
    return math.log(value) if value > 0 else math.inf * -1


def brute_force(obs, pi, A, B):
    """枚举全部路径的参考解。返回 logL、后验矩阵、Viterbi 路径与联合 logP。"""
    k = len(pi)
    t_total = len(obs)
    paths = list(itertools.product(range(k), repeat=t_total))
    joints = []
    for path in paths:
        value = _safe_log(pi[path[0]]) + _safe_log(B[path[0]][obs[0]])
        for t in range(1, t_total):
            value += _safe_log(A[path[t - 1]][path[t]]) + _safe_log(B[path[t]][obs[t]])
        joints.append(value)
    finite = [(p, v) for p, v in zip(paths, joints) if math.isfinite(v)]
    if not finite:
        return None
    log_likelihood = math.log(sum(math.exp(v) for _, v in finite))
    posterior = []
    for t in range(t_total):
        col = []
        for state in range(k):
            mass = sum(math.exp(v) for p, v in finite if p[t] == state)
            col.append(mass / math.exp(log_likelihood))
        posterior.append(col)
    best = max(v for _, v in finite)
    # 字典序最小的最优路径（tuple 直接比较即状态索引序列字典序）
    viterbi = min(p for p, v in finite if v == best)
    return {'log_likelihood': log_likelihood, 'posterior': posterior,
            'viterbi_path': list(viterbi), 'viterbi_log_joint': best}


def random_model(rng, k, m, sparse=False):
    if sparse:
        # 显式保留精确零：每行随机置零 1..2 格后重新归一，模拟真实零转移/零发射
        pi = [rng.random() + 0.05 for _ in range(k)]
        s = sum(pi)
        pi = [v / s for v in pi]
        A = []
        for _ in range(k):
            row = [rng.random() + 0.05 for _ in range(k)]
            for index in rng.sample(range(k), rng.randrange(1, k)):
                row[index] = 0.0
            s = sum(row)
            A.append([v / s for v in row])
        B = []
        for _ in range(k):
            row = [rng.random() + 0.05 for _ in range(m)]
            for index in rng.sample(range(m), rng.randrange(1, m)):
                row[index] = 0.0
            s = sum(row)
            if s == 0:
                row[0] = 1.0
            else:
                row = [v / s for v in row]
            B.append(row)
        return pi, A, B
    pi = [rng.random() + 0.05 for _ in range(k)]
    s = sum(pi)
    pi = [v / s for v in pi]
    A = []
    for _ in range(k):
        row = [rng.random() + 0.05 for _ in range(k)]
        s = sum(row)
        A.append([v / s for v in row])
    B = []
    for _ in range(k):
        row = [rng.random() + 0.05 for _ in range(m)]
        s = sum(row)
        B.append([v / s for v in row])
    return pi, A, B


def check_case(name, obs, pi, A, B, block_sizes):
    normalized = normalize_parameters(pi, A, B)
    p, an, bn = normalized['pi'], normalized['transition'], normalized['emission']
    expected = brute_force(obs, p, an, bn)
    legacy = infer_sequence(obs, pi, A, B)
    if expected is None:
        raise AssertionError(f'{name}: 穷举判定不可能，但引擎未报错')
    assert abs(legacy['log_likelihood'] - expected['log_likelihood']) < TOL, name
    assert legacy['viterbi_path'] == expected['viterbi_path'], (
        f'{name}: viterbi {legacy["viterbi_path"]} != {expected["viterbi_path"]}')
    assert abs(legacy['viterbi_log_joint'] - expected['viterbi_log_joint']) < TOL, name
    for t in range(len(obs)):
        for s in range(len(p)):
            assert abs(legacy['posterior'][t][s] - expected['posterior'][t][s]) < 1e-9, (
                f'{name}: posterior t{t} s{s}')
        assert abs(sum(legacy['posterior'][t]) - 1.0) < 1e-12, f'{name}: 后验列和'
        # 边际 argmax + 最小 index 平局规则
        col = legacy['posterior'][t]
        assert legacy['marginal_path'][t] == min(range(len(col)), key=lambda s: (-col[s], s)), name
    # 流式引擎在不同块大小下必须给出完全一致的结果（块边界不影响数值）
    for block_size in block_sizes:
        summary, blocks = infer_streaming(obs, p, an, bn, block_size=block_size, sink=MemorySink())
        ordered = sorted(blocks, key=lambda blk: blk['index'])
        flat_post = [row for blk in ordered for row in blk['posterior']]
        flat_lf = [row for blk in ordered for row in blk['log_forward']]
        flat_lb = [row for blk in ordered for row in blk['log_backward']]
        assert summary['viterbi_path'] == expected['viterbi_path'], f'{name} bs={block_size}'
        assert abs(summary['log_likelihood'] - expected['log_likelihood']) < TOL, name
        # 分块归一化会引入至多数个 ULP 的差，按 1e-10 容差逐元素对照；字段语义逐键一致
        assert len(flat_post) == len(legacy['posterior'])
        assert all(abs(x - y) < 1e-10 for r1, r2 in zip(flat_post, legacy['posterior'])
                   for x, y in zip(r1, r2)), f'{name} bs={block_size} posterior'
        assert all((x is None and y is None) or
                   (x is not None and y is not None and abs(x - y) < 1e-10)
                   for r1, r2 in zip(flat_lf, legacy['log_forward'])
                   for x, y in zip(r1, r2)), f'{name} bs={block_size} log_forward'
        assert all((x is None and y is None) or
                   (x is not None and y is not None and abs(x - y) < 1e-10)
                   for r1, r2 in zip(flat_lb, legacy['log_backward'])
                   for x, y in zip(r1, r2)), f'{name} bs={block_size} log_backward'
        assert summary['marginal_path'] == legacy['marginal_path'], name
    return expected


def main():
    rng = random.Random(20261114)
    # README 记载的标准二态三步校验
    pi = [0.6, 0.4]
    A = [[0.7, 0.3], [0.4, 0.6]]
    B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]
    result = check_case('readme-2state', [0, 1, 2], pi, A, B, [1, 2, 3, 5])
    assert abs(math.exp(result['log_likelihood']) - 907 / 25000) < 1e-12
    assert result['viterbi_path'] == [0, 0, 1]
    assert abs(result['log_likelihood'] - (-3.316488653735)) < 1e-9

    # README 的三态零转移校验：边际最高态 002 违反 A[0,2]=0，联合 MAP 为 012
    pi3 = [1.0, 0.0, 0.0]
    A3 = [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    B3 = [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.45, 0.45, 0.1]]
    result = check_case('readme-3state-zero', [0, 1, 2], pi3, A3, B3, [1, 2, 3])
    assert result['viterbi_path'] == [0, 1, 2]
    assert abs(math.exp(result['log_likelihood']) - 1 / 40) < 1e-12
    legacy = infer_sequence([0, 1, 2], pi3, A3, B3)
    assert legacy['marginal_path'] == [0, 0, 2]
    assert legacy['marginal_path_is_legal'] is False
    assert legacy['marginal_log_joint'] is None
    # null 明确表示 -inf
    assert legacy['log_forward'][0][1] is None and legacy['log_forward'][0][2] is None

    # 初态不可能发射：pi 只允许 state0，而 state0 不发射 D(=3)，整个任务失败，不拼接部分结果
    impossible = False
    try:
        infer_sequence([3], pi3, A3, B3)
    except Exception:
        impossible = True
    assert impossible, '受限发射序列必须报不可能'
    # 中途所有路径被零约束封死：state2 只能自循环且不发射 A(=0)，回到 A 即全路径为零
    bz = [[.5, .5, 0., 0.], [0., .5, .5, 0.], [0., 0., .4, .6]]
    impossible_mid = False
    try:
        infer_sequence([0, 2, 3, 0], pi3, A3, bz)
    except Exception:
        impossible_mid = True
    assert impossible_mid, '中途零概率封死必须报不可能，且不能把0改成小正数'
    # 该模型的可行序列仍要算对，并覆盖零发射列与跨块边界
    check_case('zero-emission-feasible', [0, 1, 2, 3], pi3, A3, bz, [1, 2, 3, 4])

    # 跨块零转移：在块边界（bs 强制为 3）放置零转移约束，穷举对照
    check_case('zero-across-boundary', [0, 1, 2, 0, 1, 2], pi3, A3, B3, [2, 3, 4, 6])

    # 随机参数矩阵：2..5 态、2..6 码、序列 1..7，稠密与含精确零各一轮
    case_count = 0
    for sparse in (False, True):
        for k in range(2, 6):
            for m in range(2, 7):
                pi_r, ar, br = random_model(rng, k, m, sparse=sparse)
                for t_total in range(1, 8):
                    obs = [rng.randrange(m) for _ in range(t_total)]
                    normalized = normalize_parameters(pi_r, ar, br)
                    if brute_force(obs, normalized['pi'], normalized['transition'],
                                   normalized['emission']) is None:
                        # 随机零矩阵下的不可能序列，引擎必须显式失败而不是产出部分结果
                        failed = False
                        try:
                            infer_sequence(obs, pi_r, ar, br)
                        except Exception:
                            failed = True
                        assert failed, f'随机零矩阵不可能序列未失败 k{k} m{m} t{t_total}'
                        continue
                    check_case(f'random{"-sparse" if sparse else ""}-k{k}-m{m}-t{t_total}',
                               obs, pi_r, ar, br,
                               [1, 2, max(1, t_total // 2), t_total, t_total + 3])
                    case_count += 1
    print(f'PASS 穷举对照 {case_count + 5} 组（含 README 两组标准校验、跨块零转移与零发射）')


if __name__ == '__main__':
    main()
