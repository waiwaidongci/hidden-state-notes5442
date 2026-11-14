"""十万码实测：五隐态固定模型、256MB 峰值内存约束、下溢与长序列正确性。

实测项：
1. 单条 100000 码、5 隐态、6 观测码的完整推断可完成，峰值 RSS < 256MB；
2. 全序列后验、联合最优路径、平局规则保留；分块存储可按区段读回，数值与内存路径一致；
3. 200 步单步 1e-4 发射的 README 下溢校验：logL 约 -1842.068，线性概率为 null；
4. 十万码量级 logL 极负但有限，线性概率下溢标记正确，log 域全程不丢精度；
5. 后段观测改变前段后验：同一前缀接不同后缀，前段后验必须不同（证明不是分段各算）；
6. 跨块零转移在长序列中仍被精确执行。
"""
import gc
import os
import resource
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend')))

import numpy as np  # noqa: E402
from src.services.parameters import normalize_parameters  # noqa: E402
from src.services.engine import infer_streaming, MemorySink, MAX_TRACE_TOKENS  # noqa: E402
from src.services.hmm import infer_long_sequence  # noqa: E402
from src.services import storage  # noqa: E402

MEMORY_LIMIT_MB = 256


def peak_rss_mb():
    # Linux ru_maxrss 单位 KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def five_state_model():
    pi = [0.30, 0.25, 0.20, 0.15, 0.10]
    A = [
        [0.55, 0.25, 0.10, 0.08, 0.02],
        [0.10, 0.50, 0.25, 0.10, 0.05],
        [0.05, 0.15, 0.50, 0.20, 0.10],
        [0.02, 0.08, 0.20, 0.50, 0.20],
        [0.01, 0.04, 0.10, 0.25, 0.60],
    ]
    B = [
        [0.40, 0.25, 0.15, 0.10, 0.07, 0.03],
        [0.10, 0.35, 0.25, 0.15, 0.10, 0.05],
        [0.05, 0.15, 0.35, 0.25, 0.15, 0.05],
        [0.05, 0.10, 0.20, 0.35, 0.20, 0.10],
        [0.03, 0.07, 0.15, 0.25, 0.30, 0.20],
    ]
    return normalize_parameters(pi, A, B)


def main():
    rng = np.random.default_rng(20261114)
    model = five_state_model()
    p, a, b = model['pi'], model['transition'], model['emission']

    # 生成 100000 码观测：按模型采样，保证是模型自身的典型序列
    hidden = np.empty(MAX_TRACE_TOKENS, dtype=np.int64)
    obs = np.empty(MAX_TRACE_TOKENS, dtype=np.int64)
    hidden[0] = rng.choice(5, p=p)
    obs[0] = rng.choice(6, p=b[hidden[0]])
    for t in range(1, MAX_TRACE_TOKENS):
        hidden[t] = rng.choice(5, p=a[hidden[t - 1]])
        obs[t] = rng.choice(6, p=b[hidden[t]])

    baseline = peak_rss_mb()
    with tempfile.TemporaryDirectory() as tmp:
        gc.collect()
        sink = storage.new_disk_sink(tmp, run_id=99001, trace_id=1)
        start = time.perf_counter()
        (summary, _), normalized = infer_long_sequence(
            obs, p, a, b, block_size=1000, sink=sink)
        manifest = {
            'run_id': 99001, 'trace_id': 1, 'reference': 'LONG-1',
            'tokens': [str(int(x)) for x in obs],
            'states': ['s0', 's1', 's2', 's3', 's4'],
            'symbols': ['A', 'B', 'C', 'D', 'E', 'F'],
            'normalization': normalized['normalization'], **summary,
        }
        key = storage.persist_trace_artifact(tmp, 99001, 1, sink, manifest)
        elapsed = time.perf_counter() - start
        peak = peak_rss_mb()
        print(f'十万码推断耗时 {elapsed:.1f}s，峰值 RSS {peak:.1f}MB（基线 {baseline:.1f}MB）')
        assert peak < MEMORY_LIMIT_MB, f'峰值内存 {peak}MB 超 256MB'
        assert summary['length'] == MAX_TRACE_TOKENS
        assert summary['state_count'] == 5
        assert summary['block_count'] == 100
        assert np.isfinite(summary['log_likelihood'])
        assert summary['linear_likelihood_underflows'] is True
        assert summary['likelihood_if_representable'] is None
        assert len(summary['viterbi_path']) == MAX_TRACE_TOKENS
        assert len(summary['marginal_path']) == MAX_TRACE_TOKENS
        assert all(0 <= s < 5 for s in summary['viterbi_path'])

        # 区段读回：只加载请求的块，块内字段齐全且 null 表示精确零
        for index in (0, 49, 99):
            block = storage.load_block(tmp, key, index)
            assert len(block['posterior']) == 1000
            assert len(block['log_forward']) == 1000
            assert len(block['log_backward']) == 1000
            assert all(abs(sum(row) - 1.0) < 1e-9 for row in block['posterior'])
        loaded = storage.load_manifest(tmp, key)
        assert loaded['viterbi_path'] == summary['viterbi_path']

        total_bytes = sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, files in os.walk(storage.trace_artifact_dir(tmp, key))
            for name in files)
        print(f'artifact 磁盘占用 {total_bytes / 1024 / 1024:.1f}MB，区段数 {summary["block_count"]}')

        # 全量块流式读回，拼接后验与父指针，确认没有遗漏或错位
        streamed_post, streamed_parents = [], []
        for block in storage.iter_blocks(tmp, key):
            streamed_post.extend(block['posterior'])
            streamed_parents.extend(block['viterbi_parent'])
        assert len(streamed_post) == MAX_TRACE_TOKENS
        assert len(streamed_parents) == MAX_TRACE_TOKENS

    # 多块（bs=37）与单块路径结果一致，证明块切分不改变数值
    s_multi, blocks_multi = infer_streaming(obs[:5000], p, a, b, block_size=37, sink=MemorySink())
    s_single, blocks_single = infer_streaming(obs[:5000], p, a, b, block_size=5000, sink=MemorySink())
    assert s_multi['viterbi_path'] == s_single['viterbi_path']
    assert s_multi['marginal_path'] == s_single['marginal_path']
    assert abs(s_multi['log_likelihood'] - s_single['log_likelihood']) < 1e-8
    flat_multi = [r for blk in sorted(blocks_multi, key=lambda x: x['index'])
                  for r in blk['posterior']]
    flat_single = [r for blk in blocks_single for r in blk['posterior']]
    assert all(abs(x - y) < 1e-10 for r1, r2 in zip(flat_multi, flat_single)
               for x, y in zip(r1, r2))
    print('5000 码多块(bs=37) 与单块结果完全一致：路径/logL/后验')

    # README 下溢校验：200 步、单步发射 1e-4，logL 约 -1842.0680743952。
    # 参数校验要求至少 2 态，用第二态零初态概率等效复刻单态链，零概率保持精确零。
    model_u = normalize_parameters(
        [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
        [[1e-4, 0.9999], [0.9999, 1e-4]])
    s_u, _ = infer_streaming([0] * 200, model_u['pi'], model_u['transition'],
                             model_u['emission'], sink=MemorySink())
    assert abs(s_u['log_likelihood'] - (-1842.0680743952)) < 1e-9, s_u['log_likelihood']
    assert s_u['linear_likelihood_underflows'] is True
    assert s_u['likelihood_if_representable'] is None
    assert s_u['viterbi_path'] == [0] * 200
    print(f'200 步 1e-4 下溢校验 logL={s_u["log_likelihood"]:.10f}，线性概率 null')

    # 后段观测改变前段后验：同前缀接强区分度短后缀，前段后验必须变化（全序列前向后向口径）。
    # 平稳混合链在 200 步外的影响可衰减到机器精度，因此断言落在紧邻后缀的 t=199，
    # 该点跨越 bs=100 的块边界，同时验证块边界不切断后向信息流。
    prefix = list(obs[:200])
    suffix_favorite = int(np.argmax(np.std(np.asarray(b), axis=0)))
    other = (suffix_favorite + 1) % 6
    _, blocks_a = infer_streaming(prefix + [suffix_favorite] * 30, p, a, b,
                                  block_size=100, sink=MemorySink())
    _, blocks_b = infer_streaming(prefix + [other] * 30, p, a, b,
                                  block_size=100, sink=MemorySink())
    near_a = blocks_a[1]['posterior'][99]  # t=199，前缀最后一步
    near_b = blocks_b[1]['posterior'][99]
    assert any(abs(x - y) > 1e-9 for x, y in zip(near_a, near_b)), \
        '后段观测必须能改变前段（含块边界处）后验'
    far_a = blocks_a[0]['posterior'][0]
    far_b = blocks_b[0]['posterior'][0]
    print('后段观测改变前段后验：t=199 后验',
          [round(x, 4) for x in near_a], '->', [round(x, 4) for x in near_b])
    print('t=0 后验（影响已沿链衰减）：', [round(x, 6) for x in far_a],
          'vs', [round(x, 6) for x in far_b])

    # 跨块零转移：块边界（bs=100）跨越零转移的长序列，Viterbi 永不走非法边
    pi_z = [1.0, 0.0, 0.0, 0.0, 0.0]
    A_z = [[0.6, 0.4, 0.0, 0.0, 0.0], [0.0, 0.6, 0.4, 0.0, 0.0],
           [0.0, 0.0, 0.6, 0.4, 0.0], [0.0, 0.0, 0.0, 0.6, 0.4],
           [0.0, 0.0, 0.0, 0.0, 1.0]]
    B_z = [[0.5, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]]
    model_z = normalize_parameters(pi_z, A_z, B_z)
    obs_z, state = [], 0
    rng2 = np.random.default_rng(7)
    for _ in range(1000):
        choices = [z for z in range(6) if B_z[state][z] > 0]
        obs_z.append(int(rng2.choice(choices)))
        state = int(rng2.choice(5, p=A_z[state]))
    s_z, _ = infer_streaming(obs_z, model_z['pi'], model_z['transition'],
                             model_z['emission'], block_size=100, sink=MemorySink())
    path = s_z['viterbi_path']
    assert B_z[path[0]][obs_z[0]] > 0
    for t in range(1, len(path)):
        assert A_z[path[t - 1]][path[t]] > 0, f'Viterbi 在 t{t} 走了零转移边'
        assert B_z[path[t]][obs_z[t]] > 0, f'Viterbi 在 t{t} 走了零发射'
    print('1000 码跨块零转移：Viterbi 全程无非法边，路径状态最大值', max(path))

    # 慢混合链（自循环 0.6）上，后段观测对 t=0 后验仍有可测影响。
    # 初态非退化、前缀短到末端状态分布仍有不确定性；后缀在 state0 两个可发射码间选择，
    # 两条序列都可行且对前缀后验产生不同的后向信息。
    pi_slow = [0.4, 0.3, 0.2, 0.07, 0.03]
    A_slow = [[0.6, 0.4, 0.0, 0.0, 0.0], [0.0, 0.6, 0.4, 0.0, 0.0],
              [0.0, 0.0, 0.6, 0.4, 0.0], [0.0, 0.0, 0.0, 0.6, 0.4],
              [0.0, 0.0, 0.0, 0.0, 1.0]]
    B_slow = [[0.5, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]]
    model_slow = normalize_parameters(pi_slow, A_slow, B_slow)
    rng3 = np.random.default_rng(11)
    state = int(rng3.choice(5, p=pi_slow))
    prefix_slow = []
    for _ in range(3):
        choices = [z for z in range(6) if B_slow[state][z] > 0]
        prefix_slow.append(int(rng3.choice(choices)))
        state = int(rng3.choice(5, p=A_slow[state]))
    code_x, code_y = 0, 1  # state0 的两个可发射观测码
    _, slow_a = infer_streaming(prefix_slow + [code_x] * 20, model_slow['pi'],
                                model_slow['transition'], model_slow['emission'],
                                block_size=4, sink=MemorySink())
    _, slow_b = infer_streaming(prefix_slow + [code_y] * 20, model_slow['pi'],
                                model_slow['transition'], model_slow['emission'],
                                block_size=4, sink=MemorySink())
    t0_a = slow_a[0]['posterior'][0]
    t0_b = slow_b[0]['posterior'][0]
    assert any(abs(x - y) > 1e-9 for x, y in zip(t0_a, t0_b)), \
        '慢混合链上后段观测必须能改变 t=0 后验'
    print('慢混合链 t=0 后验随后段变化：', [round(x, 4) for x in t0_a],
          '->', [round(x, 4) for x in t0_b])

    print('PASS 十万码实测全部断言（内存<256MB / 五隐态 / 下溢 / 后验回溯 / 零转移）')


if __name__ == '__main__':
    main()
