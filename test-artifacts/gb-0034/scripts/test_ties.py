"""平局规则专项：构造精确浮点 log 分数平局，对照旧引擎与新流式引擎。

旧引擎保留完整 tuple 路径并按 min(paths) 决胜，新引擎用 (前驱 index, 前缀秩) 维护
同一字典序；本脚本用对称参数制造多步、跨块的精确平局，逐路径比对两条实现。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend')))
sys.path.insert(0, '/tmp/oldcheck')

from src.services.hmm import infer_sequence as new_infer  # noqa: E402
from src.services.engine import infer_streaming, MemorySink  # noqa: E402
from svc.hmm import infer_sequence as old_infer  # noqa: E402


def main():
    # 完全对称的二态模型：任意时刻两态对同一观测的发射相同，
    # 转移矩阵对称，使大量前缀路径 log 分数精确相等，平局必须取字典序最小。
    pi = [0.5, 0.5]
    A = [[0.5, 0.5], [0.5, 0.5]]
    B = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    for t_total in (1, 2, 3, 5, 8, 13, 21):
        obs = [t_total % 4 for _ in range(t_total)]
        old = old_infer(obs, pi, A, B)
        new = new_infer(obs, pi, A, B)
        assert old['viterbi_path'] == new['viterbi_path'], (t_total, old, new)
        assert old['marginal_path'] == new['marginal_path'], t_total
        assert abs(old['viterbi_log_joint'] - new['viterbi_log_joint']) < 1e-12
        # 全字典序最小路径即全 0
        assert new['viterbi_path'] == [0] * t_total, (t_total, new['viterbi_path'])
        for bs in (1, 2, 3, 7, 100):
            summary, _ = infer_streaming(obs, pi, A, B, block_size=bs, sink=MemorySink())
            assert summary['viterbi_path'] == [0] * t_total, (t_total, bs)
        print(f't={t_total:2d} 对称平局 Viterbi={new["viterbi_path"]} 与旧引擎一致')

    # 三态对称：平局路径更多，字典序最小应恒为全 0
    pi3 = [1 / 3, 1 / 3, 1 / 3]
    A3 = [[1 / 3] * 3 for _ in range(3)]
    B3 = [[0.2, 0.2, 0.2, 0.2, 0.2] for _ in range(3)]
    for t_total in (1, 4, 9, 16):
        obs = [0] * t_total
        old = old_infer(obs, pi3, A3, B3)
        new = new_infer(obs, pi3, A3, B3)
        assert old['viterbi_path'] == new['viterbi_path'] == [0] * t_total
        assert old['marginal_path'] == new['marginal_path'] == [0] * t_total
        print(f'三态 t={t_total:2d} 平局 Viterbi={new["viterbi_path"]} 与旧引擎一致')

    # 非对称但局部平局：前两步对称，第三步出现带平局的分叉
    pi2 = [0.5, 0.5]
    A2 = [[0.5, 0.5], [0.5, 0.5]]
    B2 = [[0.4, 0.6], [0.4, 0.6]]
    old = old_infer([0, 1, 0, 1], pi2, A2, B2)
    new = new_infer([0, 1, 0, 1], pi2, A2, B2)
    assert old['viterbi_path'] == new['viterbi_path']
    assert old['posterior'] == new['posterior'] or all(
        abs(x - y) < 1e-12 for r1, r2 in zip(old['posterior'], new['posterior'])
        for x, y in zip(r1, r2))
    print('局部平局分叉路径与旧引擎一致：', new['viterbi_path'])
    print('PASS 平局规则专项：对称多态、跨块、局部平局全部与旧 tuple 决胜一致')


if __name__ == '__main__':
    main()
