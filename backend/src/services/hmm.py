"""固定 HMM 推断的兼容入口。

短轨迹（<=200 码）沿用本模块历史输出结构：log_forward/log_backward/posterior 等
完整数组随结果内联返回，旧运行、旧批准说明的快照结构因此保持可读。
长轨迹（至多 100000 码）走 services.engine 的流式两遍法与 services.storage 的分块 artifact，
两套路径共享同一数值口径与平局规则。
"""
from .parameters import normalize_parameters
from .engine import (
    infer_streaming,
    ImpossibleSequence,
    MemorySink,
    MAX_TRACE_TOKENS,
    DEFAULT_BLOCK_SIZE,
)

__all__ = [
    'infer_sequence', 'infer_long_sequence', 'ImpossibleSequence',
    'MAX_TRACE_TOKENS', 'DEFAULT_BLOCK_SIZE',
]

LEGACY_OBSERVATION_LIMIT = 200


def infer_sequence(observations, pi, transition, emission):
    """旧内存路径：输出字段与历史版本逐键兼容，供 <=200 码轨迹与旧快照使用。"""
    normalized = normalize_parameters(pi, transition, emission)
    p = normalized['pi']
    a = normalized['transition']
    b = normalized['emission']
    obs = list(observations)
    if not 1 <= len(obs) <= LEGACY_OBSERVATION_LIMIT:
        raise ValueError('观测须1..200个合法整数码索引')
    summary, blocks = infer_streaming(obs, p, a, b,
                                      block_size=min(DEFAULT_BLOCK_SIZE, len(obs)),
                                      sink=MemorySink())
    ordered = sorted(blocks, key=lambda blk: blk['index'])
    return {
        'log_likelihood': summary['log_likelihood'],
        'likelihood_if_representable': summary['likelihood_if_representable'],
        'linear_likelihood_underflows': summary['linear_likelihood_underflows'],
        'posterior': [row for blk in ordered for row in blk['posterior']],
        'marginal_path': summary['marginal_path'],
        'marginal_path_is_legal': summary['marginal_path_is_legal'],
        'marginal_log_joint': summary['marginal_log_joint'],
        'viterbi_path': summary['viterbi_path'],
        'viterbi_log_joint': summary['viterbi_log_joint'],
        'log_forward': [row for blk in ordered for row in blk['log_forward']],
        'log_backward': [row for blk in ordered for row in blk['log_backward']],
        'null_log_meaning': summary['null_log_meaning'],
        'tie_rule': summary['tie_rule'],
        'normalization': normalized['normalization'],
    }


def infer_long_sequence(observations, pi, transition, emission, block_size=DEFAULT_BLOCK_SIZE, sink=None):
    """长轨迹流式路径：返回 (summary, sink)，明细按块经 sink 落盘，不内联大数组。"""
    normalized = normalize_parameters(pi, transition, emission)
    return infer_streaming(
        list(observations), normalized['pi'], normalized['transition'], normalized['emission'],
        block_size=block_size, sink=sink,
    ), normalized
