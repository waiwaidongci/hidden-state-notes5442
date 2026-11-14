"""固定 HMM 的流式 log 域推断引擎。

支持单条轨迹至多 MAX_TRACE_TOKENS 个观测码、2..5 个隐态。
数值口径与 services/hmm.py 的旧内存实现完全一致：
- log 域前向/后向 + logsumexp，真实零概率保持 -inf（JSON 中为 null），不设概率地板；
- log 域 Viterbi，父指针回溯整条联合最优路径；
- 精确 log 分数平局取状态索引序列字典序最小者，边际平局取最小状态 index；
- 线性概率下溢时 logL 仍有效，线性概率字段为 None 并单独标记。

与旧实现的唯一工程差异是分块（block）存储：alpha/beta/父指针按块写入 sink，
推断本身始终针对整条序列一次完成，块边界使用同一个全局转移矩阵，
绝不做分段各算各的。
"""
import json
import math
import numpy as np

MAX_TRACE_TOKENS = 100_000
DEFAULT_BLOCK_SIZE = 1000
NEG_INF = -np.inf


def logsumexp_inf(values, axis):
    """与 scipy.special.logsumexp 同口径的纯 numpy 实现。

    全 -inf 的缩减轴返回 -inf（scipy 会抛 warning 并返回 -inf，这里静默处理）。
    """
    values = np.asarray(values, dtype=float)
    maxima = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maxima)
    safe = np.where(finite, maxima, 0.0)
    with np.errstate(divide='ignore'):
        summed = np.sum(np.exp(values - safe), axis=axis, keepdims=True)
        logged = np.log(np.where(finite, summed, 1.0)) + safe
    kept = np.where(finite, logged, NEG_INF)
    return np.squeeze(kept, axis=axis)


def _log_probabilities(values):
    array = np.asarray(values, dtype=float)
    out = np.full(array.shape, NEG_INF)
    positive = array > 0
    out[positive] = np.log(array[positive])
    return out


def _safe_logsumexp_three(log_a, log_b_row, log_c_row):
    """logsumexp_j(log_a(i,j)+log_b(j)+log_c(j))，三项任一为 -inf 则该项精确为 -inf。

    用零值掩码参与运算，避免 numpy where 两分支都求值时 -inf 运算产生 NaN。
    log_a 为 (K,K)，log_b_row/log_c_row 为 (K,)。
    """
    b = np.broadcast_to(log_b_row[None, :], log_a.shape)
    c = np.broadcast_to(log_c_row[None, :], log_a.shape)
    valid = np.isfinite(log_a) & np.isfinite(b) & np.isfinite(c)
    combo = np.where(valid,
                     np.where(np.isfinite(log_a), log_a, 0.0)
                     + np.where(np.isfinite(b), b, 0.0)
                     + np.where(np.isfinite(c), c, 0.0),
                     NEG_INF)
    return logsumexp_inf(combo, axis=1)


def _json_logs(array):
    return [[float(v) if np.isfinite(v) else None for v in row] for row in array]


def _from_json_logs(rows):
    """JSON 还原：None 明确还原为 -inf（精确零概率），绝不变成 NaN。"""
    return np.asarray(
        [[NEG_INF if value is None else float(value) for value in row] for row in rows],
        dtype=float)


class ImpossibleSequence(ValueError):
    pass


class BlockSink:
    """块输出接口。内存模式返回 list，磁盘模式由 storage 模块实现。"""

    def write(self, block):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError


class MemorySink(BlockSink):
    def __init__(self):
        self.blocks = []

    def write(self, block):
        self.blocks.append(block)

    def finish(self):
        return self.blocks


class DiskBlockSink(BlockSink):
    """逐块落盘的 sink：每块一个 JSON 文件，峰值内存与序列长度无关。

    第一遍顺序写 block-{index}.json；第二遍逆序 read/update 回填 beta 与后验。
    """

    def __init__(self, directory):
        self.directory = directory
        self._paths = []

    def _path(self, index):
        from pathlib import Path
        return Path(self.directory) / f'block-{index}.json'

    def write(self, block):
        index = block['index']
        path = self._path(index)
        with path.open('w', encoding='utf-8') as handle:
            json.dump(block, handle, ensure_ascii=False, allow_nan=False)
        self._paths.append(path)

    def read(self, index):
        with self._path(index).open('r', encoding='utf-8') as handle:
            return json.load(handle)

    def update(self, block):
        with self._path(block['index']).open('w', encoding='utf-8') as handle:
            json.dump(block, handle, ensure_ascii=False, allow_nan=False)

    def finish(self):
        return None


def infer_streaming(observations, pi, transition, emission, block_size=DEFAULT_BLOCK_SIZE, sink=None):
    """对整条观测序列做一次固定模型推断，按块产出明细。

    返回 (summary, blocks)：summary 为轨迹级标量与路径，blocks 为 sink.finish() 的产物。
    """
    p = np.asarray(pi, dtype=float)
    a = np.asarray(transition, dtype=float)
    b = np.asarray(emission, dtype=float)
    obs = np.asarray(observations)
    if obs.ndim != 1 or not np.issubdtype(obs.dtype, np.integer) or np.issubdtype(obs.dtype, np.bool_):
        raise ValueError('观测须为合法整数码索引')
    if not 1 <= len(obs) <= MAX_TRACE_TOKENS:
        raise ValueError(f'观测须1..{MAX_TRACE_TOKENS}个合法整数码索引')
    if (obs < 0).any() or (obs >= b.shape[1]).any():
        raise ValueError('观测码索引超出发射矩阵列范围')
    if block_size < 1 or block_size > MAX_TRACE_TOKENS:
        raise ValueError('块大小非法')
    obs = obs.astype(np.int64)
    t_total = len(obs)
    k_states = len(p)

    lp = _log_probabilities(p)
    la = _log_probabilities(a)
    lb = _log_probabilities(b)

    sink = sink if sink is not None else MemorySink()
    big = np.iinfo(np.int64).max

    # ---- 第一遍：前向 log alpha 与 log Viterbi（父指针 + 前缀秩保字典序平局）----
    alpha_row = lp + lb[:, obs[0]]
    if not np.isfinite(alpha_row).any():
        raise ImpossibleSequence('不可能序列：第1码在固定模型下概率为0')
    score_row = alpha_row.copy()
    # rank[state] 是该状态当前最优前缀路径在所有活路径中的字典序名次（越小越优）。
    # 第 0 步前缀只有单状态，名次即状态 index；死状态名次置最大哨兵。
    rank = np.where(np.isfinite(score_row), np.arange(k_states, dtype=np.int64), big)

    alpha_buf = []
    lb_buf = []
    parent_buf = []
    block_index = 0

    def flush_block(end_exclusive):
        nonlocal block_index
        start = block_index * block_size
        size = end_exclusive - start
        block = {
            'index': block_index,
            'start': start,
            'end': end_exclusive,
            'size': size,
            'log_forward': _json_logs(np.asarray(alpha_buf, dtype=float)),
            'log_emission': _json_logs(np.asarray(lb_buf, dtype=float)),
            'viterbi_parent': np.asarray(parent_buf, dtype=np.int8).tolist(),
        }
        sink.write(block)
        block_index += 1

    alpha_buf.append(alpha_row)
    lb_buf.append(lb[:, obs[0]])
    # 第 0 步没有前驱，父指针统一记 -1
    parent_buf.append(np.full(k_states, -1, dtype=np.int8))

    for t in range(1, t_total):
        # 前向：alpha_t(j)=logB(j,o_t)+logsumexp_i(alpha_{t-1}(i)+logA(i,j))
        alpha_row = logsumexp_inf(alpha_row[:, None] + la, axis=0) + lb[:, obs[t]]
        if not np.isfinite(alpha_row).any():
            raise ImpossibleSequence(f'不可能序列：到第{t + 1}码所有路径概率为0')
        # Viterbi：candidate(i,j)=score(i)+logA(i,j)+logB(j,o_t)
        candidates = score_row[:, None] + la + lb[:, obs[t]][None, :]
        best = np.max(candidates, axis=0)
        finite_best = np.isfinite(best)
        # 完整路径字典序 = (前驱状态 index, 前驱前缀字典序)，与旧实现 tuple 逐条比较等价。
        # K<=5，按列做一次向量化词法 argmin：先以前驱 index、再以前驱 rank 决胜。
        predecessor_index = np.broadcast_to(
            np.arange(k_states, dtype=np.int64)[:, None], (k_states, k_states))
        valid_candidate = np.isfinite(candidates)
        is_best = valid_candidate & np.equal(candidates, best[None, :])
        key_index = np.where(is_best, predecessor_index, big)
        key_rank = np.where(is_best, np.broadcast_to(rank[:, None], (k_states, k_states)), big)
        # 有活候选的列在 key_index 上必小于 big，argmin 即词法最小父状态；死列回退为 0 后再置 -1
        parents = np.argmin(key_index, axis=0).astype(np.int8)
        tie_on_index = np.zeros((k_states, k_states), dtype=bool)
        for j_state in range(k_states):
            tie_on_index[:, j_state] = key_index[:, j_state] == key_index[parents[j_state], j_state]
        rank_among_tie = np.where(tie_on_index, key_rank, big)
        refined = np.argmin(rank_among_tie, axis=0).astype(np.int8)
        parents = np.where(finite_best, refined, -1).astype(np.int8)
        chosen_index_key = np.where(finite_best,
                                    predecessor_index[parents, np.arange(k_states)], big)
        chosen_rank_key = np.where(finite_best,
                                   rank[np.maximum(parents, 0)], big)
        # 活状态的新名次：按本步完整路径（前驱 index, 前驱 rank）词法压缩
        new_rank = np.full(k_states, big, dtype=np.int64)
        live = np.where(finite_best)[0]
        if live.size:
            live_order = np.lexsort((chosen_rank_key[live], chosen_index_key[live]))
            new_rank[live[live_order]] = np.arange(live.size)
        alpha_buf.append(alpha_row)
        lb_buf.append(lb[:, obs[t]])
        parent_buf.append(parents)
        score_row = np.where(finite_best, best, NEG_INF)
        rank = new_rank
        if len(alpha_buf) == block_size:
            flush_block(t + 1)
            alpha_buf, lb_buf, parent_buf = [], [], []

    if alpha_buf:
        flush_block(t_total)
    block_count = block_index

    likelihood = float(logsumexp_inf(_last_forward_row(sink, block_count), axis=0))
    with np.errstate(under='ignore'):
        linear = float(math.exp(likelihood))
    representable = linear > 0

    # ---- 第二遍：逆块后向 log beta，并回填每块边际后验 ----
    next_beta = np.zeros(k_states)  # 全序列末步 log beta_T = 0；块间通过它接续，绝不重置
    marginal_rows = []
    block_summaries = []
    for bi in range(block_count - 1, -1, -1):
        block = sink.read(bi) if hasattr(sink, 'read') else sink.blocks[bi]
        lf = _from_json_logs(block['log_forward'])
        le = _from_json_logs(block['log_emission'])
        size = block['size']
        beta = np.full((size, k_states), NEG_INF)
        # 本块末步 t：beta_t(i)=logsumexp_j A(i,j) B(j,o_{t+1}) beta_{t+1}(j)，
        # o_{t+1} 是下一块首步的发射；首块逆序时 t 即全序列末步，beta 恒为 0。
        if bi == block_count - 1:
            beta[size - 1] = next_beta
        else:
            next_block = (sink.read(bi + 1) if hasattr(sink, 'read') else sink.blocks[bi + 1])
            first_emission_next = _from_json_logs(next_block['log_emission'])[0]
            beta[size - 1] = _safe_logsumexp_three(la, first_emission_next, next_beta)
        for u in range(size - 2, -1, -1):
            beta[u] = _safe_logsumexp_three(la, le[u + 1], beta[u + 1])
        # alpha/beta=-inf 的状态精确后验为 0；where 两分支都会求值，
        # 必须用安全零值避免 -inf - (-inf) 产生 NaN
        safe_lf = np.where(np.isfinite(lf), lf, 0.0)
        safe_beta = np.where(np.isfinite(beta), beta, 0.0)
        log_gamma = np.where(np.isfinite(lf) & np.isfinite(beta),
                             safe_lf + safe_beta - likelihood, NEG_INF)
        posterior = np.exp(log_gamma)
        column_sums = posterior.sum(axis=1, keepdims=True)
        # 每步至少存在一条可行路径（否则前向早已抛 ImpossibleSequence），列和必为正
        posterior = posterior / column_sums
        block['log_backward'] = _json_logs(beta)
        block['posterior'] = posterior.tolist()
        if hasattr(sink, 'update'):
            sink.update(block)
        block_summaries.append({
            'index': block['index'], 'start': block['start'], 'end': block['end'], 'size': block['size'],
        })
        next_beta = beta[0]  # 本块首步 beta，供前一块末步接续
        marginal_rows.append(np.argmax(posterior, axis=1))
    block_summaries.reverse()
    marginal_rows.reverse()
    marginal_path = np.concatenate(marginal_rows).astype(np.int8).tolist()

    # ---- Viterbi 回溯：父指针在块内逐行存储，跨块从块首状态继续 ----
    # 终态先取最大 log 分数，精确平局再按完整路径字典序（rank 最小），与旧实现 score==best 口径一致
    best_final = np.max(score_row)
    if not np.isfinite(best_final):
        raise ImpossibleSequence('不可能序列：Viterbi 无可行路径')
    tied_final = np.where(score_row == best_final)[0]
    terminal = int(tied_final[np.argmin(rank[tied_final])])
    viterbi_log_joint = float(score_row[terminal])
    viterbi = [0] * t_total
    state = terminal
    for bi in range(block_count - 1, -1, -1):
        block = sink.read(bi) if hasattr(sink, 'read') else sink.blocks[bi]
        parents = block['viterbi_parent']
        for u in range(block['size'] - 1, -1, -1):
            t = block['start'] + u
            viterbi[t] = state
            state = parents[u][state]
    # 第 0 步的父指针为 -1，回溯结束时 state 应为 -1
    if state != -1:
        raise ImpossibleSequence('Viterbi 回溯失败：第0步父指针缺失')

    # ---- 边际拼接路径的联合 log 概率（可能违反零转移，此时为 -inf）----
    joint = lp[marginal_path[0]] + lb[marginal_path[0], obs[0]]
    for t in range(1, t_total):
        joint += la[marginal_path[t - 1], marginal_path[t]] + lb[marginal_path[t], obs[t]]
    marginal_joint = float(joint)

    summary = {
        'length': t_total,
        'state_count': k_states,
        'log_likelihood': likelihood,
        'likelihood_if_representable': linear if representable else None,
        'linear_likelihood_underflows': not representable,
        'posterior_column_sums': 'renormalized to 1 per timestep',
        'marginal_path': marginal_path,
        'marginal_path_is_legal': bool(np.isfinite(marginal_joint)),
        'marginal_log_joint': marginal_joint if np.isfinite(marginal_joint) else None,
        'viterbi_path': viterbi,
        'viterbi_log_joint': viterbi_log_joint,
        'block_count': block_count,
        'block_size': block_size,
        'blocks': block_summaries,
        'null_log_meaning': 'null in log arrays means exact zero probability (-infinity), not missing data',
        'tie_rule': 'Exact floating log-score ties choose lexicographically smallest full state-index path; marginal ties choose smallest state index.',
    }
    return summary, sink.finish()


def _last_forward_row(sink, block_count):
    if hasattr(sink, 'read'):
        last = sink.read(block_count - 1)
    else:
        last = sink.blocks[-1]
    return _from_json_logs(last['log_forward'])[-1]
