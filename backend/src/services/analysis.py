import math
import shutil
from pathlib import Path
from .hmm import infer_sequence, infer_long_sequence, ImpossibleSequence
from .validation import validate_traces, validate_states
from .fingerprint import fingerprint
from . import storage

MAX_RUN_TRACES = 30


def _ordered(traces):
    return sorted(traces, key=lambda t: t['reference'])


def analyze(profile, traces):
    """短轨迹内存路径（每条 <=200 码）：完整结果内联，结构与历史版本完全一致。"""
    if not 1 <= len(traces) <= MAX_RUN_TRACES:
        raise ValueError(f'单次须选择1..{MAX_RUN_TRACES}条轨迹')
    validate_traces(profile['symbols'], traces)
    parameters = profile['parameters']
    validate_states(profile['states'], len(parameters['pi']))
    lookup = {symbol: i for i, symbol in enumerate(profile['symbols'])}
    results = []
    for trace in _ordered(traces):
        if len(trace['tokens']) > storage.LEGACY_INLINE_MAX:
            raise ValueError('长轨迹须走分段推断路径')
        try:
            inferred = infer_sequence(
                [lookup[s] for s in trace['tokens']],
                parameters['pi'], parameters['transition'], parameters['emission'])
        except ImpossibleSequence as error:
            raise ImpossibleSequence(f'{trace["reference"]}：{error}') from error
        results.append({'id': trace['id'], 'reference': trace['reference'],
                        'tokens': trace['tokens'], **inferred})
    return {'profile': profile, 'states': profile['states'], 'symbols': profile['symbols'],
            'trace_count': len(results),
            'token_count': sum(len(r['tokens']) for r in results),
            'total_log_likelihood': math.fsum(r['log_likelihood'] for r in results),
            'illegal_marginal_paths': sum(not r['marginal_path_is_legal'] for r in results),
            'traces': results, 'fingerprint': fingerprint(profile, traces),
            'result_schema': 'inline-v1',
            'interpretation': 'Fixed HMM parameters, no EM or fitting. Marginal argmax can violate legal transitions; Viterbi is the full joint MAP path. States are model labels, not equipment-fault facts or online alarms.'}


def analyze_segmented(run_id, profile, traces, artifact_root, block_size=None):
    """分段路径：整序列一次推断，明细按区段写入 artifact，数据库只保存摘要。

    分块只是存储与查看单位；前向/后向/Viterbi 都在完整序列上递推，
    块边界沿用同一个全局转移矩阵，不存在跨段零转移被忽略或分段各算各的问题。
    """
    if not 1 <= len(traces) <= MAX_RUN_TRACES:
        raise ValueError(f'单次须选择1..{MAX_RUN_TRACES}条轨迹')
    validate_traces(profile['symbols'], traces)
    parameters = profile['parameters']
    validate_states(profile['states'], len(parameters['pi']))
    lookup = {symbol: i for i, symbol in enumerate(profile['symbols'])}
    trace_summaries = []
    log_likelihoods = []
    illegal = 0
    run_dir = Path(artifact_root) / storage.ARTIFACT_RUN_PREFIX / str(run_id)
    try:
        for trace in _ordered(traces):
            sink = storage.new_disk_sink(artifact_root, run_id, trace['id'])
            try:
                (summary, _), normalized = infer_long_sequence(
                    [lookup[s] for s in trace['tokens']],
                    parameters['pi'], parameters['transition'], parameters['emission'],
                    block_size=block_size or 1000, sink=sink)
            except ImpossibleSequence as error:
                raise ImpossibleSequence(f'{trace["reference"]}：{error}') from error
            manifest = {
                'run_id': run_id, 'trace_id': trace['id'], 'reference': trace['reference'],
                'tokens': trace['tokens'], 'states': profile['states'],
                'symbols': profile['symbols'], 'normalization': normalized['normalization'],
                **summary,
            }
            artifact_key = storage.persist_trace_artifact(
                artifact_root, run_id, trace['id'], sink, manifest)
            log_likelihoods.append(summary['log_likelihood'])
            illegal += 0 if summary['marginal_path_is_legal'] else 1
            trace_summaries.append({
                'id': trace['id'], 'reference': trace['reference'],
                'length': summary['length'], 'artifact_key': artifact_key,
                'log_likelihood': summary['log_likelihood'],
                'likelihood_if_representable': summary['likelihood_if_representable'],
                'linear_likelihood_underflows': summary['linear_likelihood_underflows'],
                'marginal_path_is_legal': summary['marginal_path_is_legal'],
                'marginal_log_joint': summary['marginal_log_joint'],
                'viterbi_log_joint': summary['viterbi_log_joint'],
                'block_count': summary['block_count'], 'block_size': summary['block_size'],
                'blocks': summary['blocks'],
                'null_log_meaning': summary['null_log_meaning'], 'tie_rule': summary['tie_rule'],
            })
    except Exception:
        # 任务失败不保留半成品 artifact，避免区段 API 读到不完整明细
        if run_dir.exists():
            shutil.rmtree(run_dir)
        raise
    return {'profile': profile, 'states': profile['states'], 'symbols': profile['symbols'],
            'trace_count': len(trace_summaries),
            'token_count': sum(item['length'] for item in trace_summaries),
            'total_log_likelihood': math.fsum(log_likelihoods),
            'illegal_marginal_paths': illegal, 'traces': trace_summaries,
            'fingerprint': fingerprint(profile, traces),
            'result_schema': 'segments-v1',
            'interpretation': 'Fixed HMM parameters, no EM or fitting. Marginal argmax can violate legal transitions; Viterbi is the full joint MAP path. States are model labels, not equipment-fault facts or online alarms. Full per-timestep detail is stored in run-scoped segment artifacts, never inferred per segment independently.'}
