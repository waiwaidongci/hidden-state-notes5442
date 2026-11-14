"""服务层集成：分段 artifact 存储、analyze_segmented、冻结快照、CSV 导出与旧内联兼容。

不依赖 MySQL/Redis，直接调用 service 函数与临时 artifact 目录。
"""
import csv
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend')))

from src.services.analysis import analyze, analyze_segmented  # noqa: E402
from src.services import storage  # noqa: E402
from src.services.engine import ImpossibleSequence  # noqa: E402
from src.controllers import export as export_controller  # noqa: E402
from src.controllers.export import _segmented_rows, _inline_rows, HEADERS  # noqa: E402

PROFILE = {
    'name': '五态长序列模型', 'version': 'v1',
    'states': ['s0', 's1', 's2', 's3', 's4'],
    'symbols': ['A', 'B', 'C', 'D', 'E', 'F'],
    'parameters': {
        'pi': [0.3, 0.25, 0.2, 0.15, 0.1],
        'transition': [[0.55, 0.25, 0.1, 0.08, 0.02], [0.1, 0.5, 0.25, 0.1, 0.05],
                       [0.05, 0.15, 0.5, 0.2, 0.1], [0.02, 0.08, 0.2, 0.5, 0.2],
                       [0.01, 0.04, 0.1, 0.25, 0.6]],
        'emission': [[0.4, 0.25, 0.15, 0.1, 0.07, 0.03], [0.1, 0.35, 0.25, 0.15, 0.1, 0.05],
                     [0.05, 0.15, 0.35, 0.25, 0.15, 0.05], [0.05, 0.1, 0.2, 0.35, 0.2, 0.1],
                     [0.03, 0.07, 0.15, 0.25, 0.3, 0.2]],
    },
}


class FakeNote:
    def __init__(self, name, interpretation, next_step, review_note):
        self.name = name
        self.interpretation = interpretation
        self.next_step = next_step
        self.review_note = review_note


def make_trace(tid, reference, length, symbols):
    return {'id': tid, 'reference': reference,
            'tokens': [symbols[(tid * 7 + t * 3) % len(symbols)] for t in range(length)]}


def main():
    symbols = PROFILE['symbols']
    with tempfile.TemporaryDirectory() as tmp:
        # 导出控制器从 settings 取 artifact 根目录，测试中指向同一临时目录
        export_controller.ARTIFACT_DIR = tmp
        # 长轨迹走分段路径：一条 2500 码、一条 1001 码
        long_traces = [make_trace(1, 'LONG-001', 2500, symbols),
                       make_trace(2, 'LONG-002', 1001, symbols)]
        result = analyze_segmented(77001, PROFILE, long_traces, tmp, block_size=500)
        assert result['result_schema'] == 'segments-v1'
        assert result['trace_count'] == 2
        assert result['token_count'] == 3501
        assert len(result['traces']) == 2
        entry = result['traces'][0]
        assert entry['block_count'] == 5
        assert entry['length'] == 2500
        assert storage.artifact_exists(tmp, entry['artifact_key'])

        # 区段 API 等价读取：manifest 与逐块
        manifest = storage.load_manifest(tmp, entry['artifact_key'])
        assert manifest['length'] == 2500
        assert len(manifest['viterbi_path']) == 2500
        block0 = storage.load_block(tmp, entry['artifact_key'], 0)
        block4 = storage.load_block(tmp, entry['artifact_key'], 4)
        assert block0['start'] == 0 and block0['size'] == 500
        assert block4['start'] == 2000 and block4['size'] == 500
        total = sum(bl['size'] for bl in storage.iter_blocks(tmp, entry['artifact_key']))
        assert total == 2500
        # 末块可能不足 block_size
        entry2 = result['traces'][1]
        assert entry2['block_count'] == 3  # 500,500,1
        last = storage.load_block(tmp, entry2['artifact_key'], 2)
        assert last['size'] == 1

        # 不可能序列：整个任务失败，不产生半成品 artifact
        bad_profile = {
            **PROFILE,
            'parameters': {**PROFILE['parameters'],
                           'pi': [1.0, 0.0, 0.0, 0.0, 0.0],
                           'emission': [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
                                      + PROFILE['parameters']['emission'][1:]},
        }
        bad_trace = [{'id': 9, 'reference': 'BAD', 'tokens': ['A'] * 10}]
        failed = False
        try:
            analyze_segmented(77002, bad_profile, bad_trace, tmp, block_size=4)
        except ImpossibleSequence:
            failed = True
        assert failed
        assert not storage.artifact_exists(tmp, 'runs/77002/trace-9')

        # 冻结输入：修改源轨迹后，artifact 明细与结果不变（快照在调用前已固定）
        frozen_tokens = list(long_traces[0]['tokens'])
        long_traces[0]['tokens'] = ['F'] * 2500
        block0_again = storage.load_block(tmp, entry['artifact_key'], 0)
        assert block0_again == block0
        assert manifest['tokens'] == frozen_tokens

        # 分段 CSV 流式导出：行数 = 总码数，且不一次性载入全序列
        note = FakeNote('长序列批准说明', '解释', '下一步', '审核意见')
        rows = list(_segmented_rows(note, result))
        assert len(rows) == 3501
        assert len(rows[0]) == len(HEADERS)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(HEADERS)
        writer.writerows(rows)
        text = buffer.getvalue()
        assert 'LONG-001' in text and 's0' in text or 's4' in text

    # 旧内联路径（<=200 码）结构保持：键集合与历史版本一致
    short = [make_trace(1, 'SHORT-001', 40, symbols),
             make_trace(2, 'SHORT-002', 200, symbols)]
    inline = analyze(PROFILE, short)
    assert inline['result_schema'] == 'inline-v1'
    expected_keys = {
        'log_likelihood', 'likelihood_if_representable', 'linear_likelihood_underflows',
        'posterior', 'marginal_path', 'marginal_path_is_legal', 'marginal_log_joint',
        'viterbi_path', 'viterbi_log_joint', 'log_forward', 'log_backward',
        'null_log_meaning', 'tie_rule', 'normalization',
    }
    trace = inline['traces'][0]
    assert expected_keys <= set(trace.keys())
    assert len(trace['posterior']) == 40 and len(trace['log_forward']) == 40
    note = FakeNote('旧短轨迹说明', '解释', '下一步', '审核意见')
    inline_rows = list(_inline_rows(note, inline))
    assert len(inline_rows) == 240
    print('PASS 服务层集成：分段存储/区段读取/不可能失败/冻结快照/CSV/旧内联结构全部正确')


if __name__ == '__main__':
    main()
