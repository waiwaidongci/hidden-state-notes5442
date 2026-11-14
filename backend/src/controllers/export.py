import csv
import io
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from src.models import InterpretationNote
from src.config.settings import ARTIFACT_DIR
from src.services import storage
from src.services.audit import record

HEADERS = ['核对说明', '固定模型', '版本', '指纹', '轨迹', '步号', '观测码', 'Viterbi联合MAP',
           '单点最高态', '边际后验向量', '序列logL', 'Viterbi联合logP', '边际路径合法',
           '人工解释', '下一步', '审核意见']


def _sanitize(value):
    return ("'" + value) if isinstance(value, str) and value.startswith(('=', '+', '-', '@')) else value


def _row_values(note, profile, fingerprint, reference, states, step, token, viterbi_state,
                marginal_state, posterior, log_likelihood, viterbi_log_joint, legal):
    return [note.name, profile['name'], profile['version'], fingerprint, reference, step + 1,
            token, states[viterbi_state], states[marginal_state], str(posterior),
            log_likelihood, viterbi_log_joint, legal, note.interpretation, note.next_step,
            note.review_note]


def _inline_rows(note, result):
    for trace in result['traces']:
        for t, token in enumerate(trace['tokens']):
            yield _row_values(
                note, result['profile'], result['fingerprint'], trace['reference'],
                result['states'], t, token, trace['viterbi_path'][t],
                trace['marginal_path'][t], trace['posterior'][t],
                trace['log_likelihood'], trace['viterbi_log_joint'],
                trace['marginal_path_is_legal'])


def _segmented_rows(note, result):
    """分段结果逐块流式读取 artifact，导出进程不持有整条序列明细。"""
    for trace in result['traces']:
        manifest = storage.load_manifest(ARTIFACT_DIR, trace['artifact_key'])
        offset = 0
        for block in storage.iter_blocks(ARTIFACT_DIR, trace['artifact_key']):
            for u in range(block['size']):
                t = offset + u
                yield _row_values(
                    note, result['profile'], result['fingerprint'], manifest['reference'],
                    result['states'], t, manifest['tokens'][t],
                    manifest['viterbi_path'][t], manifest['marginal_path'][t],
                    block['posterior'][u], manifest['log_likelihood'],
                    manifest['viterbi_log_joint'], manifest['marginal_path_is_legal'])
            offset += block['size']


def export_note(db, id, actor):
    note = db.get(InterpretationNote, id)
    if not note:
        raise HTTPException(404, '说明不存在')
    if note.status != 'approved':
        raise HTTPException(409, '仅批准说明可导出')
    result = note.result_snapshot

    def stream():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(HEADERS)
        yield '﻿' + buffer.getvalue()
        rows = (_segmented_rows(note, result) if result.get('result_schema') == 'segments-v1'
                else _inline_rows(note, result))
        for values in rows:
            buffer = io.StringIO()
            csv.writer(buffer).writerow([_sanitize(v) for v in values])
            yield buffer.getvalue()

    record(db, actor, 'note.exported', {'note_id': id})
    return StreamingResponse(
        stream(), media_type='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename="hmm-note-{id}.csv"'})
