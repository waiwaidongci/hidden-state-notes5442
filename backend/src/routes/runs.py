from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.config.database import get_db
from src.config.settings import ARTIFACT_DIR as ARTIFACT_ROOT
from src.controllers.dependencies import current_user, writers
from src.models import InferenceRun
from src.schemas.run import RunInput
from src.services.runs import create_run
from src.services import storage
from src.utils.serialize import row
router = APIRouter()


def _trace_entry(run, trace_id):
    if run.result is None or run.result.get('result_schema') != 'segments-v1':
        raise HTTPException(409, '该运行不是分段结果，完整结果已随运行内联返回')
    entry = next((t for t in run.result['traces'] if t['id'] == trace_id), None)
    if entry is None:
        raise HTTPException(404, '轨迹不属于该运行')
    return entry


@router.get('/runs')
def runs(db: Session = Depends(get_db), user=Depends(current_user)):
    return [row(r) for r in db.scalars(select(InferenceRun).order_by(InferenceRun.id.desc()))]


@router.get('/runs/{id}')
def one(id: int, db: Session = Depends(get_db), user=Depends(current_user)):
    r = db.get(InferenceRun, id)
    if not r:
        raise HTTPException(404, '运行不存在')
    return row(r)


@router.get('/runs/{id}/traces/{trace_id}/segments')
def segments(id: int, trace_id: int, db: Session = Depends(get_db), user=Depends(current_user)):
    run = db.get(InferenceRun, id)
    if not run:
        raise HTTPException(404, '运行不存在')
    entry = _trace_entry(run, trace_id)
    manifest = storage.load_manifest(ARTIFACT_ROOT, entry['artifact_key'])
    return {'run_id': id, 'trace_id': trace_id, 'reference': entry['reference'],
            'length': manifest['length'], 'block_size': manifest['block_size'],
            'block_count': manifest['block_count'], 'blocks': manifest['blocks'],
            'marginal_path': manifest['marginal_path'],
            'viterbi_path': manifest['viterbi_path'],
            'marginal_path_is_legal': manifest['marginal_path_is_legal'],
            'marginal_log_joint': manifest['marginal_log_joint'],
            'viterbi_log_joint': manifest['viterbi_log_joint'],
            'log_likelihood': manifest['log_likelihood'],
            'likelihood_if_representable': manifest['likelihood_if_representable'],
            'linear_likelihood_underflows': manifest['linear_likelihood_underflows'],
            'tie_rule': manifest['tie_rule'], 'null_log_meaning': manifest['null_log_meaning']}


@router.get('/runs/{id}/traces/{trace_id}/segments/{index}')
def segment(id: int, trace_id: int, index: int, db: Session = Depends(get_db),
            user=Depends(current_user)):
    run = db.get(InferenceRun, id)
    if not run:
        raise HTTPException(404, '运行不存在')
    entry = _trace_entry(run, trace_id)
    if index < 0 or index >= entry['block_count']:
        raise HTTPException(404, '区段序号超出范围')
    block = storage.load_block(ARTIFACT_ROOT, entry['artifact_key'], index)
    manifest = storage.load_manifest(ARTIFACT_ROOT, entry['artifact_key'])
    block['tokens'] = manifest['tokens'][block['start']:block['end']]
    return {'run_id': id, 'trace_id': trace_id, **block}


@router.post('/runs')
def create(data: RunInput, db: Session = Depends(get_db), user=Depends(writers)):
    return create_run(db, data, user.email)
