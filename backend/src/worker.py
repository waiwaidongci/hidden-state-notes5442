import json
from pathlib import Path
from celery import Celery
from src.config.database import SessionLocal
from src.config.settings import REDIS_URL, ARTIFACT_DIR
from src.models import InferenceRun
from src.services.analysis import analyze, analyze_segmented
from src.services import storage
from src.services.audit import record
app = Celery('hidden_state_notes', broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(task_track_started=True, worker_prefetch_multiplier=1, task_acks_late=True)


@app.task(name='hmm.analyze', soft_time_limit=900, time_limit=960)
def analyze_run(id):
    with SessionLocal() as db:
        run = db.get(InferenceRun, id)
        if not run or run.status == 'completed':
            return
        run.status = 'running'
        db.commit()
        try:
            longest = max((len(t['tokens']) for t in run.trace_snapshot), default=0)
            if longest > storage.LEGACY_INLINE_MAX:
                result = analyze_segmented(
                    id, run.profile_snapshot, run.trace_snapshot, ARTIFACT_DIR)
            else:
                result = analyze(run.profile_snapshot, run.trace_snapshot)
            Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
            Path(ARTIFACT_DIR, f'hmm-{id}.json').write_text(
                json.dumps(result, ensure_ascii=False, allow_nan=False))
            run.result = result
            run.total_log_likelihood = result['total_log_likelihood']
            run.status = 'completed'
            run.error = None
            db.commit()
            record(db, 'worker', 'run.completed',
                   {'run_id': id, 'trace_count': result['trace_count'],
                    'schema': result['result_schema']})
        except Exception as error:
            db.rollback()
            run = db.get(InferenceRun, id)
            run.status = 'failed'
            run.error = str(error)[:1000]
            db.commit()
            record(db, 'worker', 'run.failed', {'run_id': id, 'error': run.error})
