"""端到端 API 冒烟：TestClient + SQLite + 假 Redis + 同步执行 Celery 任务。

覆盖：登录、十万码轨迹导入与 422 上限、长轨迹分段推断、区段查看 API、
说明建立/审批、分段 CSV 流式导出、旧短轨迹内联结果与旧说明可读。
"""
import csv
import io
import os
import sys
import tempfile

os.environ.setdefault('JWT_SECRET', 'test-secret')
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/hmm-e2e.db')
os.environ.setdefault('SEED_DEMO', 'false')

ARTIFACT_TMP = tempfile.mkdtemp(prefix='hmm-artifacts-')
os.environ['ARTIFACT_DIR'] = ARTIFACT_TMP
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend')))

# 清理旧库，保证可重复执行
if os.path.exists('/tmp/hmm-e2e.db'):
    os.remove('/tmp/hmm-e2e.db')

from fastapi.testclient import TestClient  # noqa: E402
import src.middlewares.rate_limit as rate_limit_module  # noqa: E402
import src.worker as worker_module  # noqa: E402
from src.config.database import Base, engine, SessionLocal  # noqa: E402
from src.main import app  # noqa: E402
from src.models import User  # noqa: E402
from src.utils.security import hash_password, issue_token  # noqa: E402


class FakeRedis:
    store = {}

    @classmethod
    def from_url(cls, url, socket_timeout=None):
        return cls()

    def incr(self, key):
        FakeRedis.store[key] = FakeRedis.store.get(key, 0) + 1
        return FakeRedis.store[key]

    def expire(self, key, seconds):
        return True

    def ping(self):
        return True


rate_limit_module.Redis = FakeRedis

# Celery 任务同步执行：runs 服务以 analyze_run.delay(run.id) 调用
worker_module.analyze_run.delay = lambda run_id: worker_module.analyze_run(run_id)

Base.metadata.create_all(engine)
with SessionLocal() as db:
    admin = User(email='admin@yanxi.demo', name='负责人', role='admin',
                 password_hash=hash_password('Yanxi2026!'))
    analyst = User(email='analyst@yanxi.demo', name='分析员', role='analyst',
                   password_hash=hash_password('Yanxi2026!'))
    db.add_all([admin, analyst])
    db.commit()
    admin_token = issue_token(admin)
    analyst_token = issue_token(analyst)

client = TestClient(app)
admin_headers = {'Authorization': f'Bearer {admin_token}'}
analyst_headers = {'Authorization': f'Bearer {analyst_token}'}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    symbols = ['A', 'B', 'C', 'D', 'E', 'F']
    states = ['s0', 's1', 's2', 's3', 's4']
    pi = [0.3, 0.25, 0.2, 0.15, 0.1]
    A = [[0.55, 0.25, 0.1, 0.08, 0.02], [0.1, 0.5, 0.25, 0.1, 0.05],
         [0.05, 0.15, 0.5, 0.2, 0.1], [0.02, 0.08, 0.2, 0.5, 0.2],
         [0.01, 0.04, 0.1, 0.25, 0.6]]
    B = [[0.4, 0.25, 0.15, 0.1, 0.07, 0.03], [0.1, 0.35, 0.25, 0.15, 0.1, 0.05],
         [0.05, 0.15, 0.35, 0.25, 0.15, 0.05], [0.05, 0.1, 0.2, 0.35, 0.2, 0.1],
         [0.03, 0.07, 0.15, 0.25, 0.3, 0.2]]

    # 导入含一条 100000 码长轨迹和一条 40 码短轨迹的研究
    long_tokens = [symbols[(t * 7 + 3) % 6] for t in range(100000)]
    payload = {
        'name': '十万码设备日志研究', 'filename': 'yx-device-202611.json',
        'description': 'e2e', 'symbols': symbols,
        'traces': [
            {'reference': 'DEV-LONG', 'tokens': long_tokens},
            {'reference': 'DEV-SHORT', 'tokens': ['A', 'B'] * 20},
        ],
    }
    response = client.post('/studies', json=payload, headers=analyst_headers)
    check(response.status_code == 200, response.text)
    study_id = response.json()['id']

    # 超过十万码被 422 拒绝
    too_long = {'reference': 'DEV-TOO-LONG', 'tokens': ['A'] * 100001}
    response = client.post('/studies', json={
        'name': '超限研究', 'filename': 'x.json', 'symbols': symbols, 'traces': [too_long],
    }, headers=analyst_headers)
    check(response.status_code == 422, f'超长轨迹应 422，实际 {response.status_code}')

    # 冻结五态模型版本
    profile_payload = {'study_id': study_id, 'name': '五态固定参数-v1', 'version': 'v1',
                       'states': states, 'pi': pi, 'transition': A, 'emission': B}
    response = client.post('/profiles', json=profile_payload, headers=analyst_headers)
    check(response.status_code == 200, response.text)
    profile_id = response.json()['id']

    traces = client.get('/traces', headers=analyst_headers).json()
    long_trace = next(t for t in traces if t['reference'] == 'DEV-LONG')
    short_trace = next(t for t in traces if t['reference'] == 'DEV-SHORT')

    # 长轨迹分段推断（同步执行 worker）
    response = client.post('/runs', json={
        'study_id': study_id, 'profile_id': profile_id, 'name': '十万码分段推断',
        'trace_ids': [long_trace['id']],
    }, headers=analyst_headers)
    check(response.status_code == 200, response.text)
    long_run_id = response.json()['id']
    long_run = client.get(f'/runs/{long_run_id}', headers=analyst_headers).json()
    check(long_run['status'] == 'completed', f'长轨迹推断未完成：{long_run.get("error")}')
    result = long_run['result']
    check(result['result_schema'] == 'segments-v1', '应为分段结果')
    check(result['token_count'] == 100000, result['token_count'])
    long_entry = result['traces'][0]
    check(long_entry['block_count'] == 100, long_entry['block_count'])
    check('posterior' not in long_entry, 'DB 结果不得内联全序列后验')

    # 区段查看 API
    segments_meta = client.get(
        f'/runs/{long_run_id}/traces/{long_trace["id"]}/segments',
        headers=analyst_headers).json()
    check(segments_meta['block_count'] == 100, '区段元数据')
    check(len(segments_meta['viterbi_path']) == 100000, '完整联合路径在 manifest')
    block0 = client.get(
        f'/runs/{long_run_id}/traces/{long_trace["id"]}/segments/0',
        headers=analyst_headers).json()
    check(len(block0['posterior']) == 1000 and len(block0['tokens']) == 1000, '首区段明细')
    check(all(abs(sum(row) - 1.0) < 1e-9 for row in block0['posterior']), '区段后验列和为1')
    last_block = client.get(
        f'/runs/{long_run_id}/traces/{long_trace["id"]}/segments/99',
        headers=analyst_headers).json()
    check(last_block['start'] == 99000 and len(last_block['posterior']) == 1000, '末区段')
    out_of_range = client.get(
        f'/runs/{long_run_id}/traces/{long_trace["id"]}/segments/100',
        headers=analyst_headers)
    check(out_of_range.status_code == 404, '越界区段 404')

    # 短轨迹走旧内联路径，结构与历史版本一致
    response = client.post('/runs', json={
        'study_id': study_id, 'profile_id': profile_id, 'name': '短轨迹内联推断',
        'trace_ids': [short_trace['id']],
    }, headers=analyst_headers)
    check(response.status_code == 200, response.text)
    short_run_id = response.json()['id']
    short_run = client.get(f'/runs/{short_run_id}', headers=analyst_headers).json()
    check(short_run['status'] == 'completed', short_run.get('error'))
    check(short_run['result']['result_schema'] == 'inline-v1', '短轨迹应为内联结果')
    short_entry = short_run['result']['traces'][0]
    check(len(short_entry['posterior']) == 40 and 'log_forward' in short_entry, '旧结构键保留')

    # 长轨迹说明：analyst 建立，admin 批准，viewer 不可批
    response = client.post('/notes', json={
        'run_id': long_run_id, 'name': '十万码工况核对说明',
        'interpretation': '固定模型解释，后验与联合路径分开读，不是故障事实。',
        'next_step': '回看设备记录并按区段核对，不作为在线告警。',
    }, headers=analyst_headers)
    check(response.status_code == 200, response.text)
    note_id = response.json()['id']
    # 未批准不能导出
    check(client.get(f'/notes/{note_id}/export', headers=admin_headers).status_code == 409,
          '未批准说明不应允许导出')
    response = client.patch(f'/notes/{note_id}/approval',
                            json={'status': 'approved', 'review_note': '批准完整明细导出。'},
                            headers=admin_headers)
    check(response.status_code == 200, response.text)
    # 重复审批 409
    check(client.patch(f'/notes/{note_id}/approval',
                       json={'status': 'approved', 'review_note': '重复'},
                       headers=admin_headers).status_code == 409, '重复审批应 409')
    # 完整明细 CSV：流式，行数 100000 + 表头
    export_response = client.get(f'/notes/{note_id}/export', headers=admin_headers)
    check(export_response.status_code == 200, export_response.text[:200])
    csv_text = export_response.content.decode('utf-8-sig')
    csv_rows = list(csv.reader(io.StringIO(csv_text)))
    check(len(csv_rows) == 100001, f'CSV 行数 {len(csv_rows)}')
    check(len(csv_rows[0]) == 16, 'CSV 表头 16 列')
    check(csv_rows[1][4] == 'DEV-LONG' and csv_rows[-1][4] == 'DEV-LONG', 'CSV 覆盖全轨迹')

    # 旧短轨迹说明同样可建立、批准、导出，结构保持可读
    response = client.post('/notes', json={
        'run_id': short_run_id, 'name': '短轨迹工况核对说明',
        'interpretation': '旧内联结果的说明仍按原结构导出。',
        'next_step': '按原核对流程归档。',
    }, headers=analyst_headers)
    check(response.status_code == 200, response.text)
    short_note_id = response.json()['id']
    client.patch(f'/notes/{short_note_id}/approval',
                 json={'status': 'approved', 'review_note': '批准。'},
                 headers=admin_headers)
    short_csv = client.get(f'/notes/{short_note_id}/export', headers=admin_headers)
    check(short_csv.status_code == 200, '旧短轨迹说明导出应成功')
    check(len(list(csv.reader(io.StringIO(short_csv.content.decode('utf-8-sig'))))) == 41,
          '短轨迹 CSV 应为 40 数据行加表头')

    # 权限：viewer 角色写操作 403（用一个未建用户验证角色链路）
    with SessionLocal() as db:
        viewer = User(email='viewer@yanxi.demo', name='观察员', role='viewer',
                      password_hash=hash_password('Yanxi2026!'))
        db.add(viewer)
        db.commit()
        viewer_token = issue_token(viewer)
    forbidden = client.post('/runs', json={
        'study_id': study_id, 'profile_id': profile_id, 'name': '越权推断',
        'trace_ids': [short_trace['id']],
    }, headers={'Authorization': f'Bearer {viewer_token}'})
    check(forbidden.status_code == 403, 'viewer 写操作应 403')

    print('PASS 端到端 API 冒烟：十万码分段/区段 API/审批导出/旧内联/权限全部正确')
    print(f'artifact 目录：{ARTIFACT_TMP}')


if __name__ == '__main__':
    main()
