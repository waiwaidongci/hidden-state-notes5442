"""推断明细的分块磁盘存储。

长轨迹（长度 > LEGACY_INLINE_MAX）的完整后验、log alpha/beta、Viterbi 父指针不落数据库，
而是以每轨迹一个 artifact 目录的形式写入共享 ARTIFACT_DIR：

    {ARTIFACT_DIR}/runs/{run_id}/trace-{trace_id}/manifest.json
    {ARTIFACT_DIR}/runs/{run_id}/trace-{trace_id}/blocks.jsonl

manifest 保存轨迹级标量与整条 Viterbi/边际路径（int8 路径本身很小），
blocks.jsonl 每行一个区段块，流式读取、按区段随机访问，调用方无需一次性载入全序列。
推断过程中块先写 spool 目录，两遍完成后按 index 顺序拼装，峰值内存与序列长度无关。
"""
import json
import os
import shutil
from pathlib import Path

from .engine import DiskBlockSink

LEGACY_INLINE_MAX = 200
ARTIFACT_RUN_PREFIX = 'runs'


class ArtifactError(RuntimeError):
    pass


def _trace_dir(artifact_root, run_id, trace_id):
    return Path(artifact_root) / ARTIFACT_RUN_PREFIX / str(run_id) / f'trace-{trace_id}'


def new_disk_sink(artifact_root, run_id, trace_id):
    """为一条轨迹创建磁盘块 sink；重复调用会清掉旧的半成品目录。"""
    target = _trace_dir(artifact_root, run_id, trace_id)
    spool = target / 'spool'
    if target.exists():
        shutil.rmtree(target)
    spool.mkdir(parents=True, exist_ok=False)
    return DiskBlockSink(spool)


def persist_trace_artifact(artifact_root, run_id, trace_id, sink, manifest):
    """两遍推断结束后拼装最终 artifact，并写入 manifest。"""
    target = _trace_dir(artifact_root, run_id, trace_id)
    spool = target / 'spool'
    blocks_path = target / 'blocks.jsonl'
    block_count = int(manifest['block_count'])
    with blocks_path.open('w', encoding='utf-8') as out:
        for index in range(block_count):
            part = spool / f'block-{index}.json'
            with part.open('r', encoding='utf-8') as src:
                out.write(src.read())
                out.write('\n')
    manifest_path = target / 'manifest.json'
    payload = {'artifact_version': 'fixed-hmm-segments-v1', **manifest}
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    shutil.rmtree(spool)
    return str(Path(ARTIFACT_RUN_PREFIX) / str(run_id) / f'trace-{trace_id}')


def trace_artifact_dir(artifact_root, artifact_key):
    return Path(artifact_root) / artifact_key


def load_manifest(artifact_root, artifact_key):
    path = trace_artifact_dir(artifact_root, artifact_key) / 'manifest.json'
    if not path.exists():
        raise ArtifactError('推断明细 artifact 不存在或已被清理')
    return json.loads(path.read_text(encoding='utf-8'))


def iter_blocks(artifact_root, artifact_key):
    """按区段顺序流式产出块，每次只持有一个块的内存。"""
    path = trace_artifact_dir(artifact_root, artifact_key) / 'blocks.jsonl'
    if not path.exists():
        raise ArtifactError('推断明细 artifact 不存在或已被清理')
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_block(artifact_root, artifact_key, index):
    """随机读取单个区段块（顺序扫描到目标 index，块数至多百级）。"""
    target = int(index)
    for block in iter_blocks(artifact_root, artifact_key):
        if block['index'] == target:
            return block
    raise ArtifactError(f'区段 {target} 不存在')


def artifact_exists(artifact_root, artifact_key):
    base = trace_artifact_dir(artifact_root, artifact_key)
    return (base / 'manifest.json').exists() and (base / 'blocks.jsonl').exists()
