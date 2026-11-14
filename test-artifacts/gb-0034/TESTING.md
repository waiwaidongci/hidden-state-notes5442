# 验收批次 gb-0034：单轨迹十万码 / 五隐态固定模型分段推断

日期：2026-11-14。本批次与源码分离，脚本在 `scripts/`，样例数据在 `data/`，原始日志在 `logs/`。

## 需求对照

|需求|实现与验证|
|---|---|
|单轨迹十万码、五隐态|`engine.MAX_TRACE_TOKENS=100000`，状态 2..5；十万码五隐态实测通过（见下）|
|推断进程峰值内存 ≤256MB|分块落盘，工作集仅一个块（默认1000×5）；实测峰值 RSS 48.1–48.4MB（基线 32.9MB）|
|结果分批按区段查看|`blocks.jsonl` 每1000码一块；新增 `GET /runs/{id}/traces/{tid}/segments[/{index}]`，前端 `SegmentBrowser.vue` 区段翻页|
|保留全序列后验|后验逐块写入 artifact，全量流式读回共 100000 行，列和均为 1|
|保留联合最优路径|父指针矩阵 + 回溯，整条 Viterbi 路径存 manifest；与旧 tuple 实现逐路径一致|
|保留原有平局规则|精确 log 平局取状态索引序列字典序最小，边际平局取最小 index；`test_ties.py` 对照旧引擎|
|不能分段各算各的|两遍法在完整序列递推，β 跨块接续；多块(bs=37)与单块(bs=5000)结果逐元素一致|
|跨段零转移算对|含精确零矩阵穷举 + 1000码跨块零转移用例，Viterbi 全程无非法边|
|长序列下溢算对|log 域全程；200步1e-4 得 logL −1842.0680743952、线性概率 null；十万码 logL 有限并标记下溢|
|后段观测改变前段后验|同前缀不同后缀，t=199 后验显著变化；慢混合链 t=0 后验随后后缀变化，证明后向信息跨块传播|
|批准说明导出完整明细|分段 CSV 逐块流式生成，e2e 实测 100000 数据行 + 表头，16 列|
|排队时冻结输入|提交即冻结 profile/trace 快照；测试中事后修改源轨迹，artifact 与 manifest 不变|
|旧短轨迹结果和说明可读|≤200码走 `inline-v1`，输出键集合与历史版本一致；旧说明 CSV 导出回归通过|

## 测试脚本与运行方式

环境：Python 3.11 + numpy 2.2.1（算法不依赖 scipy 运行时；requirements 保留 scipy 不影响）。

```sh
cd backend
python3 ../test-artifacts/gb-0034/scripts/test_exhaustive.py     # 小序列穷举（211组）
python3 ../test-artifacts/gb-0034/scripts/test_ties.py           # 平局规则对照旧引擎
python3 ../test-artifacts/gb-0034/scripts/test_long_100k.py      # 十万码实测/内存/下溢/零转移
JWT_SECRET=x DATABASE_URL=sqlite:////tmp/hmm-reg.db \
  python3 ../test-artifacts/gb-0034/scripts/test_service.py      # 服务层集成（无外部服务）
python3 ../test-artifacts/gb-0034/scripts/test_api_e2e.py        # FastAPI 端到端（TestClient+SQLite+假Redis）
```

前端：`cd frontend && npm ci && npm run build`（vue-tsc 严格类型检查 + Vite 构建通过）。

### 1. 小序列穷举 `test_exhaustive.py`

独立纯 Python 枚举全部 K^T 路径（不与被测代码共用数值实现），覆盖 2..5 隐态、2..6 观测码、序列长度 1..7、稠密与含精确零矩阵共 211 组；每组在 5 种块大小（含 bs=1）下与穷举解和旧内联实现对照 logL、后验（容差1e-10）、Viterbi、边际路径、null 语义。另含 README 两组标准校验（二态三步 likelihood 907/25000、logL −3.316488653735、MAP 001；三态零转移 likelihood 1/40、MAP 012、边际 002 非法）与不可能序列（初态零发射、中途全路径封死）。

### 2. 平局规则 `test_ties.py`

对称二态/三态模型制造大量精确浮点平局，逐路径对照 git HEAD 的旧 `infer_sequence`（完整 tuple 比较）与新秩机制，跨块大小 1/2/3/7/100 全部一致，字典序最小路径恒为全 0。

### 3. 十万码实测 `test_long_100k.py`（日志 `logs/long-100k.log`）

- 五隐态六码、按模型采样的 100000 码序列：推断 13.9–14.6s，**峰值 RSS 48.1–48.4MB（基线 32.9MB，约束 256MB）**，artifact 磁盘 42.7MB，100 区段；
- 区段随机读取（首/中/末块）后验列和为 1，全量流式读回 100000 行无错位；
- 5000码 bs=37 多块与单块：Viterbi/边际路径相同、logL 差 <1e-8、后验逐元素差 <1e-10；
- 下溢：200步单步1e-4，logL=−1842.0680743952（与 README 一致），线性概率 null；
- 后段影响：同前缀接不同后缀，t=199 后验 `[.384,.372,.157,.064,.023]→[.138,.506,.207,.106,.043]`；慢混合链 t=0 后验 `[1,0,0,0,0]→[.954,.046,0,0,0]`；
- 1000码跨块零转移（bs=100）：Viterbi 全程无零转移/零发射非法边。

### 4. 服务层与端到端

- `test_service.py`：分段存储、末块不足整块、不可能任务失败且无半成品 artifact、冻结快照、分段 CSV 3501 行、旧内联键集合与 240 行 CSV；
- `test_api_e2e.py`（日志 `logs/api-e2e.log`）：登录鉴权、十万码轨迹导入、100001 码 422、长任务分段完成且 DB 不内联后验、区段 API 与越界 404、说明建立/审批/重复审批409/未批准409、**批准 CSV 100001 行（含表头）16 列**、旧短轨迹内联结果与说明可读、viewer 写操作 403；
- 种子回归：`scripts.seed_demo` 在 SQLite 下产出计数 3/3/240/3/1/1，A研究总 logL −969.126197943428、首条 −49.50643547309272，与 README 记载一致。

## 数据与哈希

- `data/sample-segment-artifact.json`：2500码五态样例的结果摘要、manifest 摘录与首块前20行（含 null 精确零、父指针、后验）。
- 源码 SHA-256（验收时）：

```
92b0279e…d4ba  backend/src/services/engine.py
3f532ea6…99d9  backend/src/services/hmm.py
ca5a6b01…2815  backend/src/services/storage.py
adb9e65b…ce635  backend/src/services/analysis.py
3e679ba8…a1fc  backend/src/worker.py
36bbbcc1…e187  backend/src/controllers/export.py
4bc7f7fb…00b4  backend/src/routes/runs.py
05ddc7b8…f146  backend/src/schemas/trace.py
6f62772b…f0ec  backend/src/services/validation.py
```

## 未覆盖

未在真实 MySQL8/Redis/Celery 容器与浏览器中复跑（本地无 Docker），端到端以 TestClient+SQLite+同步任务+假 Redis 等价覆盖路由、ORM、快照、artifact 与导出链路；未做生产级并发压力与 artifact 卷清理运维流程。
