---
name: hyra-discovery
description: >-
  本地、无外部 API、纯 Python 标准库的科学发现智能体 harness（受腾讯混元
  Hyra 递归自我改进思路启发）。当用户要做「科学发现 / 假设演化 / 药物-靶点
  SAR 发现 / 时间序列预测演化 / 棋类策略演化」，或要求「用 HY3/WorkBuddy 跑
  无人值守的假设搜索循环」「把科学发现智能体接入 WorkBuddy」「自研 Hunyuan
  Research Agent」，或要做医疗方向（药物发现、靶点结合、结构-活性关系）的
  自主实验时触发。框架自带沙盒打分、防奖励黑客（选模只用验证集、测试集仅
  最终报告）、warm-start 经验库与启发式提议器（GuidedProposer）。
---

# Hyra-Discovery 科学发现智能体（医疗方向预设）

## 这是什么
一个**自研**、**运行于 WorkBuddy 会话内**的科学发现 harness：推理/提议由 **WorkBuddy
已接入大模型（HY3）** 承担，**不调用任何外部 LLM/推理 API**、也不需自建本地算力。核心
**纯 Python 标准库**仅作「客观评分沙盒」（运行候选解、算 R²/合法性），可选 `torch`
用于 mathlaw 真实训练、可选 `rdkit` 用于 synthesis 真实化学——二者是**真实数据/化学
增强**，不是推理引擎。它把一类科学问题
建模成「基因组搜索」：

1. 你（或 HY3 在会话内）写一个 `solution.py` —— 一个候选假设/模型；
2. harness 在**加固沙盒**（独立子进程、清空 PYTHONPATH）里运行它并打分；
3. 打分**只用验证集**，测试集仅用于最终报告 —— 这堵死了「针对测试集过拟合」
   的奖励黑客；`Evaluator` 还会对过大 overfit_gap 施加惩罚；
4. **主路径 = HY3 会话内驱动闭环**：在 WorkBuddy 会话里，HY3 读经验库与本轮分数，
   **自己推理**下一个候选（为什么低分、该改哪些特征/变换/片段、机制假设是什么），
   写 `solution.py` 再由沙盒打分——这是真实的「大模型当算力」递归自我改进；
5. **无人值守回退**：无会话 LLM  steer 时（如定时自动化），`GuidedProposer` 用启发式
   （最优个体引导变异 + top2 交叉 + 探索 + 去重 + warm-start）兜底；
6. 结果回写经验库，下一轮从累积历史继续。

## 跨任务递归自我改进（SharedExperienceBank）

这是 harness 对标 Hyra「递归自我改进」的关键一层，分两级：

1. **任务内经验库** `ExperienceBank`（每任务 `eb_hy3_auto/`）：记录本任务探索过的每个
   基因组与结果，`GuidedProposer.warm_start` 据此去重、续跑、避免重复评估。
2. **跨任务共享记忆** `SharedExperienceBank`（`harness/shared_eb/`，纯标准库）：
   每个任务跑完，**只写一条「可迁移元知识」**——不是原始基因组（基因组是任务专属的），
   而是「哪种范式赢了 + 赢了的方案有多稀疏」。例如医疗任务的 lesson：
   `{"family":"medical","sparsity":0.30,"won_by_parsimony":true,"archetype":"sparse_subset"}`。

**迁移机制**：新任务冷启动时，若同家族（如 `medical`）已有先验，则：
- 起始种子从「全特征开启」改为**按家族典型稀疏度稀疏化**（`seed_from_priors`）；
- `GuidedProposer` 对每个布尔维度施加 **Occam 偏置**（`set_family_prior`）：变异/随机
  时更可能关掉特征，让搜索从「稀疏优先」起步，而非盲目全开。

**实证**：`test_cross_task_shared` 验证——一条 `medical` 先验
（`median_sparsity=0.30`）注入后，`drugcombo` 冷启动种子自动从全开 48 维变稀疏
（on=14/48）；无人值守循环里 `auto_loop` 也会在 `event:seed` 打印 `seed_on` 与
`priors`，可审计迁移是否生效。这是「一个医疗任务学到的稀疏偏好，自动传给下一个医疗
任务」的 Hyra 式跨任务递归改进，且**全程纯标准库、零外部 API**。

## 医疗方向（已内置 5 个预设）

### 1. 药物-靶点结合 / SAR（drugtarget）
结构-活性关系发现代理任务：
- **数据**：`tasks/drugtarget/data/dataset.csv` 透明、可复现的**合成** SAR 表
  （已知真因特征 `{2,5,11}` + 交互，并含 olaparib 式参考分子）。它**不是**真实
  生化数据，只是让自主发现循环端到端演示的占位基准。
- **接真实数据**：把 CSV（列 `f0..fF-1, activity` 或 `label`）放到该目录，框架按
  列数自动适配，无需改代码。
- **实证**（本地）：无人值守循环重新发现真因 `{2,5,11}`（val_r2≈0.99），parsimony
  （调整 R² 选模）生效。
- **评估升级（诚实性）**：选模改用** K 折交叉验证**（不再依赖单次固定切分，避免稀疏胜出被当成单次运气），并加入**全特征最小二乘强基线**对照——稀疏发现必须 `beats_baseline=True`（在测试集上优于全特征模型）才算真有意义。

### 2. 老药新用（drugrepurposing）
基因表达签名匹配（CMap / LINCS 式）代理任务：
- **数据**：`tasks/drugrepurposing/data/{disease_expr.csv, drug_profiles.csv}`。
  疾病模块由 10 个混合方向基因构成；药物扰动矩阵中埋设**某一已上市药能反向该
  疾病模块**作为真因（老药新用候选）。
- **机制**：候选基因子集→在药物表达谱上预测「反转疾病签名」强度→与药物真值
  反转做验证集检索一致性（pearson），测试集仅报告该老药排名。
- **实证**（本地）：无人值守循环能命中 9/10 模块基因、测试集把真药排到 top（rank≈1）。

### 3. 机制药 / MOA 分类（moa）
多类作用机制（Mechanism of Action）判别签名发现：
- **数据**：`tasks/moa/data/dataset.csv`。每类机制各有 1 个独占「标记特征」（共 6
  类 / 6 个标记特征），其余为噪声，使真机制签名**稀疏**（6/20）。
- **机制**：候选特征子集→1-vs-rest 最小二乘分类→验证集准确率 + Occam 惩罚（特征
  越少越好）作为选模分。
- **实证**（本地）：标记特征子集 score≈0.72 > 全特征 0.49，循环能收敛到稀疏机制签名。

### 4. 合成药片段生成（synthesis，现已升级为 RDKit 真实化学）
- **数据**：`tasks/synthesis/data/{fragments.csv, library.csv}`。片段库为**真实 SMILES
  片段**（每个带一个 `[*]` 连接点）；library.csv 是「已知片段组合」参考集（用于新颖性判定）。
- **机制**：候选片段子集→通过 RDKit **真实拼接**成分子（连接点成键 + Sanitize 校验）→
  **合法性是真实化学检查**（RDKit 价键/芳香性），目标为 **QED 真实类药物性**（rdkit
  QED）。无效组合（价键超限）→ 低分淘汰。运行需 `rdkit`（见「真实化学」段）。
- **实证**（本地，venv + rdkit）：生成真实合法分子如 `[H]C(=O)c1ccccc1NC=O`
  （QED≈0.65，valid=True, novel=True）。smoke 在缺 rdkit 时自动 SKIP。
- **诚实边界**：这是**基于真实 RDKit 的片段组装生成**（真实分子、真实 QED），**仍非**
  Hyra 式 de-novo Transformer；`activity` 仍为合成代理（无湿实验读数）。

> 前四个任务的**真值均为合成埋设**，用于验证 harness 的「自主发现闭环」是否成立；
> 接真实数据时把对应 CSV 替换即可，框架自动适配维度。

### 5. 多药联用协同（drugcombo，DrugComb 式合成基准）
- **这是什么**：把「两种药联用是否协同」建模成药对表征的特征子集选择问题
  （同类任务如 DrugComb / NCI-ALMANAC 组合筛选，但这里是**合成透明基准**）。
- **数据**：`tasks/drugcombo/data/dataset.csv`。每个药对有 48 维表征 = 药 A 机制(16)
  + 药 B 机制(16) + 同索引交互(16)；780 个药对，60/20/20 切分。**真因稀疏**：协同
  仅由特征 `{1, 27, 35, 37}` 驱动（药 A 通路1、药 B 通路11、两个同通路 AND 交互）。
- **机制**：基因组=药对表征的特征子集；runner 做最小二乘 + K 折 CV 选模 + 全特征
  基线对照（`beats_baseline` 必须为真）。
- **实证**（本地，smoke）：真因子集 val_r2≈0.996、test_r2≈0.996 且 `beats_baseline=True`。
- **诚实边界**：合成代理，验证「自主发现协同相关特征」闭环，**非**预测真实临床联用疗效。

> 五个医疗任务的**真值均为合成埋设**，用于验证 harness 的「自主发现闭环」是否成立；
> 接真实数据时把对应 CSV 替换即可，框架自动适配维度。

## 真实数据接入（real-data 模式）

harness 内建「真实数据适配层」（`hyra/realdata.py`）：每个医疗任务只要在
`tasks/<任务>/data_real/` 放入**符合其 schema 的真实 CSV**，框架就**自动切换**到真实数据，
无需改任何代码。合成数据只是默认占位基准。

**约定（schema 必须与合成文件一致：相同列数 / 特征列 / 目标列名）**：
- `drugtarget` / `moa` / `drugcombo` ：`data_real/dataset.csv`（列 `id, f0..fF-1, activity/label, split`）。
- `drugrepurposing` ：`data_real/disease_expr.csv` + `data_real/drug_profiles.csv`。
- `synthesis` ：用内置 **真实 RDKit 片段库**（默认即真实化学），无需放文件。

**已验证的真实数据源（drugtarget）**：`tasks/drugtarget/gen_real.py` 从公开
**ChEMBL API** 拉取 **CHEMBL205（HIV-1 蛋白酶）真实 IC₅₀**，用 RDKit 算 12 个真实
2D/理化描述符作为特征、pIC₅₀ 作目标，写成 `data_real/dataset.csv`。
实测无人值守循环在**真实数据**上达到 **test R²≈0.39**（6 特征线性模型）——这是**真实
QSAR 预测力**，不是合成占位。运行：

```bash
# 用装了 rdkit 的 Python 拉取真实数据（只需一次）
python tasks/drugtarget/gen_real.py
# 然后在真实数据上跑发现循环
python scripts/dispatch.py tasks.drugtarget.task --mode auto --iters 18 --fresh
```

> 诚实边界：真实数据给出**真实预测 R² vs 全特征基线**，但**不恢复「真因」**（真实数据
> 没有透明埋设的稀疏真因），也**不是** Hyra / 外部基准的官方验收。要接 LINCS L1000 /
> ChEMBL 全量 / TCGA 等更大真实数据，按上述 schema 放置 CSV 即可，但需领域专家对数据
> 与指标做复核。

> `gen_real.py` 目前为 drugtarget 提供 ChEMBL 直连；drugrepurposing 接 LINCS、其他任务的
> 真实源可按同样 schema 扩展——真实数据获取与领域对齐由使用者负责（诚实声明边界）。

## 通用科学发现方向（非医疗，已内置 5 个代理任务）

### 5. 太阳黑子预测（sunspot，含官方多步口径 + 强基线对照）
- **数据**：`tasks/sunspot/data/sn_m_tot.csv` 是真实历史月均太阳黑子数（1749–2026，
  固定 CSV，无生成器）。train 1749–1931 切 60/40 选模，test 1932–2026 **仅最终报告**。
- **官方多步口径**：滚动多步递归预测，报告 h=1/3/6/12 各视野 R²（非单点）。
- **强基线对照**（诚实 contextualise 本方法技能）：
  - 持续性（persistence，上一值）：短视野强基线（h=1 已达 0.873）。
  - 季节性朴素（period=132，即 ~11 年太阳周期）：物理正确的周期，但**实测
    R²≈0.37 反而很弱**——揭示太阳周期存在**相位漂移**，固定周期重复法并不适合，
    恰是深度学习/可学习周期模型的价值所在（诚实洞见，非缺陷）。
- **实证**（本地，无人值守循环，最优 genome P=28/periods=[132,66,264]/phase_gain+sqrt）：
  逐视野 R² —— 模型 **0.901 / 0.864 / 0.835 / 0.768**（h=1/3/6/12），
  持续性 0.873 / 0.791 / 0.708 / 0.483，季节性朴素(11y) 0.374 / 0.373 / 0.372 / 0.369。
  模型在所有视野**全面优于两基线**，长视野优势显著（h=12 领先 persistence 0.28）。
- **诚实边界**：模型是线性 AR+谐波（可学习相位），**非深度学习**；短期增益有限
  （h=1 仅 +0.03），价值在长视野。要真正对标 Solar-TALENT 式官方赛制，需接入官方
  数据集/指标并用 torch 训练 Transformer（mathlaw 的 torch 集成已可行）。
- 接真实数据：替换该 CSV（格式 `年;月;值`，分号分隔）即可。

### 6. 数学定律发现（mathlaw，符号回归本地代理）
- **数据**：`tasks/mathlaw/data/dataset.csv` 由隐藏定律生成——**开普勒第三定律
  T = a^1.5**（a 为半长轴，T 为周期），外加一个**无关噪声特征 z** 与高斯观测噪声。
- **机制**：基因组=对原始列施加的变换集合（原始 / √ / a^1.5 / log / 平方 / 截距 /
  是否含 z）。**三种引擎**（torch 可用时 genome 自动出现 `engine` 维度）：
  - `numpy`：闭式最小二乘 + 调整 R² 选模（稀疏优先，原始本地代理）。
  - `torch_sparse`：**真实 torch 训练**——基函数张量上 Adam + **L1 稀疏**，无关项
    权重自动趋零；实证自动发现 `a^1.5` 并剔除噪声 z（selected=['t_pow15']）。
  - `torch_transformer`：**真实 torch 训练**——极小 Transformer（特征间注意力）端到端
    拟合 a→T（test_r2≈0.997），展示深度学习训练闭环；它*拟合*映射但**不输出符号
    表达式**（诚实边界：深度网络不是符号发现）。
  验证集选模只用验证集，测试集仅报告。
- **实证**（本地）：numpy 与 torch_sparse 均锁定真因 `a^1.5`（test_r2≈0.997）并剔除 z；
  torch_transformer 拟合到同等精度。三者均通过 smoke 测试（torch 不可用该引擎自动隐藏）。
- **诚实边界**：`torch_sparse` 是**真实梯度下降符号回归**（非闭式代理）；
  `torch_transformer` 是**真实小模型训练**（非符号发现）。二者仍在本地 harness 框架内，
  **不是** Hyra 官方 15 参数加法 Transformer（那需大规模 GPU 训练与专用合成数据）。
  启用 torch 引擎：在运行 harness 的 Python 环境 `pip install torch`，框架自动检测。

### 7. 量子线路路由（quantum_routing，SWAP 网络深度最小化代理）
- **数据**：`tasks/quantum_routing/data/circuit.json` 固定基准线路（环形 5 比特耦合图
  + 14 个双比特门，种子固定可复现）。
- **机制**：基因组=路由策略（route_closer 选更短环路径 / use_bridge 用桥门省层 /
  order 门处理顺序）；runner 在耦合图上做 SWAP 网络映射并**层打包**算线路深度，
  score=−深度（越小越好）。
- **实证**（本地）：无人值守循环发现最小深度策略 depth=16（远优于弱策略 28）。
- **诚实边界**：这是**启发式路由本地代理**（IBM Qiskit 式 routing 子问题的简化版）；
  **不是** Hyra 官方量子验收（那需 qiskit / 真实 simulator / 硬件）。

> 数学与量子两个任务的**真值/基准均为合成或简化**，仅用于验证「跨域自主发现闭环」；
> 要接真实领域数据与算力（如真实Transformer训练、真实量子编译器），需引入 GPU/torch/
> qiskit，超出本纯 stdlib harness 范围——这是诚实声明的边界。

## 如何运行（由 WorkBuddy 直接调度）
harness **运行于 WorkBuddy 会话内**：推理由 WorkBuddy 已接入大模型（HY3）承担，
Python 仅作「客观评分沙盒」（经 WorkBuddy 工具运行时执行 `solution.py` 并算分）。
入口 `scripts/dispatch.py` 解析内置 harness 根并派发。

**主路径（推荐）：HY3 会话内驱动闭环** —— HY3 逐轮读分数→推理→改写 `solution.py`→沙盒打分：
```bash
# 单步：HY3 在会话内写 solution.py，harness 打分并回写经验库
python scripts/dispatch.py tasks.drugtarget.task --mode step \
       --solution path/to/sol.py --summary "假设：特征 2/5/11 主导结合"
# agent 在 WorkBuddy 会话里反复调用上条命令 = HY3 当算力的自发发现循环
```

**无人值守回退（无会话 LLM steer 时）**：`GuidedProposer` 启发式循环：
```bash
python scripts/dispatch.py tasks.drugtarget.task --mode auto --iters 20 [--fresh]
```

# 冒烟测试（验证 harness 自洽，含全部 10 个任务）
python scripts/dispatch.py --mode smoke

# 4) 重新生成合成数据（所有带生成器的任务均可显式调用）
python scripts/dispatch.py tasks.drugrepurposing.task --mode gen-data
python scripts/dispatch.py tasks.moa.task --mode gen-data
python scripts/dispatch.py tasks.synthesis.task --mode gen-data
python scripts/dispatch.py tasks.drugcombo.task --mode gen-data
python scripts/dispatch.py tasks.drugtarget.task --mode gen-data
python scripts/dispatch.py tasks.mathlaw.task --mode gen-data
python scripts/dispatch.py tasks.quantum_routing.task --mode gen-data
```

**无人值守统一用法（无会话 LLM 时的回退）**（`--mode auto` 即 GuidedProposer 启发式循环，
`--iters` 控制步数，`--fresh` 清空经验库重跑）：
```bash
# 医疗
python scripts/dispatch.py tasks.drugrepurposing.task --mode auto --iters 25
python scripts/dispatch.py tasks.moa.task            --mode auto --iters 25
python scripts/dispatch.py tasks.synthesis.task      --mode auto --iters 25
python scripts/dispatch.py tasks.drugcombo.task      --mode auto --iters 25
# 通用科学发现
python scripts/dispatch.py tasks.sunspot.task        --mode auto --iters 14
python scripts/dispatch.py tasks.mathlaw.task        --mode auto --iters 22
python scripts/dispatch.py tasks.quantum_routing.task --mode auto --iters 22
```

其它已带任务：`tasks.sunspot`（太阳黑子多步预测 + 强基线对照）、`tasks.othello`
（黑白棋策略演化，配 alpha-beta 强基线）、`tasks.mathlaw`（数学定律发现）、
`tasks.quantum_routing`（量子线路路由）。共 **10 个本地代理任务**
（5 医疗 + 太阳黑子 + 黑白棋 + 数学 + 量子）。

## 验收（自洽验收 / 最终形态）

本 Skill 的最终交付形态是 **WorkBuddy 可调度的 Skill**（harness 为其内部实现），
验收分两层：

### A 层：自洽验收（本次可达）
判定 = 10 个任务在 HY3 驱动 / GuidedProposer 兜底下均产出「发现闭环成立」证据，
且 ≥1 个真实数据端到端。

**主路径 = HY3 会话驱动**（`scripts/hy3_drive.py`）：HY3（WorkBuddy 已接入大模型）
在会话内读经验库→提假设→沙盒评分→回写；推理由 HY3 完成，Python 仅打分。
```bash
# 真实数据模式（gen_real 需 venv+rdkit；hy3_drive 评分用 managed python 即可）
python tasks/drugtarget/gen_real.py
python scripts/hy3_drive.py --task tasks.drugtarget.task --scan
python scripts/hy3_drive.py --task tasks.drugtarget.task --sel 0,4,5,7,8,10 --inter
python scripts/hy3_drive.py --task tasks.drugtarget.task --best
```
HY3 驱动 drugtarget 真实 QSAR 实测：稀疏交互 `{0,4,5,7,8,10}+inter` → **test R²≈0.42**，
显著优于单变量上限(f4 r=0.38)、无人值守 auto_loop(0.39)、稠密全特征基线(共线性发散 R²≈−33万)。

**回退 = 无人值守 GuidedProposer**（`auto_loop.py` / `dispatch --mode auto`）：无会话 LLM 时
启发式搜索兜底（诚实标注非 LLM 推理）。

各任务验收证据详见 `ACCEPTANCE.md`。

### B 层：Hyra 官方 vendor 同口径验收（需腾讯资源，未来）
需官方数据集 + 官方评估脚本 + 官方指标（太阳黑子官方多步 R²、黑白棋 top-3 同规则、
Hyra 15 类任务全集）。非本 Skill 能力范围，需向腾讯侧申请官方资源。

## 添加一个新任务
在 `harness/tasks/<name>/task.py` 实现 `Task` 类，提供：
- `workdir(tag)` / `runner_code()` / `parse_run(run)`
- **基因组接口**（供无人值守循环）：`genome_space()` → `dict{key:[选项]}`、
  `seed_genome()` → `dict`、`render(genome)` → `(code, summary)`
- 时间序列类还需定义 `P` 与 `features(window, t_idx)`；棋类需 `choose_move(board, player)`

## 诚实边界（务必如实传达）
- 这是**本地代理验收**，并非腾讯 Hyra 官方同口径验收；太阳黑子/黑白棋是本地
  基线，不是官方赛制。
- **在 WorkBuddy 会话内，「自主闭环」的推理/提议确实由 WorkBuddy 已接入大模型（HY3）
  完成**——这是真实的「大模型当算力」递归自我改进（HY3 读分数→推理→改假设→沙盒验证）。
  无会话的无人值守跑则由 `GuidedProposer` 启发式兜底（诚实标注为启发式搜索，非 LLM 推理）。
- **太阳黑子多步预测**：最优模型 h=1/3/6/12 达 R²=0.90/0.86/0.84/0.77，全面优于
  持续性(0.87/0.79/0.71/0.48)与 11 年周期季节性朴素(0.37)基线；但模型是线性 AR+谐波
  **非深度学习**，短期因 persistence 已很高增益有限，价值在长视野。固定周期 naive
  反而弱（相位漂移）——这正是引入 torch/Transformer 的意义。请勿宣称
  「达到 Solar-TALENT 式官方赛制水平」（直接可比的官方太阳黑子多步 R² 未公开）。
- 共 **10 个本地代理任务**（5 医疗 + 太阳黑子 + 黑白棋 + 数学 + 量子），距 Hyra 官方
  15+ 类验收仍有差距。
- 合成药 `synthesis` 已升级为 **RDKit 真实片段组装 + QED 真实类药物性**（生成合法真实分子，
  如 `[H]C(=O)c1ccccc1NC=O` QED≈0.65，valid=True, novel=True），**仍非** de-novo Transformer
  生成；且 `activity` 仍为合成代理（无湿实验读数），请勿夸大。
- 数学 `mathlaw` 现已支持 **torch 真实训练引擎**（torch_sparse 梯度下降稀疏符号回归 /
  torch_transformer 小模型拟合），但仍是本地 harness 框架内的轻量演示，**不是** Hyra
  官方 15 参数加法 Transformer（那需大规模 GPU 训练与专用合成数据）。量子
  `quantum_routing` 是**启发式路由本地代理**；二者真值/基准均为合成或简化，
  **用于验证跨域自主发现闭环**。
  - 要真正复现 Hyra 的数学（15 参数加法 Transformer）/ 量子验收，仍需要大规模
    **GPU / torch 训练或 qiskit / 真实量子编译器**——这是诚实声明的边界。
- 五个医疗任务默认**合成埋设**真值，用于验证发现闭环；现已内建「真实数据适配层」
  (`hyra/realdata.py`) 与 `drugtarget/gen_real.py`（ChEMBL HIV-1 蛋白酶真实 IC₅₀），
  实测真实 QSAR HY3 驱动 **test R²≈0.42**（无人值守 0.39）；`synthesis` 真实 RDKit 化学亦已落地——二者均为**真实能力**，
  但**非**官方 vendor 验收，接入更大真实数据（LINCS L1000 / TCGA / ChEMBL 全量）需领域专家复核。
- 完整审计与诚实口径见 `harness/REPORT.md` 与 `harness/VERDICT.md`。
