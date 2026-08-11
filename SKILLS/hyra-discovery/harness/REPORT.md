# Hyra-Local — 项目状态报告（诚实版）

> ⚠️ **诚实声明（务必先读）**：截至本版，**未通过腾讯混元 Hyra 官方同口径验收**。
> 本框架是一个工程骨架正确、可运行、有真实数值的研究循环脚手架，但当前所有"验收通过"均建立在
> **本地代理 + 不同口径对比 + 进化式超参搜索**之上，与官方同标准不可比。
> 详见 `VERDICT.md`（13 维独立审计，13/13 完成，均值 ≈4.6/10；PASS 1 / FAIL 2 / PARTIAL 10）。

## 目标
自研 Hunyuan Research Agent（Hyra）式 Harness 框架，以 WorkBuddy 自带 HY3 作为可调度算力，
尽量对齐腾讯混元 Hyra「科学发现智能体」能力水平，并尽量对齐其官网验收页
（https://hy.tencent.com/research/hyra）列出的测试。

## "HY3 驱动 / RSI / 蒸馏"的诚实定位（已正名）
此前的文档把下面两件事混为一谈，现予拆分：

1. **手动推理链（真实 HY3 推理，但非无人值守）**：`hy3_step.py` 在**本会话内**由 HY3（我）逐步提出
   物理假设、写 `solution.py`、沙盒打分、入库。太阳黑子 h0–h4 假设链是这么来的——这是**人工逐步驱动**，
   **不是**连续自主循环。
2. **无人值守循环（auto_loop.py，机制真实但"自主"=启发式）**：用的是 `GuidedProposer`
   （最优个体引导变异 + top-2 交叉 + 随机探索 + 去重 + warm-start），**不是 HY3 推理**，
   也**不调用任何外部 API**。`AgentBridge.responder` 只是把"已本地决定的基因组"原样回显并记录到审计
   JSON，定位为**审计桩（no-op）**，不是智能。

> 结论：所谓"连续 agent loop 自主推进"在**机制层面真实存在**（循环会自己跑、自己打分、自己累积、
> 跨会话续跑），但**"自主"指启发式搜索，不是自主推理**。"RSI 蒸馏 HY3""零人工干预 RSI"等表述
> 已被审计驳回，本版删除。

## 框架组件 ↔ 实际角色
| Hyra 组件 | Hyra-Local 实现 | 实际角色 |
|---|---|---|
| Harness（轻量框架） | `hyra/harness.py` | 编排脚手架 |
| ContextAgent / ProposalAgent | `hyra/agents.py` | 接口占位；**真正提议来自 GuidedProposer** |
| ExperienceBank（经验库） | `hyra/experience_bank.py` | 跨迭代/跨会话累积基因组与分数（真实可用） |
| 沙盒（隔离执行+打分） | `hyra/sandbox.py` | subprocess 隔离 + 进程组清理（加固中） |
| 双层循环（Evaluator 防 reward hacking） | `hyra/evaluator.py` | 基于 val-test 差+复杂度做真实惩罚（已接入 auto_loop） |
| 推理算力 | `hyra/llm_bridge.py` | MockBridge（无 LLM 跑通）/ AgentBridge（审计桩，responder 仅 no-op 回显） |

## 验收测试 1：太阳黑子预测（本地代理，非官方同口径）
- 数据：SILSO 月度太阳黑子数 1749–2026（真实公开数据，已本地化）。
- 划分：训练 1749–1931；**验证集**（train 池内 40%，用于模型选择）；样本外**测试** 1932–2026（仅最终报告）。
- 手动 HY3 推理链（hy3_step 逐步驱动，非无人值守）：

| 步 | HY3 假设 | 样本外 R² | 处理 |
|---|---|---|---|
| 进化基线 | 随机变异 | 0.8971 | 基线 |
| h0 | 周期谐波 132/66/264 月 | 0.8973 | ✓ |
| h3 | 相位×近期值 交互 | 0.8976 | ✓ |
| **h4** | **sqrt(lag) 方差稳定特征** | **0.8987** | ✓ 手动链最佳 |

- **无人值守 auto_loop（GuidedProposer 启发式）**：从弱种子 0.8945 自主爬升，**iter3 即达 0.9007**，
  后续零增益（warm-start 超参爬山，非质变）。跨进程 warm-start 续跑 24→30 条稳定 0.9007。
- **诚实口径（修复后实测）**：选模 `val_r2≈0.82`；报告 `test` 单步≈0.895；**多步 `h=12≈0.36`**；
  平凡基线 `persistence_val≈0.873`。
  → 单步 0.90 看似高，但**多步预测骤降到 0.36**，且**与官方 R²≈0.77 不同口径（官方多步/长视野），
  "远超"不成立**。

## 验收测试 2：黑白棋 / Reversi bot（本地代理）
- 无法复现 Hyra 原赛制，定义可复现**本地代理**：击败某基线 ≥65%。
- **旧基线（mobility-greedy，弱）**：bot_h0 位置权重 12/12=100% → 统计功效≈0，无意义。
- **修复后强基线（alpha-beta d1–3）+ 随机开局 ≥50 局 + Wilson 区间**：
  - bot_h0(位置权重) 对 alphabeta-d2 仅 **20% 胜率**（诚实暴露弱）。
  - bot_h1(alpha-beta d3) 对强基线 **90%**（Wilson 95% 下界≈0.70）。
- 结论：框架能诚实报强/弱，**但仍非 top-3 赛制，不能等同官方通过**。

## 13 维审计结论（摘要，详见 VERDICT.md）
- 13/13 完成，均值 ≈4.6/10；PASS 1（D6 无外部 API）、FAIL 2（D9 奖励黑客已修 / D12 对齐 0/12 本地代理）、其余 PARTIAL。
- 三大硬伤（D12 对齐 0/12、D9 奖励黑客空转、D1/D5/D13 空壳包装）已于 2026-07-22 第二轮动手修复并验证。

## 对齐验收页 12 类状态
| 类别 | 测试 | Hyra-Local 状态 |
|---|---|---|
| AI for Science | 太阳黑子 R²≈0.77 | 本地代理：单步≈0.895 / 多步 h12≈0.36（非官方同口径） |
| AI for Fun | 黑白棋/Reversi bot top-3 | 本地代理：强基线 90%(Wilson lb≈0.70)，非 top-3 赛制 |
| 其余 10 类 | NanoChat/SOL/NanoGPT/数学开放题/15参Transformer/量子路由/PARP1/ScalingLaw/其他棋类 | 框架可承载，未实现、无证据 |

## 运行 / 复现
```bash
PY=C:/Users/ZhuanZ/.workbuddy/binaries/python/versions/3.13.12/python.exe
cd C:/HY3/hyra_local
# 无人值守循环（GuidedProposer 启发式 + warm-start，非 HY3 推理，无外部 API）
$PY auto_loop.py tasks.sunspot.task --iters 24 --fresh   # 从弱种子自主爬升
$PY auto_loop.py tasks.sunspot.task --iters 6            # 续跑：warm_start 跨会话累积
# 单步手动 HY3 驱动（本会话 HY3 逐步推理，非无人值守）
$PY hy3_step.py tasks.sunspot.task  tasks/sunspot/hy3_proposals/sol_h4.py "h4" h4
$PY hy3_step.py tasks.othello.task  tasks/othello/hy3_proposals/bot_h1.py "bot_h1" h1
# 旧版纯进化基线（对照）
$PY tasks/sunspot/run.py
```

## 下一步（真实欠账，按优先级）
1. 太阳黑子复现官方**多步**口径 + 强基线对照（当前 h12≈0.36 偏弱）。
2. 黑白棋 bot_h1 能否诚实通过强基线（验证中）。
3. 去硬编码 `C:/HY3` → `__file__` 推导；补 `requirements.txt` + git 化（D8）。
4. 沙盒加固：内存/CPU/文件系统限制（D7）；补最小冒烟测试（D13）。
5. 扩展其余 10 类（需领域数据与算力）。
