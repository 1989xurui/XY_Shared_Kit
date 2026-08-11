# SOP · VEP 变异解读（MANE 陷阱）

来源：本项目在 SATB1、MEF2C 两个位点两次踩坑后固化的解读流程。
适用：用 Ensembl VEP（含 REST API）解读错义/剪接/非编码变异时。

## 核心陷阱
`most_severe_consequence` 是**跨全部转录本取最坏**，其中包含非 MANE 的小转录本
（如某些只出现在 3 个非 MANE 转录本的 splice_acceptor）。它**是陷阱**，会夸大严重性。

## 必做三步（缺一不可）
1. **报 MANE Select 后果 + HGVS**：以 MANE Select 转录本为准，给出其 consequence
   与蛋白/ cDNA 变化（如 `5'UTR c.-125G>A`，后果 MODIFIER）。
2. **报来源转录本的 MANE / canonical 状态**：说明该后果落在哪个转录本、
   是否 MANE Select。非 MANE 的后果权重低。
3. **报全转录本后果分布**：列出所有转录本上的后果，标注哪些是 MANE、哪些非。

## 禁止
- **禁止只报 `most_severe_consequence`**。它跨全部转录本取最坏，含非 MANE 小转录本，
  会系统性高估致病性。
- 非 MANE 转录本上的严重后果（如 splice_acceptor）若无 MANE 支持，不能单独作为
  致病判据；需用独立建库 / 双链 UMI / dPCR 等验证其存在（可能是建库伪迹）。

## 实测加固案例（本项目）
- SATB1 (chr3:18423727 C>T)：MANE Select NM_002971.6 = 5'UTR c.-125G>A (MODIFIER)，
  splice_acceptor 仅在 3 个非 MANE 转录本出现，且仅 1/40 单链支持 → 判为建库伪迹，
  降为"只做技术排伪，不能称致因"。
- MEF2C 易位断点 (chr5:88799918)：MANE ENST00000504921 = 深内含子 c.258+4680T>A，
  原称"落在外显子"属非 MANE 转录本 → 机制改述为基因体破坏/单倍剂量不足。
