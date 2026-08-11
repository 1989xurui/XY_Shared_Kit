---
name: trio-gvcf-cnv
description: 从 trio(三人各自单样本) GATK gVCF 的读深度(DP)做全基因组 CNV 扫描 + 在指定 de
  novo/候选位点重查父母嵌合与 child 低等位基因平衡。适用于"已有 gVCF 但无 BAM/无临床 CMA"时，自主找出 CNV
  盲区与父母嵌合伪影。纯 Python stdlib，无需 pysam/bcftools。
disable: true
---

# trio-gvcf-cnv

## 何时用
- 手上有 trio WGS 的 **gVCF（每人一个单样本 .g.vcf.gz，含 FORMAT/DP）**，但缺少 BAM/CRAM 或临床染色体芯片（CMA）。
- 想自主做**读深度 CNV 扫描**（尤其 de novo CNV，如 GRIN2B、16p11.2、15q、22q11.2、SHANK3 等 NDD 热点），以及**重查 de novo 变异的父母嵌合 / child 低 AB 伪影**。

## 核心方法（已在 XY 家系实战验证）
1. 三份 gVCF 各流式读一遍（纯 gzip，~170k 行/s，1 亿行样本约 10min）。
2. 按 5kb 分箱累积 `DP × 长度`，得到每箱平均深度。
3. 每样本用**常染色体中位深度**归一化。
4. 计算 `M = log2(child_norm / comparator)`：
   - **常染色体** comparator = (Father+Mother)/2 → 偏离 ±0.4（≈1.32×/0.76×）且连续 ≥3 箱（≥15kb）判 CNV。
   - **chrX / chrY 关键修正**：患儿与父亲同为男性（单拷贝），comparator 只能取 **Father**，不能用 (Father+Mother)/2，否则整条 X 会被误判为缺失。
5. A8 顺带在指定位点（如 de novo 坐标）抓取三样本 AD/DP：重算 child `AB=alt/(ref+alt)`，查父母 alt 数。父母 alt=0 → 无可检测的父母生殖腺嵌合（但 gVCF 参考块看不到 <5% 低比例嵌合，需 BAM/ddPCR 才能定）。

## 用法
```bash
python scripts/gvcf_cnv_mosaic.py \
  --gvcf-dir /path/to/per_sample_gvcf \
  --out /path/to/outdir \
  --bin 5000 \
  --targets targets.json \
  --ndd-loci ndd_loci.json
```
- `targets.json`：`{"chr3:101565164":"TRMT10C", ...}`（要重查嵌合的位点）。
- `ndd-loci.json`：`[["16p11.2","chr16",29500000,30200000], ...]`（想重点看深度的 NDD 区间）。

输出：`gvcf_cnv_mosaic.json`（全部）、`gvcf_cnv_calls.tsv`（CNV 候选，按 |log2| 排序，带 NDD 重叠 flag）。

## 硬边界（务必如实告知用户）
- 读深度法，非 asm/BAF，分辨率约 15kb；**不能替代临床 CMA / 染色体芯片**。
- gVCF 参考块只存"已调用等位基因"：父母若在某位点是参考块(0/0)，gVCF 看不到低比例(<5%)父母嵌合——只能靠 BAM/深度重测序/ddPCR 定。
- 真胚系 de novo 的 child AB 应 ~0.5；若 0.1–0.3 提示 child 体细胞嵌合或 calling 伪影，需 Sanger/ddPCR 确认。
- 本流程产出候选，**非诊断**；最终需临床遗传医生 + CMA/IGV 验证。

## 环境坑（Windows + managed python）
- managed venv 在 `Scripts/` 不是 `bin/`：`.../python/envs/default/Scripts/python.exe`。
- pysam 可能装不上 → 本流程刻意不依赖 pysam，纯 stdlib 即可。
- 杀后台 python：PowerShell `Get-CimInstance Win32_Process | Where-Object{$_.CommandLine -like '*gvcf_cnv_mosaic*'} | Stop-Process -Force`（**勿用 %VAR% 语法**，会被安全拦截）。
