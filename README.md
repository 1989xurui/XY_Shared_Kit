# XY 科研发现工具箱 · 可共享项目包（已脱敏）

把一套"科学发现智能体 + 医学机制发现"的方法论、技能与工具整理成一个**可分发、不含任何真实患者隐私数据**的项目包。源自一个把腾讯混元 Hyra 范式迁移到
（a）时间序列预测、（b）单例儿科机制发现 的实践，已剔除全部受保护健康信息（PHI）。

> ⚠️ **隐私声明（重要）**
> 本包**不包含**任何真实患者的基因报告、用药反应、病历或知识库。
> - `SKILLS/medical-discovery-loop` 已脱敏为**通用模板**，原版嵌入的患者专属基因红线
>   与失败药清单被替换为 `<PLACEHOLDER>`，使用前必须由你自己填充。
> - `TOOLS/mechfit/probes.py` 里的 31 条探针是**合成示例（SYNTHETIC-DEMO）**，
>   仅用于演示方法，不对应任何真实患者。
> - 所有 `*.md` 报告与知识库均**未纳入**本包（它们含患者 PII）。
> 把本包共享给其它账号前，请确认你**没有**往里塞入任何真实病历。

---

## 目录结构

```
XY_Shared_Kit/
├── README.md                         ← 你正在读的文件
├── LICENSE                           ← MIT（可自由分享/修改，保留署名）
├── SKILLS/                           ← 5 个 WorkBuddy 技能（复制到 ~/.workbuddy/skills/ 即用）
│   ├── hyra-discovery/               # 本地无外部API科学发现harness（9个通用代理任务；已剔除2个患儿专属任务）
│   ├── medical-discovery-loop/       # 全领域机制扫描流水线【脱敏模板】
│   ├── trio-gvcf-cnv/                # 从trio gVCF读深度做CNV扫描+嵌合重查
│   ├── multiagent_lit_synthesis/     # 自家地基+外部检索的多智能体证据综合
│   └── research-agent-landscape-analyzer/  # 科研智能体可调用性/可学习性分析
├── TOOLS/
│   └── mechfit/                      # 机制发现框架（13轴岭回归+LOOCV+置换+硬门控）
│       ├── probes.py                 # ← 换成你自己的干预史（当前是合成示例）
│       ├── mechfit.py                # 拟合+排序（通用代码）
│       ├── plot.py                   # 画图（需 matplotlib）
│       ├── results.json              # 运行后生成的合成示例输出
│       └── README.md                 # 该框架的用法与替换数据说明
├── SOPS/                             ← 踩坑固化的标准作业流程（通用）
│   ├── vep_interpretation.md         # VEP 解读：MANE 陷阱三步法
│   ├── safe_delete_redlines.md       # 大批量删除的环境红线（Windows/WorkBuddy）
│   └── mechanism_discovery_workflow.md  # 机制发现 13 轴工作流
└── DOCS/
    └── methodology_overview.md       # 方法论总览与组件地图
```

---

## 安装与使用

### 技能（SKILLS/）
把需要的子目录**整个复制**到你的 WorkBuddy 技能目录：
- Windows：`C:\Users\<你>\.workbuddy\skills\`
- 其它：`~/.workbuddy/skills/`

例如：
```powershell
Copy-Item -Recurse XY_Shared_Kit\SKILLS\hyra-discovery  $env:USERPROFILE\.workbuddy\skills\
```
> 注意：`medical-discovery-loop` 是**模板**——复制后先编辑其 `SKILL.md`，
> 把 `<PLACEHOLDER>` 换成你自己的患者画像与红线，再启用。

### 机制发现框架（TOOLS/mechfit）
```bash
cd XY_Shared_Kit/TOOLS/mechfit
python mechfit.py     # 需 numpy；生成 results.json
python plot.py        # 可选，需 matplotlib；生成图
```
**换成你自己的数据**：编辑 `probes.py`，把 `PROBES` / `CANDIDATES` 换成你的真实
干预史，严守"编码纪律"（特征只按机制填、不参考结局）。详见 `TOOLS/mechfit/README.md`。

### SOP / 文档
直接阅读 `SOPS/` 与 `DOCS/`，按需套用。

---

## 诚实边界（务必传达给使用者）
- `hyra-discovery` 是**本地代理验收**，非腾讯 Hyra 官方同口径验收。
- `mechfit` 的排序在"已有实验无法区分候选"时会塌成 0，必须交外部证据。
- 任何医学输出都是**决策支持、非处方建议**，需临床/遗传医生裁决。
- 本包所有示例数据均为合成，不含真实患者信息。

---

## 分享方式
本目录本身就是一个完整项目。你可以：
- 整个文件夹压缩后发给他人；
- 或自行上传到你常用的平台（内部知识库 / 代码仓库 / 网盘）。
接收方按上面的"安装与使用"即可复用。

**请勿**在本包内添加任何真实患者数据后再分享。
