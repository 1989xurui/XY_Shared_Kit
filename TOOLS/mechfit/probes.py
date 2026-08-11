"""
机制发现 · 探针集（结局已知的干预）——【合成示例数据，仅供演示方法】
=====================================================================
⚠⚠⚠ 重要：本文件里的 PROBES / CANDIDATES 全部是【虚构的合成示例】，
   不对应任何真实患者、真实用药或真实结局。它们只是为了让你：
     (1) 直接 `python mechfit.py` 能跑通、看到输出长什么样；
     (2) 理解"机制轴 + 结局编码 + 留一交叉验证 + 置换检验 + 硬门控"这套方法。
   要用于你自己的病例，请把 PROBES / CANDIDATES 换成【你自己的】真实干预史，
   并严格遵守下方"编码纪律"。

方法学出处：Hyra 太阳黑子任务的 `srfit` 官方评测器角色。
在那个任务里，任何搜索器都必须先能复现官方评测器的打分，才有资格排序候选。
在这里，任何机制模型都必须先能复现这 N 个"已经在患者身上跑过的实验"的结局，
才有资格给未试过的机制排序。

⚠ 编码纪律（防循环论证，对应 Hyra 的"见 valid 前写死规则"）：
   FEATURES 只依据药理学/生化机制填写，**不参考 outcome 列**。
   outcome 只在 fit/评估阶段被读取。
   任何一条特征若无法脱离结局独立论证，一律填 0。

数据来源（真实使用时）：你的患者数据库 response / failed_drug / adverse_event /
regimen 表 + 用药史底稿。逐条可回溯。
"""

# ---------------------------------------------------------------- 结局编码
# +1 = 部分有效（有可复现的具体获益）
#  0 = 无效（试足疗程、无可辨识变化）
# -1 = 失败/停用（无获益 + 明确副作用）
# -2 = 加重（核心症状恶化，可辨识）
OUTCOME = {"POS": 1.0, "NULL": 0.0, "FAIL": -1.0, "WORSE": -2.0}

# ---------------------------------------------------------------- 机制特征
# ⚠ 全部为【有符号轴】，不是 0/1 标志位。+1 = 沿该方向推，-1 = 沿该方向拉回。
#   理由（对应 Hyra 的教训"用错尺子会整族淘汰真赢家"）：
#   若把"降通量"另立一个新键，它在探针里采样数 = 0，权重不可辨识，
#   模型只能给 0 分，等于把候选直接判死。而"降通量"在生化上就是"升通量"的反方向，
#   同一根轴取负号即可从既有实验中继承信息。符号由机制决定，与结局无关。
#
# mono_push   : 单胺（5-HT/NE/DA）可利用量。+1 提高合成/阻断再摄取；-1 提高清除
# ox_flux     : 氧化代谢/高能磷酸通量。+1 推 PDH/KGDH/ETC/ATP 池；-1 抑制
# fuel_switch : 改变燃料种类（酮体/中链脂肪酸），而非增加同一燃料的量
# recep_block : 在突触受体层做硬拮抗
# gut_source  : 减少肠道/免疫来源的炎症输入（"关来源"）
# iron_gate   : 作用于 hepcidin / ferroportin 铁闸门（"开闸门"）
# iron_sub    : 单纯补铁底物（"加原料"）
# a2_tone     : α2A 肾上腺素能张力调节（不增加单胺总量，改变增益）
# nmda_mod    : 温和的突触/NMDA 位点调节（非硬拮抗）
# sedate      : 温和镇静 / GABA 能安抚
# behav_only  : 纯行为/感统训练，无药理作用
# leak_close  : 关闭 ATP 合酶 c 亚基质子泄漏（某发育障碍特异机制）
# mtor_down   : 下调 mTOR/MMP-9 通路（某病理主线）
#
# 最后两根轴在示例探针里采样数 = 0 —— 这是有意保留的，
# 目的就是让脚本把"从未采样过的机制轴"显式暴露出来，而不是偷偷补零。

FEATURES = ["mono_push", "ox_flux", "fuel_switch", "recep_block",
            "gut_source", "iron_gate", "iron_sub", "a2_tone",
            "nmda_mod", "sedate", "behav_only",
            "leak_close", "mtor_down"]

# name, outcome, 各特征, 结局判定所依据的终点(endpoint), 出处
# ⚠⚠ 以下 PROBES 为【合成虚构示例】，仅演示编码方式，请勿当作真实证据。
PROBES = [
    # ---- 加重组：全部落在 mono_push 上（演示"推单胺必加重"）----
    ("示例-单胺前体A", "WORSE",
     dict(mono_push=1),                       "情绪/冲动", "SYNTHETIC-DEMO"),
    ("示例-维生素B6活性型", "WORSE",
     dict(mono_push=1),                       "自语/入睡", "SYNTHETIC-DEMO"),
    ("示例-肌酸", "WORSE",
     dict(mono_push=1, ox_flux=1),            "情绪", "SYNTHETIC-DEMO"),
    ("示例-胆碱补充", "WORSE",
     dict(mono_push=1),                       "情绪", "SYNTHETIC-DEMO"),
    ("示例-复合多维(含B3)", "WORSE",
     dict(mono_push=1),                       "情绪", "SYNTHETIC-DEMO"),

    # ---- 失败组 ----
    ("示例-兴奋剂(甲基哌啶类)", "FAIL",
     dict(mono_push=1),                       "注意力", "SYNTHETIC-DEMO"),
    ("示例-非典型抗精神病药", "FAIL",
     dict(recep_block=1),                     "整体", "SYNTHETIC-DEMO"),

    # ---- 无效组 ----
    ("示例-感统训练3年", "NULL",
     dict(behav_only=1),                      "整体", "SYNTHETIC-DEMO"),
    ("示例-镁剂补充", "NULL",
     dict(nmda_mod=1),                        "整体", "SYNTHETIC-DEMO"),
    ("示例-IVIG×4次", "NULL",
     dict(gut_source=1),                      "整体", "SYNTHETIC-DEMO"),
    ("示例-无麸无酪饮食2年", "NULL",
     dict(gut_source=1),                      "整体", "SYNTHETIC-DEMO"),
    ("示例-低剂量纳曲酮", "NULL",
     dict(),                                  "整体", "SYNTHETIC-DEMO"),
    ("示例-口服铁剂", "NULL",
     dict(iron_sub=1),                        "整体/铁", "SYNTHETIC-DEMO"),
    ("示例-维生素D3单补", "NULL",
     dict(),                                  "整体", "SYNTHETIC-DEMO"),
    ("示例-牛磺酸", "NULL",
     dict(sedate=1),                          "整体", "SYNTHETIC-DEMO"),
    ("示例-茶氨酸", "NULL",
     dict(sedate=1),                          "整体", "SYNTHETIC-DEMO"),
    ("示例-α硫辛酸", "NULL",
     dict(ox_flux=1),                         "整体", "SYNTHETIC-DEMO"),
    ("示例-间歇性禁食", "NULL",
     dict(fuel_switch=1),                     "行为波动", "SYNTHETIC-DEMO"),
    ("示例-益生元(菊粉)", "NULL",
     dict(gut_source=1),                      "肠道/整体", "SYNTHETIC-DEMO"),

    # ---- 部分有效组（演示正向轴）----
    ("示例-中链脂肪酸油", "POS",
     dict(fuel_switch=1),                     "能量底盘", "SYNTHETIC-DEMO"),
    ("示例-锌剂", "POS",
     dict(nmda_mod=1),                        "刻板行为", "SYNTHETIC-DEMO"),
    ("示例-α2A调节剂", "POS",
     dict(a2_tone=1),                         "注意/觉醒", "SYNTHETIC-DEMO"),
]

# ---------------------------------------------------------------- 待评机制
# 尚未在患者身上试过、或试过但被判定依据存疑的候选。
# ⚠⚠ 合成示例：演示"候选打分 + 外推轴 + 硬门控"三种情形。
CANDIDATES = [
    ("示例-二甲双胍类",
     dict(iron_gate=1, ox_flux=-1, mtor_down=1),
     "复合体I抑制→降通量 + AMPK/SHP→↓hepcidin 开铁闸 + mTOR↓"),
    ("示例-乳铁蛋白",
     dict(iron_gate=1),
     "直接↓hepcidin，绕开口服铁被闸门锁死"),
    ("示例-核黄素B2(先测后用)",
     dict(ox_flux=1, mono_push=-1),
     "FAD 前体：一手推通量(+)，一手修复单胺清除(-)"),
    ("示例-线粒体泄漏阻断剂",
     dict(leak_close=1, ox_flux=-1),
     "关闭 ATP 合酶 c 亚基泄漏（某发育障碍机制药）"),
    ("示例-静脉铁剂",
     dict(iron_sub=1),
     "绕开肠道吸收，但不动闸门"),
    ("示例-抗IL-6R单抗",
     dict(iron_gate=1, gut_source=1),
     "上游关 IL-6→hepcidin，但为重武器"),
    ("示例-NAC",
     dict(nmda_mod=1),
     "xCT 调节谷氨酸 + 补 GSH"),
    ("示例-米诺环素",
     dict(gut_source=1, mtor_down=1),
     "MMP-9/小胶质，情绪 RCT p=0.049"),
    ("示例-SSRIs低剂量",
     dict(mono_push=1),
     "语言证据最强，但撞单胺红线（演示用）"),
    ("示例-GABA能镇静剂",
     dict(sedate=1),
     "演示用：若某患者红线禁止据此推导则否决"),
]

# ---------------------------------------------------------------- 硬门控
# 对应 Hyra `solve.sh` 的 fallback 保护 / reversi 的 arena 门控：
# 无论模型给多高分，撞到下面任何一条的候选一律不得进入推荐位。
# ⚠ 真实使用时，把这里换成【你自己的】红线裁决（例如某基因已判不解释病例）。
HARD_VETO = {
    "示例-SSRIs低剂量":
        "单胺慢代谢红线：推单胺方向在历史探针里全部加重，mono_push>0 一律否决",
    "示例-GABA能镇静剂":
        "知识库红线：某受体基因已判不解释本病例，禁止据此推导 GABA 能用药",
}

if __name__ == "__main__":
    print(f"探针 n={len(PROBES)}  候选 n={len(CANDIDATES)}")
    from collections import Counter
    print(Counter(p[1] for p in PROBES))
    bad = set()
    for _, f, _ in CANDIDATES:
        bad |= set(f) - set(FEATURES)
    for _, _, f, _, _ in PROBES:
        bad |= set(f) - set(FEATURES)
    print("特征空间越界键:", bad or "无 ✓")
