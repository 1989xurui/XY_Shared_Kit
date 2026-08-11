import json, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

R = json.loads(io.open("results.json", encoding="utf-8").read())
W = R["weights"]

fig = plt.figure(figsize=(15.5, 6.4))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.72, 1.35], wspace=0.42,
                      left=0.055, right=0.985, top=0.855, bottom=0.115)

# ---------------- Panel A : 机制轴权重
ax = fig.add_subplot(gs[0])
items = [(k, v) for k, v in W.items() if v["n"] > 0]
items.sort(key=lambda kv: kv[1]["w"])
ys = np.arange(len(items))
ws = [v["w"] for _, v in items]
lo = [v["w"] - v["lo"] for _, v in items]
hi = [v["hi"] - v["w"] for _, v in items]
cols = ["#c0392b" if w < -0.25 else ("#27ae60" if w > 0.25 else "#95a5a6") for w in ws]
ax.barh(ys, ws, color=cols, height=0.62, zorder=3)
ax.errorbar(ws, ys, xerr=[lo, hi], fmt="none", ecolor="#2c3e50",
            elinewidth=1.3, capsize=3, zorder=4)
ax.axvline(0, color="#2c3e50", lw=1.1, zorder=2)
ax.set_yticks(ys)
ax.set_yticklabels([f"{k}  (n={v['n']})" for k, v in items], fontsize=9.5)
ax.set_xlabel("权重 w  （+ 有利 / − 有害）", fontsize=10)
ax.set_title("A. 从 XY 自己的 31 次实验学到的机制轴权重\n"
             f"LOOCV R²={R['loo_r2']:+.3f}  置换检验 p={R['perm_p']:.4f}",
             fontsize=11.5, weight="bold", pad=9)
ax.grid(axis="x", alpha=0.25, zorder=0)
ax.text(-1.50, len(items) - 1.42, "唯一 90%CI 不跨 0 的轴\n= 推单胺必加重",
        fontsize=8.6, color="#c0392b", weight="bold", ha="left", va="center")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---------------- Panel B : 采样覆盖度
ax = fig.add_subplot(gs[1])
keys = list(W.keys())
ns = [W[k]["n"] for k in keys]
o = np.argsort(ns)
ys = np.arange(len(keys))
cs = ["#e74c3c" if ns[i] == 0 else "#34495e" for i in o]
ax.barh(ys, [ns[i] for i in o], color=cs, height=0.62, zorder=3)
ax.set_yticks(ys)
ax.set_yticklabels([keys[i] for i in o], fontsize=9.5)
for i, idx in enumerate(o):
    if ns[idx] == 0:
        ax.text(0.12, i, "从未采样", va="center", fontsize=8.8,
                color="#e74c3c", weight="bold")
ax.set_xlabel("31 次干预中踩到该轴的次数", fontsize=10)
ax.set_title("B. 机制轴采样覆盖度\n盲区：3 根轴 8 年从未碰过",
             fontsize=11.5, weight="bold", pad=9)
ax.grid(axis="x", alpha=0.25, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---------------- Panel C : 候选排序
ax = fig.add_subplot(gs[2])
rows = [r for r in R["candidates"]]
rows.sort(key=lambda r: r["score"])
ys = np.arange(len(rows))
h = 0.36
tc = {"A 内插": "#27ae60", "B 符号外推": "#e67e22", "C 轴外推": "#2980b9"}
ax.barh(ys + h / 2, [r["score"] for r in rows], height=h, zorder=3,
        color=[("#7f8c8d" if r["veto"] else tc[r["tier"]]) for r in rows],
        label="乐观分（相信线性外推）")
ax.barh(ys - h / 2, [r["score_cons"] for r in rows], height=h, zorder=3,
        color=[("#bdc3c7" if r["veto"] else tc[r["tier"]]) for r in rows],
        alpha=0.42, label="保守分（只算真正内插过的证据）")
ax.axvline(0, color="#2c3e50", lw=1.1, zorder=2)
labs = []
for r in rows:
    m = "[否决] " if r["veto"] else ""
    labs.append(f"{m}{r['name']}")
ax.set_yticks(ys)
ax.set_yticklabels(labs, fontsize=9.3)
ax.set_xlabel("预测结局  (+1 部分有效 / 0 无效 / −1 失败 / −2 加重)", fontsize=10)
ax.set_title("C. 候选机制打分：乐观 vs 保守\n两者一分开，本地模型就排不出名次了",
             fontsize=11.5, weight="bold", pad=9)
ax.legend(fontsize=8.6, loc="lower right", framealpha=0.94)
ax.grid(axis="x", alpha=0.25, zorder=0)
ax.set_xlim(-1.95, 1.95)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
hp = [plt.Rectangle((0, 0), 1, 1, color=v) for v in tc.values()]
ax.legend(hp + [plt.Rectangle((0, 0), 1, 1, color="#7f8c8d")],
          list(tc.keys()) + ["硬门控否决"],
          fontsize=8.4, loc="lower right", framealpha=0.94, ncol=2)

fig.suptitle("XY 机制发现 · 把 8 年 31 次干预当作评测器，反推还没试过的机制",
             fontsize=13.5, weight="bold", y=0.975)
fig.savefig("xy_mech.png", dpi=155, facecolor="white")
print("saved xy_mech.png")
