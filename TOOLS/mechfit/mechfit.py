"""
XY 机制发现 · 拟合器与候选排序
==============================
对应 Hyra 太阳黑子搜索器里的 `fit_mid` + `cv_eval` 角色。

纪律（逐条从 Hyra 那边搬过来，都是踩过坑换来的）：
  1. 【先量尺子再排序】任何机制模型必须先能复现 31 条已知结局，才有资格排候选。
     判据不是"训练集拟合得好"，而是 LOOCV 留一交叉验证优于"全预测均值"基线。
  2. 【防 reward hacking】加置换检验：把结局随机打乱 2000 次重拟合，
     若真实 LOOCV R² 落在置换分布里，说明模型只是在背 31 个点，直接作废。
  3. 【区分内插与外推】某根机制轴若在 31 条探针里采样数 = 0，其权重不可辨识。
     绝不偷偷补零后当成"预测到了"，必须单列为"外推轴"，另找外部证据。
  4. 【硬门控】撞红线的候选不得进推荐位，无论分数多高（对应 reversi 的 arena 门控）。
"""
import io, json, sys, random
import numpy as np
from probes import PROBES, CANDIDATES, FEATURES, OUTCOME, HARD_VETO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(20260810)
F = FEATURES
P = len(F)


def vec(d):
    return np.array([float(d.get(k, 0.0)) for k in F])


X = np.array([vec(p[2]) for p in PROBES])          # (n, P)
y = np.array([OUTCOME[p[1]] for p in PROBES])      # (n,)
names = [p[0] for p in PROBES]
n = len(y)

# ---------------------------------------------------------------- ① 轴采样覆盖度
support = (np.abs(X) > 0).sum(axis=0)
tested = support > 0
print("=" * 74)
print("① 机制轴采样覆盖度（31 条已跑过的干预分别踩到哪根轴）")
print("=" * 74)
for j, f in enumerate(F):
    m = np.abs(X[:, j]) > 0
    mu = y[m].mean() if m.any() else float("nan")
    bar = "█" * support[j]
    flag = "" if support[j] else "   ← 从未采样"
    print(f"  {f:<12} n={support[j]:<3} 均值结局={mu:+.2f}  {bar}{flag}")
untested_axes = [F[j] for j in range(P) if not tested[j]]
print(f"\n  未采样轴 = {untested_axes}")

# 每根轴实际被观测到的取值符号。这一步是 Hyra 教训的直接移植：
# 太阳黑子那边"单步 R² 高 ≠ 自由奔跑好"，根源就是拿采样区外的行为当已知。
# 这里同理：某轴若只见过 +1，那么候选取 -1 就是"镜像外推"，
# 它会被 100% 覆盖率伪装成内插，必须单独揪出来。
signs = {}
for j, f in enumerate(F):
    s = set(np.sign(X[:, j][np.abs(X[:, j]) > 0]).astype(int).tolist())
    signs[f] = s
print("\n  各轴已观测到的取值符号：")
for f in F:
    print(f"    {f:<12} {sorted(signs[f]) if signs[f] else '（无）'}")
one_sided = [f for f in F if signs[f] and len(signs[f]) == 1]
print(f"\n  ⚠ 单侧采样轴（只见过一个方向）= {one_sided}")
print("     含义：XY 这 31 次实验在这些轴上【只往一个方向推过，从未往回拉过】。")


# ---------------------------------------------------------------- ② 岭回归
def ridge_fit(Xtr, ytr, alpha):
    xb, yb = Xtr.mean(0), ytr.mean()
    Xc, yc = Xtr - xb, ytr - yb
    A = Xc.T @ Xc + alpha * np.eye(P)
    w = np.linalg.solve(A, Xc.T @ yc)
    w[~tested] = 0.0                    # 不可辨识轴强制归零，杜绝幻觉权重
    b = yb - xb @ w
    return w, b


def loo_r2(alpha, yy):
    pred = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        w, b = ridge_fit(X[m], yy[m], alpha)
        pred[i] = X[i] @ w + b
    ss_res = ((yy - pred) ** 2).sum()
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, pred


print("\n" + "=" * 74)
print("② 留一交叉验证选正则强度（基线 = 恒预测均值，R²=0）")
print("=" * 74)
best = (-9e9, None)
for a in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
    r2, _ = loo_r2(a, y)
    star = ""
    if r2 > best[0]:
        best, star = (r2, a), "  ←"
    print(f"  alpha={a:<6} LOOCV R² = {r2:+.4f}{star}")
ALPHA = best[1]
R2_LOO, PRED = loo_r2(ALPHA, y)
print(f"\n  选定 alpha={ALPHA}   LOOCV R² = {R2_LOO:+.4f}")
print(f"  基线(恒均值) R² = 0.0000   → 模型{'优于' if R2_LOO > 0 else '并未优于'}基线")

# ---------------------------------------------------------------- ③ 置换检验
NPERM = 2000
perm_r2 = np.empty(NPERM)
yp = y.copy()
for k in range(NPERM):
    RNG.shuffle(yp)
    perm_r2[k], _ = loo_r2(ALPHA, yp)
pval = (1 + (perm_r2 >= R2_LOO).sum()) / (1 + NPERM)
print("\n" + "=" * 74)
print("③ 置换检验（结局随机打乱 2000 次，看模型是不是在背答案）")
print("=" * 74)
print(f"  置换 LOOCV R² 分布: 均值{perm_r2.mean():+.4f} "
      f"95分位{np.percentile(perm_r2, 95):+.4f} 最大{perm_r2.max():+.4f}")
print(f"  真实 R² = {R2_LOO:+.4f}   p = {pval:.4f}"
      f"   → {'通过，非偶然' if pval < 0.05 else '未通过，模型作废'}")

# ---------------------------------------------------------------- ④ 权重 + 自助置信区间
W, B = ridge_fit(X, y, ALPHA)
NB = 2000
Wb = np.zeros((NB, P))
for k in range(NB):
    idx = RNG.integers(0, n, n)
    Wb[k], _ = ridge_fit(X[idx], y[idx], ALPHA)
lo, hi = np.percentile(Wb, 5, axis=0), np.percentile(Wb, 95, axis=0)
print("\n" + "=" * 74)
print("④ 机制轴权重（+ = 沿该方向推有利，- = 沿该方向推有害）")
print("=" * 74)
order = np.argsort(-W)
for j in order:
    if not tested[j]:
        continue
    sig = "✔" if (lo[j] > 0 or hi[j] < 0) else " "
    print(f"  {sig} {F[j]:<12} w={W[j]:+.3f}  90%CI[{lo[j]:+.3f},{hi[j]:+.3f}]  n={support[j]}")
print(f"    截距 b = {B:+.3f}")
for j in range(P):
    if not tested[j]:
        print(f"  ? {F[j]:<12} 不可辨识（采样数 0），权重强制 0，不参与打分")

# ---------------------------------------------------------------- ⑤ 候选打分
print("\n" + "=" * 74)
print("⑤ 候选机制打分（结局刻度：+1 部分有效 / 0 无效 / -1 失败 / -2 加重）")
print("=" * 74)
rows = []
for cname, cf, note in CANDIDATES:
    x = vec(cf)
    ext = [F[j] for j in range(P) if x[j] != 0 and not tested[j]]
    flip = [F[j] for j in range(P)
            if x[j] != 0 and tested[j] and int(np.sign(x[j])) not in signs[F[j]]]
    # 乐观分：相信线性可外推到反方向
    score = float(x @ W + B)
    bs = Wb @ x + B
    clo, chi = float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
    # 保守分：把符号外推的贡献也一并归零，只留真正内插过的部分
    xc = x.copy()
    for f in flip:
        xc[F.index(f)] = 0.0
    score_cons = float(xc @ W + B)
    tier = ("C 轴外推" if ext else ("B 符号外推" if flip else "A 内插"))
    rows.append(dict(name=cname, score=score, score_cons=score_cons,
                     lo=clo, hi=chi, tier=tier,
                     extrapolated=ext, sign_flip=flip, note=note,
                     veto=HARD_VETO.get(cname),
                     feats={F[j]: float(x[j]) for j in range(P) if x[j] != 0}))

rows.sort(key=lambda r: -r["score"])
for i, r in enumerate(rows, 1):
    tag = "🚫否决" if r["veto"] else {"A 内插": "✅", "B 符号外推": "⚠", "C 轴外推": "⚠"}[r["tier"]]
    print(f"\n {i:>2}. {r['name']}   乐观 {r['score']:+.2f} "
          f"[{r['lo']:+.2f},{r['hi']:+.2f}]   保守 {r['score_cons']:+.2f}   "
          f"{tag} 证据级别 {r['tier']}")
    print(f"     轴: {r['feats']}")
    print(f"     {r['note']}")
    if r["sign_flip"]:
        print(f"     ⚠ 符号外推 {r['sign_flip']}：该轴只观测过相反方向，"
              f"本分靠【线性可反推】假设，未经检验")
    if r["extrapolated"]:
        print(f"     ⚠ 轴外推 {r['extrapolated']}：XY 从未采样，本分数未包含其贡献")
    if r["veto"]:
        print(f"     🚫 硬门控：{r['veto']}")

print("\n  —— 只看真正内插过的证据（保守分）时的排序 ——")
for r in sorted([r for r in rows if not r["veto"]], key=lambda r: -r["score_cons"]):
    print(f"    {r['score_cons']:+.2f}  {r['name']}  [{r['tier']}]")
print("  → 若保守分把所有候选压到 0 附近，说明【XY 已有的 31 次实验无法区分它们】，"
      "\n    排序必须交给外部证据 + 信息增益，而不是这个模型。")

# ---------------------------------------------------------------- ⑥ 信息增益
print("\n" + "=" * 74)
print("⑥ 实验信息量排序（做哪个实验最能减少不确定性）")
print("=" * 74)
info = []
for r in rows:
    if r["veto"]:
        continue
    gain = sum(1.0 / (1.0 + support[F.index(k)]) for k in r["feats"])
    info.append((gain, r["score"], r["name"], r["extrapolated"]))
info.sort(reverse=True)
for g, s, nm, ex in info:
    print(f"  信息增益 {g:.2f}  预测 {s:+.2f}  {nm}"
          + (f"   （首次点亮 {ex}）" if ex else ""))

out = dict(n_probes=n, alpha=ALPHA, loo_r2=R2_LOO, perm_p=float(pval),
           intercept=float(B),
           weights={F[j]: dict(w=float(W[j]), lo=float(lo[j]), hi=float(hi[j]),
                               n=int(support[j])) for j in range(P)},
           untested_axes=untested_axes, candidates=rows)
io.open("results.json", "w", encoding="utf-8").write(
    json.dumps(out, ensure_ascii=False, indent=2))
print("\n→ results.json 已写出")
