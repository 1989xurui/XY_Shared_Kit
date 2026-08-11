"""Math scientific-discovery task: rediscover a hidden physics law.

LOCAL SYMBOLIC-REGRESSION PROXY -- extended with OPTIONAL real torch engines.

The harness generates data from Kepler's third law T = a**1.5 (plus a nuisance
feature z and noise). The genome selects which *transformations* of the raw
columns to feed the fit. The unattended loop must rediscover that the a**1.5
transform (and an intercept) explains the data, and reject the useless z.

Three engines (the last two require torch, which is only offered in
`genome_space()` when torch imports successfully in the running interpreter):

  * numpy            -- pure-Python least squares + adjusted-R^2 parsimony
                        (the original local proxy; runs anywhere, no deps)
  * torch_sparse    -- REAL torch training: linear model over the candidate
                        basis functions, trained with Adam + L1 sparsity so the
                        useless terms are driven to ~0 weights automatically.
                        This is a *real gradient-descent symbolic-regression*
                        loop, not the numpy closed-form solve.
  * torch_transformer -- REAL torch training: a tiny Transformer (attention over
                        the feature tokens) that learns the a -> T mapping
                        end-to-end. Demonstrates a genuine deep-learning training
                        loop; it *fits* the law but does not output a symbolic
                        expression (honest boundary: deep nets are not symbolic
                        discovery).

Model selection uses the VALIDATION set only; the test set is reported once
and never used to choose -- preventing reward hacking.
"""
import os
import json
import math
import importlib

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "dataset.csv")
WORK_ROOT = os.path.join(_HERE, "solutions")
EB_DIR = os.path.join(_HERE, "eb_hy3")
SANDBOX_TIMEOUT = 120


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# numpy engine: pure-Python least squares + adjusted-R^2 parsimony
# --------------------------------------------------------------------------
SOLUTION_TMPL = '''import math, json, csv

USE = {use}
USE_Z = {use_z}
INTERCEPT = {intercept}

def features(row):
    a = float(row[0]); z = float(row[1])
    f = []
    if USE.get("t_raw"): f.append(a)
    if USE.get("t_sqrt"): f.append(math.sqrt(abs(a)) if a >= 0 else 0.0)
    if USE.get("t_pow15"): f.append((a ** 1.5) if a >= 0 else 0.0)
    if USE.get("t_log"): f.append(math.log(abs(a)) if a > 0 else 0.0)
    if USE.get("t_sq"): f.append(a * a)
    if USE_Z: f.append(z)
    if INTERCEPT: f.append(1.0)
    return f

def solve_lstsq(X, y):
    n = len(X[0])
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for row, yi in zip(X, y):
        for i in range(n):
            b[i] += row[i] * yi
            for j in range(n):
                A[i][j] += row[i] * row[j]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        d = A[col][col]
        if abs(d) < 1e-12:
            continue
        for j in range(col, n):
            A[col][j] /= d
        b[col] /= d
        for r in range(n):
            if r != col and A[r][col] != 0:
                ff = A[r][col]
                for j in range(col, n):
                    A[r][j] -= ff * A[col][j]
                b[r] -= ff * b[col]
    return b

def solve(rows, k, m):
    X = [[float(v) for v in features(r)] for r in rows]
    y = [float(r[2]) for r in rows]
    Xtr, Xva, Xte = X[:k], X[k:m], X[m:]
    Ytr, Yva, Yte = y[:k], y[k:m], y[m:]
    theta = solve_lstsq(Xtr, Ytr)
    p = len(Xtr[0])
    def r2(pred, Y):
        if not Y: return 0.0
        ym = sum(Y) / len(Y)
        ssr = sum((yi - pi) ** 2 for yi, pi in zip(Y, pred))
        sst = sum((yi - ym) ** 2 for yi in Y)
        return 1 - ssr / sst if sst > 1e-12 else 0.0
    pva = [sum(a * b for a, b in zip(theta, x)) for x in Xva]
    pte = [sum(a * b for a, b in zip(theta, x)) for x in Xte]
    val_r2 = r2(pva, Yva); test_r2 = r2(pte, Yte)
    n = len(Yva)
    val_adj = 1 - (1 - val_r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else 0.0
    sel = [kk for kk, v in USE.items() if v] + (["z"] if USE_Z else []) + (["intercept"] if INTERCEPT else [])
    return {{"score": val_adj, "val_r2": val_r2, "test_r2": test_r2,
            "n_features": p, "selected": sel}}
'''


# --------------------------------------------------------------------------
# torch_sparse engine: REAL gradient-descent symbolic regression with L1
# --------------------------------------------------------------------------
TORCH_SPARSE_TMPL = '''import math, json, csv
import torch

USE = {use}
USE_Z = {use_z}
INTERCEPT = {intercept}

def features(row):
    a = float(row[0]); z = float(row[1])
    f = []
    if USE.get("t_raw"): f.append(a)
    if USE.get("t_sqrt"): f.append(math.sqrt(abs(a)) if a >= 0 else 0.0)
    if USE.get("t_pow15"): f.append((a ** 1.5) if a >= 0 else 0.0)
    if USE.get("t_log"): f.append(math.log(abs(a)) if a > 0 else 0.0)
    if USE.get("t_sq"): f.append(a * a)
    if USE_Z: f.append(z)
    if INTERCEPT: f.append(1.0)
    return f

NAMES = [kk for kk, v in USE.items() if v] + (["z"] if USE_Z else []) + (["intercept"] if INTERCEPT else [])

def solve(rows, k, m):
    X = torch.tensor([[float(v) for v in features(r)] for r in rows], dtype=torch.float32)
    y = torch.tensor([float(r[2]) for r in rows], dtype=torch.float32)
    # standardize features so L1 sparsity is fair across columns
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Xn = (X - Xm) / Xs
    Ym, Ys = y.mean(), y.std() + 1e-8
    Yn = (y - Ym) / Ys
    Xtr, Xva, Xte = Xn[:k], Xn[k:m], Xn[m:]
    Ytr, Yva, Yte = Yn[:k], Yn[k:m], Yn[m:]
    w = torch.zeros(Xn.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(4000):
        opt.zero_grad()
        pred = Xtr @ w
        loss = ((pred - Ytr) ** 2).mean() + 1e-2 * w.abs().sum()  # L1 sparsity
        loss.backward(); opt.step()
    def r2(p, Y):
        ym = Y.mean(); return 1 - ((Y - p) ** 2).sum() / ((Y - ym) ** 2).sum()
    with torch.no_grad():
        val_r2 = float(r2(Xva @ w, Yva)); test_r2 = float(r2(Xte @ w, Yte))
        wa = w.detach().abs()
        thr = 0.1 * wa.max().item()          # relative sparsity threshold
        nz = wa > thr
    sel = [NAMES[i] for i in range(len(NAMES)) if bool(nz[i])]
    return {{"score": val_r2, "val_r2": val_r2, "test_r2": test_r2,
            "n_features": int(nz.sum().item()), "selected": sel}}
'''


# --------------------------------------------------------------------------
# torch_transformer engine: REAL tiny Transformer that fits a -> T
# (honest note: it learns the mapping, it does not emit a symbolic law)
# --------------------------------------------------------------------------
TORCH_TRANSFORMER_TMPL = '''import json, csv
import torch
import torch.nn as nn

USE_Z = {use_z}

def solve(rows, k, m):
    feats = []
    for r in rows:
        f = [float(r[0])]
        if USE_Z: f.append(float(r[1]))
        feats.append(f)
    X = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor([[float(r[2])] for r in rows], dtype=torch.float32)
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    ym, ys = y.mean(), y.std() + 1e-8
    Xn, yn = (X - Xm) / Xs, (y - ym) / ys
    Xtr, Xva, Xte = Xn[:k], Xn[k:m], Xn[m:]
    Ytr, Yva, Yte = yn[:k], yn[k:m], yn[m:]
    d = Xn.shape[1]

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Linear(1, 16)            # each feature -> a token
            self.tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(16, 2, 32, batch_first=True), 2)
            self.head = nn.Linear(16, 1)
        def forward(self, x):
            x = self.tok(x.unsqueeze(-1))          # (B, d, 16) tokens
            x = self.tf(x)                         # attention over feature tokens
            x = x.mean(1)                          # pool
            return self.head(x)

    net = Tiny()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(1000):
        opt.zero_grad()
        loss = ((net(Xtr) - Ytr) ** 2).mean()
        loss.backward(); opt.step()
    def r2(p, Y):
        ym = Y.mean(); return 1 - ((Y - p) ** 2).sum() / ((Y - ym) ** 2).sum()
    with torch.no_grad():
        val_r2 = float(r2(net(Xva), Yva)); test_r2 = float(r2(net(Xte), Yte))
    return {{"score": val_r2, "val_r2": val_r2, "test_r2": test_r2,
            "n_features": d, "selected": ["transformer_fit(a->T)"]}}
'''


RUNNER = '''import solution, json, csv

DATA = "{data}"

def load():
    rows = []
    with open(DATA) as f:
        for line in csv.reader(f):
            if not line or line[0] == "a":
                continue
            rows.append([float(x) for x in line])
    return rows

rows = load()
n = len(rows); k = int(n * 0.6); m = int(n * 0.8)
out = solution.solve(rows, k, m)
print(json.dumps(out))
'''


class MathLawTask:
    def __init__(self):
        self.columns = ["a", "z", "T"]
        self._torch = _torch_available()

    def workdir(self, i):
        d = os.path.join(WORK_ROOT, f"iter_{i}")
        os.makedirs(d, exist_ok=True)
        return d

    # ---- genome interface (auto_loop) ----
    def genome_space(self):
        space = {
            "t_raw": [False, True],
            "t_sqrt": [False, True],
            "t_pow15": [False, True],
            "t_log": [False, True],
            "t_sq": [False, True],
            "use_z": [False, True],
            "intercept": [False, True],
        }
        if self._torch:
            space["engine"] = ["numpy", "torch_sparse", "torch_transformer"]
        else:
            space["engine"] = ["numpy"]
        return space

    def seed_genome(self):
        # deliberately weak / over-complete start: include everything, numpy
        g = {k: True for k in self.genome_space() if k != "engine"}
        g["engine"] = "numpy"
        return g

    def render(self, genome):
        if isinstance(genome, dict):
            use = {k: bool(genome.get(k, False)) for k in
                   ["t_raw", "t_sqrt", "t_pow15", "t_log", "t_sq"]}
            use_z = bool(genome.get("use_z", False))
            intercept = bool(genome.get("intercept", True))
            engine = genome.get("engine", "numpy")
        else:
            use = {"t_raw": True, "t_sqrt": True, "t_pow15": True,
                   "t_log": True, "t_sq": True}
            use_z = True
            intercept = True
            engine = "numpy"

        if engine == "torch_sparse":
            code = TORCH_SPARSE_TMPL.format(use=use, use_z=use_z, intercept=intercept)
            sel = [k for k, v in use.items() if v]
            if use_z:
                sel.append("z")
            if intercept:
                sel.append("b")
            summary = "torch_sparse feats=" + ",".join(sel)
        elif engine == "torch_transformer":
            code = TORCH_TRANSFORMER_TMPL.format(use_z=use_z)
            summary = "torch_transformer z=" + str(use_z)
        else:
            code = SOLUTION_TMPL.format(use=use, use_z=use_z, intercept=intercept)
            sel = [k for k, v in use.items() if v]
            if use_z:
                sel.append("z")
            if intercept:
                sel.append("b")
            summary = "numpy feats=" + ",".join(sel)

        return code, summary

    def runner_code(self):
        return RUNNER.format(data=DATA.replace("\\", "/"))

    def parse_run(self, run):
        if not run.get("ok"):
            return {"ok": False, "score": None, "stderr": run.get("stderr")}
        lines = [l for l in run["stdout"].splitlines() if l.strip()]
        if not lines:
            return {"ok": False, "score": None, "stderr": "empty stdout"}
        try:
            obj = json.loads(lines[-1])
            return {
                "ok": True,
                "score": obj.get("score"),
                "val_r2": obj.get("val_r2"),
                "test_r2": obj.get("test_r2"),
                "n_features": obj.get("n_features"),
                "selected": obj.get("selected"),
            }
        except Exception:
            return {"ok": False, "score": None, "stderr": run["stdout"][-200:]}
