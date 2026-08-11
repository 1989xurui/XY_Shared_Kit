"""Sunspot task: out-of-sample R^2 forecasting of monthly sunspot numbers.

LOCAL ACCEPTANCE PROXY ONLY -- NOT an official Hyra-equivalent result.

Train pool 1749-1931; split into fit (first 60%) + validation (last 40%) for
MODEL SELECTION; report on test 1932-2026 (1-step and multi-step h=12) plus a
persistence baseline. The unattended loop evolves an AR(p)+harmonic forecaster
via a warm-started, evolution-style hyperparameter search (GuidedProposer);
each candidate runs in the Sandbox and is scored on the VALIDATION set so the
test set is never used for selection (prevents reward hacking). The runner fits
theta on the training set (pure-Python least squares).
"""
import os
import re
import json
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data", "sn_m_tot.csv")
WORK_ROOT = os.path.join(_HERE, "solutions")
EB_DIR = os.path.join(_HERE, "eb_hy3")
SANDBOX_TIMEOUT = 120
TRAIN_END = 1932

SOLUTION_TMPL = '''import math

P = {P}
N_HARM = {n_harm}
PERIOD = 132

def features(window, t_idx):
    feat = [float(x) for x in window[-P:]]
    for k in range(1, N_HARM + 1):
        ang = 2 * math.pi * k * t_idx / PERIOD
        feat.append(math.sin(ang))
        feat.append(math.cos(ang))
    feat.append(1.0)
    return feat

def predict(window, theta, t_idx):
    f = features(window, t_idx)
    return sum(t * x for t, x in zip(theta, f))
'''

RUNNER = '''import json
import math

DATA = "{data}"
TRAIN_END = {train_end}

def load():
    series = []
    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(";")
            if len(p) < 4:
                continue
            try:
                y = int(p[0]); m = int(p[1]); v = float(p[3])
            except Exception:
                continue
            if v < 0:
                continue
            series.append((y, m, v))
    return series

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
                f = A[r][col]
                for j in range(col, n):
                    A[r][j] -= f * A[col][j]
                b[r] -= f * b[col]
    return b

def r2(pred, Y):
    if not Y:
        return 0.0
    ym = sum(Y) / len(Y)
    ssr = sum((yi - pi) ** 2 for yi, pi in zip(Y, pred))
    sst = sum((yi - ym) ** 2 for yi in Y)
    return 1 - ssr / sst if sst > 1e-12 else 0.0

series = load()
pool = [v for (y, m, v) in series if y < TRAIN_END]
n = len(pool); k = int(n * 0.6)
fit = pool[:k]; val = pool[k:]
test = [v for (y, m, v) in series if y >= TRAIN_END]

import solution
P = solution.P

# ---- fit on the FIT portion only ----
X = []; Y = []
for t in range(P, len(fit)):
    X.append(solution.features(fit[t - P:t], t)); Y.append(fit[t])
theta = solve_lstsq(X, Y)

# ---- VALIDATION (1-step): used for MODEL SELECTION only ----
Xv = []; Yv = []
for t in range(P, len(val)):
    Xv.append(solution.features(val[t - P:t], t)); Yv.append(val[t])
pv = [sum(a * b for a, b in zip(theta, x)) for x in Xv]
val_r2 = r2(pv, Yv)
# persistence baseline on validation (predict the previous value)
pers_v = r2(val[P + 1:], val[P:-1]) if len(val) > P + 1 else 0.0

# ---- final fit on ALL pre-test history, REPORT on test (never selected on) ----
Xa = []; Ya = []
for t in range(P, len(pool)):
    Xa.append(solution.features(pool[t - P:t], t)); Ya.append(pool[t])
theta_all = solve_lstsq(Xa, Ya)
full = pool + test
Xte = []; Yte = []
for t in range(len(pool), len(full)):
    Xte.append(solution.features(full[t - P:t], t)); Yte.append(full[t])
pte = [sum(a * b for a, b in zip(theta_all, x)) for x in Xte]
test_r2 = r2(pte, Yte)

# ---- rolling multi-step forecast (official-style horizons) ----
# For each origin we recursively forecast H steps ahead using ONLY information
# available at that origin, then roll the origin forward across the whole test
# set. R^2 is aggregated per horizon. Two strong classical baselines are
# reported alongside so the loop's skill is honestly contextualised:
#   - persistence      : predict the last known value (naive)
#   - seasonal_naive   : 11-year (period=132) repeat -- the genuine strong
#                         seasonal baseline for sunspot numbers
HORIZONS = [1, 3, 6, 12]
SN_PERIOD = 132   # ~11-year solar cycle: the genuine strong seasonal period
model_preds = dict()
sn_preds = dict()
per_preds = dict()
actuals = dict()
for h in HORIZONS:
    model_preds[h] = []
    sn_preds[h] = []
    per_preds[h] = []
    actuals[h] = []

for o in range(len(test)):
    known = pool + test[:o]
    nk = len(known)
    cur = list(known)
    rec = []
    for hstep in range(1, HORIZONS[-1] + 1):
        t_idx = len(cur)
        x = solution.features(cur[-P:], t_idx)
        yhat = sum(a * b for a, b in zip(theta_all, x))
        rec.append(yhat)
        cur.append(yhat)
    for h in HORIZONS:
        if o + h - 1 < len(test):
            model_preds[h].append(rec[h - 1])
            si = nk - 1 - SN_PERIOD + h
            sn_preds[h].append(known[si] if si >= 0 else known[nk - 1])
            per_preds[h].append(known[nk - 1])         # last known value
            actuals[h].append(test[o + h - 1])

ms = dict()
for h in HORIZONS:
    ms[str(h)] = dict()
    ms[str(h)]["model"] = round(r2(model_preds[h], actuals[h]), 4)
    ms[str(h)]["seasonal_naive"] = round(r2(sn_preds[h], actuals[h]), 4)
    ms[str(h)]["persistence"] = round(r2(per_preds[h], actuals[h]), 4)

test_r2_h12 = ms["12"]["model"]

print(json.dumps({{
    "score": val_r2,
    "val_r2": val_r2,
    "test_r2": test_r2,
    "test_r2_h12": test_r2_h12,
    "persistence_val_r2": pers_v,
    "multi_step": ms,
    "n_test": len(Yte),
}}))
'''


class SunspotTask:
    def __init__(self):
        self.series = self._load()
        self.train = [v for (y, m, v) in self.series if y < TRAIN_END]
        self.test = [v for (y, m, v) in self.series if y >= TRAIN_END]

    def _load(self):
        out = []
        with open(DATA) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = line.split(";")
                if len(p) < 4:
                    continue
                try:
                    y = int(p[0]); m = int(p[1]); v = float(p[3])
                except Exception:
                    continue
                if v < 0:
                    continue
                out.append((y, m, v))
        return out

    def proposal_prompt(self):
        return ("Design a forecasting solution for monthly sunspot numbers. "
                "Return Python code with global P (AR order) and N_HARM (harmonic pairs).")

    def workdir(self, i):
        d = os.path.join(WORK_ROOT, f"iter_{i}")
        os.makedirs(d, exist_ok=True)
        return d

    def make_solution(self, proposal, insp, i):
        base = self._best_individual(insp)
        ind = self._mutate(base)
        P, n_harm = ind
        code = SOLUTION_TMPL.format(P=P, n_harm=n_harm)
        summary = f"AR order={P}, harmonics={n_harm}"
        return code, summary

    def _best_individual(self, insp):
        if insp:
            m = re.search(r"order=(\d+).*harmonics=(\d+)", insp[0].get("summary", ""))
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return (12, 2)

    def _mutate(self, base):
        P, n = base
        P = max(1, min(36, P + random.choice([-2, -1, 0, 1, 2])))
        n = max(0, min(6, n + random.choice([-1, 0, 0, 1])))
        return (P, n)

    # -- genome interface for the unattended auto-loop ---------------------
    # The genome parameterises exactly the hypotheses HY3 discovered by hand:
    #   P           : AR order (raw lags)
    #   periods     : which solar-cycle harmonics to include
    #   phase_gain  : s*last / c*last interaction (amplitude modulation, h3)
    #   sqrt        : sqrt(lag) variance-stabilising features (h4)
    def genome_space(self):
        return {
            "P": [10, 12, 14, 16, 18, 20, 24, 28],
            "periods": [
                [132.0],
                [132.0, 66.0],
                [132.0, 66.0, 264.0],
                [132.0, 66.0, 44.0, 264.0],
                [132.0, 66.0, 264.0, 1000.0],
            ],
            "phase_gain": [False, True],
            "sqrt": [False, True],
        }

    def seed_genome(self):
        # Deliberately weak start so the loop must *discover* the good model.
        return {"P": 12, "periods": [132.0], "phase_gain": False, "sqrt": False}

    def render(self, genome):
        P = genome["P"]
        periods = list(genome["periods"])
        pg = bool(genome["phase_gain"])
        sq = bool(genome["sqrt"])
        code = (
            "import math\n"
            f"P = {P}\n"
            f"PERIODS = {periods}\n"
            f"PHASE_GAIN = {pg}\n"
            f"SQRT = {sq}\n\n"
            "def features(window, t_idx):\n"
            "    w = [float(x) for x in window[-P:]]\n"
            "    feat = list(w)\n"
            "    if SQRT:\n"
            "        for x in w:\n"
            "            feat.append(math.sqrt(max(x, 0.0)))\n"
            "    last = w[-1]\n"
            "    for per in PERIODS:\n"
            "        ang = 2 * math.pi * t_idx / per\n"
            "        s = math.sin(ang); c = math.cos(ang)\n"
            "        feat.append(s); feat.append(c)\n"
            "        if PHASE_GAIN:\n"
            "            feat.append(s * last); feat.append(c * last)\n"
            "    feat.append(1.0)\n"
            "    return feat\n\n"
            "def predict(window, theta, t_idx):\n"
            "    f = features(window, t_idx)\n"
            "    return sum(t * x for t, x in zip(theta, f))\n"
        )
        summary = f"P={P} periods={periods} phase_gain={pg} sqrt={sq}"
        return code, summary

    def runner_code(self):
        # inject as a forward-slash path so Windows backslashes are never
        # interpreted as escape sequences inside the generated runner string
        return RUNNER.format(data=DATA.replace("\\", "/"), train_end=TRAIN_END)

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
                # `score` is the VALIDATION R^2 and is the ONLY metric used for
                # model selection -- the test set below is never used to choose.
                "score": obj.get("score"),
                "val_r2": obj.get("val_r2"),
                "test_r2": obj.get("test_r2"),
                "test_r2_h12": obj.get("test_r2_h12"),
                "persistence_val_r2": obj.get("persistence_val_r2"),
                "multi_step": obj.get("multi_step"),
            }
        except Exception:
            return {"ok": False, "score": None, "stderr": run["stdout"][-200:]}
