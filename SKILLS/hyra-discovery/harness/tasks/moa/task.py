"""Mechanism-of-action (MOA) discovery task (multi-class classification).

Discovery goal: given compound features + known mechanism labels, FIND the
minimal feature subset that classifies mechanism = rediscover the mechanism
signature.

Data contract (tasks/moa/data/dataset.csv):
  - header; columns f0..fF-1, `label` (int mechanism id 0..K-1), `split`
    (train/val/test). Optional: omit split for a fixed-seed random split.

Leakage policy (fixes D9): model SELECTION uses VALIDATION accuracy only,
penalized by feature count so the parsimonious mechanism signature wins;
TEST accuracy is reported once and never used for selection.
"""
import os
import json
import csv
import random
from hyra.realdata import discover_csv, realdata_present

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = discover_csv(_HERE, os.path.join("data", "dataset.csv"))
SEED = 0


SOLUTION_TMPL = '''\
# AUTO-GENERATED candidate solution (MOA feature subset)
SELECTED = {sel}
'''


def _solve(A, b, ridge=1e-6):
    n = len(A); m = len(A[0])
    AtA = [[0.0] * m for _ in range(m)]; Atb = [0.0] * m
    for i in range(n):
        for p in range(m):
            Atb[p] += A[i][p] * b[i]
            for q in range(m):
                AtA[p][q] += A[i][p] * A[i][q]
    for p in range(m):
        AtA[p][p] += ridge
    for col in range(m):
        piv = max(range(col, m), key=lambda rr: abs(AtA[rr][col]))
        AtA[col], AtA[piv] = AtA[piv], AtA[col]
        Atb[col], Atb[piv] = Atb[piv], Atb[col]
        pv = AtA[col][col] or 1e-12
        for rr in range(col + 1, m):
            fct = AtA[rr][col] / pv
            for c in range(col, m):
                AtA[rr][c] -= fct * AtA[col][c]
            Atb[rr] -= fct * Atb[col]
    x = [0.0] * m
    for col in range(m - 1, -1, -1):
        s = Atb[col]
        for rr in range(col + 1, m):
            s -= AtA[col][rr] * x[rr]
        x[col] = s / (AtA[col][col] or 1e-12)
    return x


RUNNER = '''\
import os, sys, json, csv, random
import importlib.util
spec = importlib.util.spec_from_file_location("solution", os.path.join(os.getcwd(), "solution.py"))
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
DATA = {data}

def _solve(A, b, ridge=1e-6):
    n = len(A); m = len(A[0])
    AtA = [[0.0] * m for _ in range(m)]; Atb = [0.0] * m
    for i in range(n):
        for p in range(m):
            Atb[p] += A[i][p] * b[i]
            for q in range(m):
                AtA[p][q] += A[i][p] * A[i][q]
    for p in range(m):
        AtA[p][p] += ridge
    for col in range(m):
        piv = max(range(col, m), key=lambda rr: abs(AtA[rr][col]))
        AtA[col], AtA[piv] = AtA[piv], AtA[col]
        Atb[col], Atb[piv] = Atb[piv], Atb[col]
        pv = AtA[col][col] or 1e-12
        for rr in range(col + 1, m):
            fct = AtA[rr][col] / pv
            for c in range(col, m):
                AtA[rr][c] -= fct * AtA[col][c]
            Atb[rr] -= fct * Atb[col]
    x = [0.0] * m
    for col in range(m - 1, -1, -1):
        s = Atb[col]
        for rr in range(col + 1, m):
            s -= AtA[col][rr] * x[rr]
        x[col] = s / (AtA[col][col] or 1e-12)
    return x


def load():
    feats=[]; rows=[]
    with open(DATA) as f:
        r=csv.DictReader(f)
        feats=[c for c in r.fieldnames if c not in ("label","split")]
        for row in r:
            x=[float(row[c]) for c in feats]
            rows.append((x, int(row["label"]), row.get("split","")))
    return feats, rows

def expand(x, sel):
    return [x[i] for i in sel]

feats, rows = load()
F = len(feats)
random.seed(0)
tr=[r for r in rows if r[2] in ("train","")]
va=[r for r in rows if r[2]=="val"]
te=[r for r in rows if r[2]=="test"]
if not va and not te:
    random.shuffle(tr); k=len(tr)//5; va=tr[:k]; te=tr[k:2*k]; tr=tr[2*k:]
classes=sorted({{r[1] for r in rows}})
SELECTED = solution.SELECTED
sel = SELECTED

# one-vs-rest least squares
Atr=[]; Btr=[]
for r in tr:
    e=expand(r[0], sel)
    Atr.append(e)
    for ci,c in enumerate(classes):
        Btr.append(1.0 if r[1]==c else 0.0)
# reshape Btr into per-class coefficient vectors
coefs=[]
for ci,c in enumerate(classes):
    b=[1.0 if r[1]==c else 0.0 for r in tr]
    coefs.append(_solve(Atr, b))

def pred(x):
    e=expand(x, sel)
    scores=[sum(cx*ev for cx,ev in zip(coef, e)) for coef in coefs]
    return max(range(len(scores)), key=lambda i: scores[i])

def acc(set_):
    if not set_: return 0.0
    ok=sum(1 for r in set_ if pred(r[0])==r[1])
    return ok/len(set_)

val_acc = acc(va)
test_acc = acc(te)
# parsimony: prefer fewer features when accuracy is comparable
score = val_acc - 0.02 * len(sel)
print(json.dumps({{
    "score": score,
    "val_r2": val_acc,
    "test_r2": test_acc,
    "val_acc": val_acc,
    "test_acc": test_acc,
    "selected": sel,
    "n_sel": len(sel)
}}))
'''


class MoATask:
    def __init__(self):
        self.F = self._n_features()
        self._name = "MoATask"

    def _n_features(self):
        with open(DATA) as f:
            r = csv.DictReader(f)
            return len([c for c in r.fieldnames if c not in ("label", "split")])

    def workdir(self, tag):
        d = os.path.join(_HERE, "solutions", f"iter_{tag}")
        os.makedirs(d, exist_ok=True)
        return d

    def runner_code(self):
        return RUNNER.format(data=repr(DATA))

    def make_solution(self, inspirations, task_prompt=None):
        genome = self.seed_genome()
        for insp in (inspirations or []):
            g = (insp or {}).get("genome")
            if g:
                genome = g
                break
        return self.render(genome)

    def parse_run(self, run):
        if not run.get("ok"):
            return {"ok": False, "score": None, "stderr": run.get("stderr", "")[:200]}
        try:
            obj = json.loads(run["stdout"].splitlines()[-1])
            return {"ok": True,
                    "score": obj.get("score"),
                    "val_r2": obj.get("val_r2"),
                    "test_r2": obj.get("test_r2"),
                    "val_acc": obj.get("val_acc"),
                    "test_acc": obj.get("test_acc"),
                    "selected": obj.get("selected"),
                    "n_sel": obj.get("n_sel")}
        except Exception as e:
            return {"ok": False, "score": None, "stderr": repr(e)[:200]}

    def genome_space(self):
        return {f"f{i}": [0, 1] for i in range(self.F)}

    def seed_genome(self):
        return {f"f{i}": 1 for i in range(self.F)}  # start: all features on

    def render(self, genome):
        if isinstance(genome, dict):
            sel = [i for i in range(self.F) if int(genome.get(f"f{i}", 0))]
        else:
            sel = [i for i, b in enumerate(genome) if b]
        code = SOLUTION_TMPL.format(sel=sel)
        summary = f"feats={sel}"
        return code, summary

    def render_selected(self, sel):
        return SOLUTION_TMPL.format(sel=sel)

    # ---- cross-task transfer (Hyra-style shared priors) ----
    def family(self):
        return "medical"

    def seed_from_priors(self, priors):
        target = max(1, round(priors.get("median_sparsity", 0.3) * self.F))
        g = {f"f{i}": (1 if i < target else 0) for i in range(self.F)}
        return g

    def lesson(self, genome, parsed):
        if genome is None:
            return None
        on = sum(1 for i in range(self.F) if int(genome.get(f"f{i}", 0)))
        won = parsed.get("score") is not None
        sparse = on < self.F * 0.6
        return {
            "sparsity": on / self.F if self.F else 0,
            "score": parsed.get("score"),
            "val_r2": parsed.get("val_r2"),
            "test_r2": parsed.get("test_r2"),
            "won_by_parsimony": bool(sparse) if won else False,
            "archetype": "sparse_subset" if sparse else "dense",
        }

    def realdata_available(self):
        """True if a real dataset is wired in via tasks/moa/data_real/."""
        return realdata_present(_HERE)


Task = MoATask
EB_DIR = os.path.join(_HERE, "eb_hy3")
