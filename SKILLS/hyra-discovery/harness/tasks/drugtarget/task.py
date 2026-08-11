"""Drug-Target binding discovery task (ligand-based SAR, offline).

Discovery goal: given molecule features + activity labels, FIND which feature
subset best predicts activity = rediscover the structure-activity relationship.

Data contract (CSV at tasks/drugtarget/data/dataset.csv):
  - header row; columns: optional `id`, then numeric FEATURE columns, then
    `activity` (float, higher = more active) OR `label` (0/1 binder).
  - optional `split` column with values train/val/test to fix splits;
    otherwise a fixed-seed stratified-ish random split is used.
  - optional tasks/drugtarget/data/reference.csv : one row of features for a
    reference molecule (e.g. olaparib) to "beat".

Honesty note: the harness's auto proposer does FEATURE-SUBSET search over a
linear model. That is a legitimate SAR-rediscovery loop, NOT a claim of
de-novo generative drug design. Real docking needs GPU/force-fields.

Leakage policy (fixes D9): model SELECTION uses the VALIDATION set only;
the TEST set is reported once at the end and never used for selection.
"""
import os
import json
import csv
import random
from hyra.realdata import discover_csv, realdata_present

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = discover_csv(_HERE, os.path.join("data", "dataset.csv"))
REF = os.path.join(_HERE, "data", "reference.csv")
SEED = 0

SOLUTION_TMPL = '''\
# AUTO-GENERATED candidate solution (drug-target SAR)
SELECTED = {sel}
INTER = {inter}
COEF = None

def _feat(x_all):
    x = [x_all[i] for i in SELECTED]
    if INTER:
        n = len(x)
        for i in range(n):
            for j in range(i + 1, n):
                x.append(x[i] * x[j])
    x.append(1.0)
    return x

def predict(x_all):
    return sum(c * v for c, v in zip(COEF, _feat(x_all)))
'''

RUNNER = '''\
import os, sys, json, csv, random
DATA = {data}
REF = {ref}

def load():
    rows = []
    with open(DATA) as f:
        r = csv.DictReader(f)
        feats = [c for c in r.fieldnames if c not in ("id", "activity", "label", "split")]
        for row in r:
            x = [float(row[c]) for c in feats]
            if "label" in r.fieldnames:
                y = float(row["label"]); kind = "cls"
            else:
                y = float(row["activity"]); kind = "reg"
            sp = row.get("split", "")
            rows.append((x, y, kind, sp))
    return feats, rows

def solve(A, b, ridge=1e-6):
    n = len(A); m = len(A[0])
    AtA = [[0.0]*m for _ in range(m)]; Atb = [0.0]*m
    for i in range(n):
        for p in range(m):
            Atb[p] += A[i][p]*b[i]
            for q in range(m):
                AtA[p][q] += A[i][p]*A[i][q]
    for p in range(m):
        AtA[p][p] += ridge
    for col in range(m):
        piv = max(range(col, m), key=lambda rr: abs(AtA[rr][col]))
        AtA[col], AtA[piv] = AtA[piv], AtA[col]
        Atb[col], Atb[piv] = Atb[piv], Atb[col]
        pv = AtA[col][col] or 1e-12
        for rr in range(col+1, m):
            fct = AtA[rr][col]/pv
            for c in range(col, m):
                AtA[rr][c] -= fct*AtA[col][c]
            Atb[rr] -= fct*Atb[col]
    x = [0.0]*m
    for col in range(m-1, -1, -1):
        s = Atb[col]
        for rr in range(col+1, m):
            s -= AtA[col][rr]*x[rr]
        x[col] = s/(AtA[col][col] or 1e-12)
    return x

def expand(x_all, sel, inter):
    x = [x_all[i] for i in sel]
    if inter:
        n = len(x)
        for i in range(n):
            for j in range(i+1, n):
                x.append(x[i]*x[j])
    x.append(1.0)
    return x

def r2(yt, yp):
    m = sum(yt)/len(yt)
    sr = sum((a-b)**2 for a,b in zip(yt,yp))
    st = sum((a-m)**2 for a in yt)
    return 1 - sr/st if st > 0 else 0.0

def auc(scores, labels):
    pos = [s for s,l in zip(scores,labels) if l==1]
    neg = [s for s,l in zip(scores,labels) if l==0]
    if not pos or not neg: return 0.5
    c = 0
    for p in pos:
        for nn in neg:
            c += 1 if p>nn else (0.5 if p==nn else 0)
    return c/(len(pos)*len(neg))

feats, rows = load()
F = len(feats)
import importlib.util
spec = importlib.util.spec_from_file_location("solution", os.path.join(os.getcwd(), "solution.py"))
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
sel = solution.SELECTED; inter = solution.INTER

random.seed(0)
test = [r for r in rows if r[3] == "test"]
pool = [r for r in rows if r[3] != "test"]
if not test:
    random.shuffle(rows)
    k = len(rows)//5
    test = rows[:k]; pool = rows[k:]

kind = rows[0][2]

def fit_eval(tr_set, va_set):
    # fit on tr_set, evaluate on va_set (val) and the held-out test set
    Xtr=[expand(r[0], sel, inter) for r in tr_set]; ytr=[r[1] for r in tr_set]
    coef=solve(Xtr,ytr); solution.COEF=coef
    def preds(set_): return [solution.predict(r[0]) for r in set_]
    if kind=="reg":
        vm=r2([r[1] for r in va_set], preds(va_set))
        tm=r2([r[1] for r in test], preds(test))
        return vm, tm
    else:
        vm=auc(preds(va_set),[r[1] for r in va_set])
        tm=auc(preds(test),[r[1] for r in test])
        return vm, tm

# K-fold CV on the training POOL -> robust, parsimony-aware selection score
# (a single fixed split made the sparse model's win look like a lucky draw).
K = 5
idxs=list(range(len(pool))); random.shuffle(idxs)
folds=[idxs[p::K] for p in range(K)]
cv_scores=[]
for fidx in range(K):
    te_idx=set(folds[fidx])
    tr_set=[pool[i] for i in range(len(pool)) if i not in te_idx]
    va_set=[pool[i] for i in te_idx]
    vm,_=fit_eval(tr_set, va_set)
    p=len(sel)+inter+1; nv=len(va_set)
    adj=(1-(1-vm)*(nv-1)/max(1,nv-p-1)) if nv>p+1 else vm
    cv_scores.append(adj)
cv_score=sum(cv_scores)/len(cv_scores)

# refit on all of the pool, evaluate once on the held-out test set
val_metric, test_metric = fit_eval(pool, test)

# honest baseline: the ALL-features least-squares model on the SAME test set.
# the sparse discovery must BEAT this to be meaningful (not just "beats nothing").
orig_sel, orig_inter = sel, inter
sel=list(range(F)); inter=0
_, baseline_test = fit_eval(pool, test)
sel, inter = orig_sel, orig_inter
beats_baseline = test_metric > baseline_test

score = cv_score   # selection uses cross-validated, parsimony-adjusted R^2

beats = None
if os.path.exists(REF):
    with open(REF) as f:
        rr = csv.reader(f); hdr = next(rr); vals = next(rr)
    ref_x = [float(v) for v in vals]
    ref_score = solution.predict(ref_x)
    test_scores = [solution.predict(r[0]) for r in test]
    beats = sum(1 for s in test_scores if s > ref_score)

print(json.dumps({{
    "score": score,
    "val_metric": val_metric,
    "test_metric": test_metric,
    "cv_adj": cv_score,
    "baseline_test_r2": baseline_test,
    "beats_baseline": beats_baseline,
    "kind": kind,
    "selected": sel,
    "inter": inter,
    "beats_ref": beats,
    "n_train": len(pool), "n_val": len(test), "n_test": len(test)
}}))
'''


class DrugTargetTask:
    def __init__(self):
        self.F = self._n_features()
        self._name = "DrugTargetTask"

    def _n_features(self):
        with open(DATA) as f:
            r = csv.DictReader(f)
            return len([c for c in r.fieldnames
                        if c not in ("id", "activity", "label", "split")])

    # ---- Task interface (harness + hy3_step) ----
    def workdir(self, tag):
        d = os.path.join(_HERE, "solutions", f"iter_{tag}")
        os.makedirs(d, exist_ok=True)
        return d

    def runner_code(self):
        return RUNNER.format(data=repr(DATA), ref=repr(REF))

    def make_solution(self, inspirations, task_prompt=None):
        # harness path: use best inspiration's genome if available else seed
        genome = self.seed_genome()
        for insp in (inspirations or []):
            g = (insp or {}).get("genome")
            if g:
                genome = g; break
        return self.render(genome)

    def parse_run(self, run):
        if not run.get("ok"):
            return {"ok": False, "score": None,
                    "stderr": run.get("stderr", "")[:200]}
        try:
            obj = json.loads(run["stdout"].splitlines()[-1])
            # Map the runner's val_metric/test_metric onto the harness's
            # expected val_r2/test_r2 so the anti-reward-hacking reporting and
            # warm-start bookkeeping line up across tasks.
            return {"ok": True,
                    "score": obj.get("score"),        # val-based selection score
                    "val_r2": obj.get("val_metric"),
                    "test_r2": obj.get("test_metric"),
                    "cv_adj": obj.get("cv_adj"),
                    "baseline_test_r2": obj.get("baseline_test_r2"),
                    "beats_baseline": obj.get("beats_baseline"),
                    "test_metric": obj.get("test_metric"),
                    "kind": obj.get("kind"),
                    "selected": obj.get("selected"),
                    "beats_ref": obj.get("beats_ref")}
        except Exception as e:
            return {"ok": False, "score": None, "stderr": repr(e)[:200]}

    # ---- genome interface (dict contract for GuidedProposer / auto_loop) ----
    # One boolean key per feature (f0..f{F-1}) + an "inter" interaction flag.
    # This is the search space the unattended loop explores to REDISCOVER the
    # structure-activity relationship (which features truly drive binding).
    def genome_space(self):
        space = {f"f{i}": [0, 1] for i in range(self.F)}
        space["inter"] = [0, 1]
        return space

    def seed_genome(self):
        g = {f"f{i}": 1 for i in range(self.F)}  # start: all features on
        g["inter"] = 0
        return g

    def render(self, genome):
        if isinstance(genome, dict):
            sel = [i for i in range(self.F) if int(genome.get(f"f{i}", 0))]
            inter = int(bool(genome.get("inter", 0)))
        else:  # legacy list form: [bit0..bitF-1, inter]
            sel = [i for i, b in enumerate(genome[:-1]) if b]
            inter = int(bool(genome[-1]))
        code = SOLUTION_TMPL.format(sel=sel, inter=inter)
        summary = f"sel={sel} inter={inter}"
        return code, summary

    # convenience for manual HY3 proposals
    def render_selected(self, sel, inter=0):
        return SOLUTION_TMPL.format(sel=sel, inter=int(bool(inter)))

    # ---- cross-task transfer (Hyra-style shared priors) ----
    def family(self):
        return "medical"

    def seed_from_priors(self, priors):
        """Cold-start sparser using sibling medical tasks' winning sparsity."""
        target = max(1, round(priors.get("median_sparsity", 0.3) * self.F))
        g = {f"f{i}": (1 if i < target else 0) for i in range(self.F)}
        g["inter"] = 0
        return g

    def lesson(self, genome, parsed):
        if genome is None:
            return None
        on = sum(1 for i in range(self.F) if int(genome.get(f"f{i}", 0)))
        inter = int(bool(genome.get("inter", 0)))
        total = self.F + 1
        sub = on + (1 if inter else 0)
        won = parsed.get("score") is not None
        sparse = (inter == 0 and on < self.F * 0.6)
        return {
            "sparsity": sub / total,
            "score": parsed.get("score"),
            "val_r2": parsed.get("val_r2"),
            "test_r2": parsed.get("test_r2"),
            "won_by_parsimony": bool(sparse) if won else False,
            "archetype": "sparse_subset" if sparse else ("interaction" if inter else "dense"),
        }

    def realdata_available(self):
        """True if a real dataset is wired in via tasks/drugtarget/data_real/."""
        return realdata_present(_HERE)


# exports for hy3_step / auto_loop
Task = DrugTargetTask
EB_DIR = os.path.join(_HERE, "eb_hy3")
