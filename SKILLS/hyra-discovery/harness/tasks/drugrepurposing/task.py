"""Drug-repurposing discovery task (gene-signature / connectivity-map style).

Discovery goal: given a disease gene-expression signature and a library of
drug perturbation profiles, FIND which gene SUBSET best retrieves the drug
that reverses the disease = rediscover the repurposing mechanism.

Data contract (two CSVs in tasks/drugrepurposing/data/):
  - disease_expr.csv : patient rows; columns g0..gG-1, `label` (1 disease /
    0 control), `split` (train/val/test). The disease signature is learned
    from TRAIN patients only.
  - drug_profiles.csv : drug rows; columns drug_id, g0..gG-1, `split`
    (val/test), `true_reversal` (oracle ground truth, used ONLY for
    selecting the gene subset on VAL drugs and reporting once on TEST).

Leakage policy (fixes D9): model SELECTION uses VAL-drug true_reversal to
pick the best gene subset; TEST-drug true_reversal is reported exactly once
and never used for selection. The true repurposing drug (drug_0) lives in
the TEST set so we can honestly report how high it ranks.
"""
import os
import json
import csv
import random
from hyra.realdata import discover_csv, realdata_present

_HERE = os.path.dirname(os.path.abspath(__file__))
DISEASE = discover_csv(_HERE, os.path.join("data", "disease_expr.csv"))
PROFILES = discover_csv(_HERE, os.path.join("data", "drug_profiles.csv"))
SEED = 0

SOLUTION_TMPL = '''\
# AUTO-GENERATED candidate solution (drug-repurposing gene subset)
SELECTED = {sel}
'''


def _spearman(x, y):
    n = len(x)
    if n < 2:
        return 0.0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    return _pearson(rx, ry)


def _pearson(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


import math  # noqa: E402  (kept after helpers for readability)


RUNNER = '''\
import os, sys, json, csv, random, math
import importlib.util
spec = importlib.util.spec_from_file_location("solution", os.path.join(os.getcwd(), "solution.py"))
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
DISEASE = {disease}
PROFILES = {profiles}

def load_disease():
    feats = []; rows = []
    with open(DISEASE) as f:
        r = csv.DictReader(f)
        feats = [c for c in r.fieldnames if c not in ("label", "split")]
        for row in r:
            x = [float(row[c]) for c in feats]
            rows.append((x, int(row["label"]), row["split"]))
    return feats, rows

def load_drugs():
    feats = []; rows = []
    with open(PROFILES) as f:
        r = csv.DictReader(f)
        feats = [c for c in r.fieldnames if c not in ("drug_id", "split", "true_reversal")]
        for row in r:
            x = [float(row[c]) for c in feats]
            rows.append((row["drug_id"], x, row["split"], float(row["true_reversal"])))
    return feats, rows

def pearson(a, b):
    n = len(a)
    if n < 2: return 0.0
    ma = sum(a)/n; mb = sum(b)/n
    num = sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da = math.sqrt(sum((x-ma)**2 for x in a)); db = math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db) if da>0 and db>0 else 0.0

def spearman(x, y):
    n = len(x)
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
        while i < len(v):
            j=i
            while j+1<len(v) and v[order[j+1]]==v[order[i]]: j+=1
            avg=(i+j)/2.0+1
            for k in range(i,j+1): r[order[k]]=avg
            i=j+1
        return r
    return pearson(rank(x), rank(y))

feats_d, drows = load_disease()
G = len(feats_d)
# disease signature from TRAIN patients only
sig = [0.0]*G
dis = [r[0] for r in drows if r[1]==1 and r[2]=="train"]
con = [r[0] for r in drows if r[1]==0 and r[2]=="train"]
for g in range(G):
    md = sum(x[g] for x in dis)/len(dis) if dis else 0.0
    mc = sum(x[g] for x in con)/len(con) if con else 0.0
    sig[g] = md - mc
neg_sig = [-s for s in sig]

feats_p, drugs = load_drugs()
SELECTED = solution.SELECTED
S = SELECTED

# predicted reversal for each drug using the chosen gene subset
pred = dict()
for did, x, split, _ in drugs:
    rev = pearson([x[g] for g in S], [neg_sig[g] for g in S]) if S else 0.0
    pred[did] = rev

val = [(did, pred[did], tr) for did, x, split, tr in drugs if split=="val"]
test = [(did, pred[did], tr) for did, x, split, tr in drugs if split=="test"]

def col(pairs, i): return [p[i] for p in pairs]
val_score = spearman(col(val,1), col(val,2)) if val else 0.0
test_score = spearman(col(test,1), col(test,2)) if test else 0.0

# rank of the true repurposing drug (drug_0) within the TEST set
test_sorted = sorted(test, key=lambda p: -p[1])
rank_true = next((i+1 for i,p in enumerate(test_sorted) if p[0]=="drug_0"), -1)
n_test = len(test)

print(json.dumps({{
    "score": val_score - 0.01 * len(S) / G,
    "val_r2": val_score,
    "test_r2": test_score,
    "selected": S,
    "rank_true_rep": rank_true,
    "n_test": n_test,
    "n_val": len(val)
}}))
'''


class DrugRepurposingTask:
    def __init__(self):
        self.G = self._n_genes()
        self._name = "DrugRepurposingTask"

    def _n_genes(self):
        with open(DISEASE) as f:
            r = csv.DictReader(f)
            return len([c for c in r.fieldnames if c not in ("label", "split")])

    def workdir(self, tag):
        d = os.path.join(_HERE, "solutions", f"iter_{tag}")
        os.makedirs(d, exist_ok=True)
        return d

    def runner_code(self):
        return RUNNER.format(disease=repr(DISEASE), profiles=repr(PROFILES))

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
                    "selected": obj.get("selected"),
                    "rank_true_rep": obj.get("rank_true_rep"),
                    "n_test": obj.get("n_test")}
        except Exception as e:
            return {"ok": False, "score": None, "stderr": repr(e)[:200]}

    # ---- genome interface (dict contract for GuidedProposer / auto_loop) ----
    def genome_space(self):
        return {f"g{i}": [0, 1] for i in range(self.G)}

    def seed_genome(self):
        return {f"g{i}": 1 for i in range(self.G)}  # start: all genes on

    def render(self, genome):
        if isinstance(genome, dict):
            sel = [i for i in range(self.G) if int(genome.get(f"g{i}", 0))]
        else:
            sel = [i for i, b in enumerate(genome) if b]
        code = SOLUTION_TMPL.format(sel=sel)
        summary = f"genes={sel}"
        return code, summary

    def render_selected(self, sel):
        return SOLUTION_TMPL.format(sel=sel)

    # ---- cross-task transfer (Hyra-style shared priors) ----
    def family(self):
        return "medical"

    def seed_from_priors(self, priors):
        target = max(1, round(priors.get("median_sparsity", 0.3) * self.G))
        g = {f"g{i}": (1 if i < target else 0) for i in range(self.G)}
        return g

    def lesson(self, genome, parsed):
        if genome is None:
            return None
        on = sum(1 for i in range(self.G) if int(genome.get(f"g{i}", 0)))
        won = parsed.get("score") is not None
        sparse = on < self.G * 0.6
        return {
            "sparsity": on / self.G if self.G else 0,
            "score": parsed.get("score"),
            "val_r2": parsed.get("val_r2"),
            "test_r2": parsed.get("test_r2"),
            "won_by_parsimony": bool(sparse) if won else False,
            "archetype": "sparse_subset" if sparse else "dense",
        }

    def realdata_available(self):
        """True if real LINCS-style data is wired via tasks/drugrepurposing/data_real/."""
        return realdata_present(_HERE, "disease_expr.csv", "drug_profiles.csv")


Task = DrugRepurposingTask
EB_DIR = os.path.join(_HERE, "eb_hy3")
