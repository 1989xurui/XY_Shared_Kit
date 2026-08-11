"""Synthesis (generative) discovery task -- REAL fragment assembly via RDKit.

Discovery goal: given a library of REAL chemical fragments, FIND a fragment
combination that assembles into a NOVEL, VALID, drug-like molecule (high QED).

This is a genuine generative-design loop:
  - fragments are REAL SMILES, each carrying one `[*]` attachment point;
  - molecules are built by chaining fragments through their attachment points
    with RDKit (real valence / sanitization checks);
  - VALIDITY is a real chemistry check (RDKit sanitize), not an index heuristic;
  - the objective is QED (real drug-likeness from rdkit.Chem.QED).

Honesty note: QED / novelty are REAL, but the `activity` proxy is still a
surrogate (there is no wet-lab readout). This is fragment-based generative
design with real chemistry -- NOT a Hyra-style de-novo transformer, and NOT a
claim of synthesized/preclinically-validated compounds. Requires rdkit in the
running interpreter (use the torch/rdkit venv).

Genome: one boolean per fragment (f0..f{NF-1}) selecting which fragments to
combine. The runner internally searches fragment ORDERINGS to maximize QED.
"""
import os
import json
import csv
import random

try:
    import rdkit  # noqa: F401
    HAS_RDKit = True
except Exception:
    HAS_RDKit = False

_HERE = os.path.dirname(os.path.abspath(__file__))
FRAGS = os.path.join(_HERE, "data", "fragments.csv")
LIB = os.path.join(_HERE, "data", "library.csv")
SEED = 0

# ---- REAL fragment library (SMILES with one [*] attachment point) ----
REAL_FRAGMENTS = [
    "c1ccccc1[*]",       # 0 phenyl
    "c1ccncc1[*]",       # 1 pyridine
    "c1ccsc1[*]",        # 2 thiophene
    "C1CCCCC1[*]",       # 3 cyclohexyl
    "C(=O)[*]",          # 4 carbonyl
    "C(=O)O[*]",         # 5 carboxyl / ester
    "C(=O)N([*])",       # 6 amide (N attach)
    "O[*]",              # 7 ether oxygen
    "N(C)[*]",           # 8 N-methyl amine
    "C([*])C",           # 9 methylene
    "S([*])",            # 10 thioether
    "c1ccc2ccccc2c1[*]", # 11 naphthalene
]
NF = len(REAL_FRAGMENTS)
MAX_FRAG = 5

SOLUTION_TMPL = '''\
# AUTO-GENERATED candidate solution (real fragment assembly)
SELECTED = {sel}
'''


RUNNER = '''\
import os, sys, json, csv, random
import importlib.util
spec = importlib.util.spec_from_file_location("solution", os.path.join(os.getcwd(), "solution.py"))
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
LIB = {lib}
REAL_FRAGMENTS = {frags}
MAX_FRAG = {maxfrag}

from rdkit import Chem
from rdkit.Chem import QED

DUMMY = Chem.MolFromSmarts("[#0]")  # atomic number 0 == attachment dummy

def chain_join(frag_smiles):
    if not frag_smiles:
        return None
    mol = Chem.MolFromSmiles(frag_smiles[0])
    if mol is None:
        return None
    for fs in frag_smiles[1:]:
        fm = Chem.MolFromSmiles(fs)
        if fm is None:
            return None
        mol = Chem.ReplaceSubstructs(mol, DUMMY, fm, replaceAll=False)[0]
    # cap the final attachment point with H
    mol = Chem.ReplaceSubstructs(mol, DUMMY, Chem.MolFromSmiles("[H]"))[0]
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol

def load_lib():
    s = set()
    try:
        with open(LIB) as f:
            for row in csv.DictReader(f):
                s.add(row["frag_set"])
    except Exception:
        pass
    return s

SELECTED = solution.SELECTED
sel = SELECTED[:MAX_FRAG]
frags = [REAL_FRAGMENTS[i] for i in sel]
lib_set = load_lib()
novel = (";".join(map(str, sel)) not in lib_set)

# internally search fragment orderings to maximize real QED
best = None
best_qed = -1.0
orderings = [sel]
orderings.append(list(reversed(sel)))
random.seed(0)
for _ in range(10):
    o = list(sel); random.shuffle(o); orderings.append(o)
for order in orderings:
    mol = chain_join([REAL_FRAGMENTS[i] for i in order])
    if mol is None:
        continue
    try:
        q = QED.qed(mol)
    except Exception:
        continue
    if q > best_qed:
        best_qed = q; best = mol

valid = best is not None
smiles = Chem.MolToSmiles(best) if valid else ""
pred_act = float(best_qed) if valid else 0.0
# score: maximize real drug-likeness; penalize invalid / non-novel
score = (pred_act if (valid and novel) else (pred_act - 5.0))
print(json.dumps({{
    "score": score,
    "val_r2": pred_act,
    "test_r2": pred_act,
    "pred_activity": pred_act,
    "valid": valid,
    "novel": novel,
    "selected": sel,
    "smiles": smiles,
    "n_sel": len(sel)
}}))
'''

if not HAS_RDKit:
    RUNNER = '''\
import os, json
print(json.dumps({"ok": False, "score": None, "stderr": "rdkit not installed"}))
'''


class SynthesisTask:
    def __init__(self):
        self.NF = NF
        self._name = "SynthesisTask"

    def workdir(self, tag):
        d = os.path.join(_HERE, "solutions", f"iter_{tag}")
        os.makedirs(d, exist_ok=True)
        return d

    def runner_code(self):
        return RUNNER.format(lib=repr(LIB), frags=repr(REAL_FRAGMENTS),
                             maxfrag=MAX_FRAG)

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
                    "pred_activity": obj.get("pred_activity"),
                    "valid": obj.get("valid"),
                    "novel": obj.get("novel"),
                    "selected": obj.get("selected"),
                    "smiles": obj.get("smiles"),
                    "n_sel": obj.get("n_sel")}
        except Exception as e:
            return {"ok": False, "score": None, "stderr": repr(e)[:200]}

    def genome_space(self):
        return {f"f{i}": [0, 1] for i in range(self.NF)}

    def seed_genome(self):
        # start: a small valid-ish build (phenyl + carbonyl + amide)
        g = {f"f{i}": 0 for i in range(self.NF)}
        g["f0"] = 1
        g["f4"] = 1
        g["f6"] = 1
        return g

    def render(self, genome):
        if isinstance(genome, dict):
            sel = sorted([i for i in range(self.NF) if int(genome.get(f"f{i}", 0))])
        else:
            sel = sorted([i for i, b in enumerate(genome) if b])
        sel = sel[:MAX_FRAG]
        code = SOLUTION_TMPL.format(sel=sel)
        summary = f"frags={sel}"
        return code, summary

    def render_selected(self, sel):
        return SOLUTION_TMPL.format(sel=sel)

    # ---- cross-task transfer (Hyra-style shared priors) ----
    def family(self):
        return "medical"

    def seed_from_priors(self, priors):
        target = max(2, round(priors.get("median_sparsity", 0.4) * self.NF))
        g = {f"f{i}": 0 for i in range(self.NF)}
        g["f0"] = 1
        on = 1
        for i in range(self.NF):
            if i == 0:
                continue
            if on < target:
                g[f"f{i}"] = 1
                on += 1
        return g

    def lesson(self, genome, parsed):
        if genome is None:
            return None
        on = sum(1 for i in range(self.NF) if int(genome.get(f"f{i}", 0)))
        score = parsed.get("score")
        won = score is not None
        return {
            "sparsity": on / self.NF if self.NF else 0,
            "score": score,
            "val_r2": parsed.get("val_r2"),
            "test_r2": parsed.get("test_r2"),
            "won_by_parsimony": bool(parsed.get("novel") and parsed.get("valid")),
            "archetype": "fragment_assembly",
        }

    def realdata_available(self):
        """Synthesis uses a built-in real RDKit fragment library by default."""
        return HAS_RDKit


Task = SynthesisTask
EB_DIR = os.path.join(_HERE, "eb_hy3")
