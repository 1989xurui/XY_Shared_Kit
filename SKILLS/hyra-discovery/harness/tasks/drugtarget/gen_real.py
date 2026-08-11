"""Fetch REAL drug-target binding data from ChEMBL and convert it to the
drugtarget schema (id, f0..fF, activity, split) so the harness can run a REAL
benchmark instead of synthetic data.

Source : CHEMBL205 (HIV-1 protease) IC50 activities from the public ChEMBL API.
Features: 12 real RDKit 2D / physicochemical descriptors of each compound SMILES.
Target : pIC50 (real binding potency, -log10[M]).

The harness auto-detects tasks/drugtarget/data_real/dataset.csv and switches to
it (see hyra/realdata.py). Requires rdkit in the running interpreter.

Honesty note: this is REAL bioactivity data. The loop reports a real predictive
R^2 vs an all-features baseline -- it does NOT recover a "true cause" (real data
has no transparent ground-truth subset), and it is not an official acceptance.
"""
import os
import csv
import json
import math
import random
import urllib.request
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Lipinski

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_real", "dataset.csv")
TARGET = "CHEMBL205"
N_FEATURES = 12
MAX_ROWS = 800

DESC_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
              "NumRotatableBonds", "NumAromaticRings", "FractionCSP3", "QED",
              "NumAliphaticRings", "NumHeteroatoms", "NumSaturatedRings"]


def featurize(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return [
            Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
            Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
            Descriptors.NumRotatableBonds(m), Descriptors.NumAromaticRings(m),
            Descriptors.FractionCSP3(m), QED.qed(m),
            Descriptors.NumAliphaticRings(m), Descriptors.NumHeteroatoms(m),
            Descriptors.NumSaturatedRings(m),
        ]
    except Exception:
        return None


def fetch(target, max_rows):
    rows = []
    offset = 0
    limit = 1000
    while len(rows) < max_rows:
        url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
               f"target_chembl_id={target}&standard_type=IC50"
               f"&limit={limit}&offset={offset}")
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        acts = data.get("activities", [])
        if not acts:
            break
        for a in acts:
            smi = a.get("canonical_smiles")
            val = a.get("standard_value")
            units = (a.get("standard_units") or "").upper()
            if not smi or val is None:
                continue
            try:
                val = float(val)
            except Exception:
                continue
            if val <= 0:
                continue
            if units == "NM":
                factor = 1e-9
            elif units == "UM":
                factor = 1e-6
            elif units == "MM":
                factor = 1e-3
            elif units == "M":
                factor = 1.0
            else:
                continue
            pic50 = -math.log10(val * factor)
            if not (0 < pic50 < 20):
                continue
            rows.append((smi, pic50))
        if len(acts) < limit:
            break
        offset += limit
    return rows


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    raw = fetch(TARGET, MAX_ROWS)
    random.seed(0)
    random.shuffle(raw)
    kept = []
    seen = set()
    for smi, pic50 in raw:
        if smi in seen:
            continue
        seen.add(smi)
        f = featurize(smi)
        if f is None:
            continue
        kept.append((smi, pic50, f))
        if len(kept) >= MAX_ROWS:
            break
    random.seed(0)
    random.shuffle(kept)
    n = len(kept)
    ntr = int(n * 0.7)
    nva = int(n * 0.15)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"f{i}" for i in range(N_FEATURES)] + ["activity", "split"])
        for i, (smi, pic50, feats) in enumerate(kept):
            sp = "train" if i < ntr else ("val" if i < ntr + nva else "test")
            w.writerow([f"mol{i}"] + [f"{x:.4f}" for x in feats] +
                       [f"{pic50:.4f}", sp])
    print(f"[gen_real] drugtarget <- ChEMBL {TARGET} IC50: {n} real compounds, "
          f"{N_FEATURES} RDKit descriptors -> {OUT}")


if __name__ == "__main__":
    main()
