"""Generate the synthesis (generative) benchmark data.

The generative loop now uses REAL RDKit chemistry (see task.py): fragments are
real SMILES, molecules are built and validated with RDKit, and the objective is
real QED. This generator only produces the NOVELTY REFERENCE library -- a set of
"known" fragment combinations so the runner can report whether a generated
combination is novel. It does NOT require rdkit (fragment-set novelty is
string-based).

Hidden structure (kept for honesty about the surrogate): a generated combo is
considered "in the known library" when its fragment-set appears here. The
runner's novelty check is therefore a real set-membership test against this
library.
"""
import os
import csv
import random

SEED = 13
HERE = os.path.dirname(os.path.abspath(__file__))
NF = 12                 # must match REAL_FRAGMENTS in task.py
N_LIB = 120
MAX_FRAG = 5


def main():
    random.seed(SEED)
    lib = []
    seen = set()
    attempts = 0
    while len(lib) < N_LIB and attempts < N_LIB * 20:
        attempts += 1
        k = random.randint(2, MAX_FRAG)
        comp = tuple(sorted(random.sample(range(NF), k)))
        key = ";".join(map(str, comp))
        if key in seen:
            continue
        seen.add(key)
        lib.append(comp)
    with open(os.path.join(HERE, "data", "library.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frag_set", "activity"])
        for comp in lib:
            w.writerow([";".join(map(str, comp)), 0.0])
    # fragments.csv (synthetic tokens) kept for backward compatibility; the
    # runner uses the REAL_FRAGMENTS list from task.py, not this file.
    fragments = ["c1ccccc1", "c1ccncc1", "c1ccsc1", "C1CCCCC1", "C(=O)O",
                 "C(=O)N", "N", "O", "S", "Cl", "F", "c1ccc2ccccc2c1"]
    with open(os.path.join(HERE, "data", "fragments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "token"])
        for i, tok in enumerate(fragments):
            w.writerow([i, tok])
    print(f"[gen] synthesis novelty library N={len(lib)} NF={NF} (run with the rdkit venv for real chemistry)")


if __name__ == "__main__":
    main()
