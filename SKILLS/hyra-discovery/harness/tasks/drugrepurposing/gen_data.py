"""Generate a transparent CMap/LINCS-style drug-repurposing benchmark.

Hidden truth (oracle ground truth, used only for validation/test SELECTION --
exactly like activity labels in drugtarget):

  * A DISEASE is driven by a gene MODULE M (10 of G=40) with MIXED direction:
    half the module genes are up-regulated, half down-regulated (a realistic
    differential-expression signature, not all-one-direction).
  * The TRUE repurposing drug (drug_0) reverses ALL of M (up->down, down->up)
    -> it reverses the disease. It lives in the TEST set.
  * VALIDATION drugs each reverse a *random subset* of M -> their true_reversal
    grows with how many module genes they hit. This gives the loop a real
    gradient: picking the mixed-sign module maximizes val-spearman.
  * Other TEST drugs have random profiles (true_reversal ~ 0); because M is the
    only structured signal, including the 30 noise genes only dilutes it, so
    the MODULE subset is what makes drug_0 top the test ranking.
"""
import os
import csv
import random
import math

SEED = 7
HERE = os.path.dirname(os.path.abspath(__file__))
G = 40
# 10 module genes with mixed disease direction (+1 up, -1 down)
MODULE = {2: +1, 5: -1, 9: +1, 11: -1, 14: +1, 20: -1, 23: +1, 27: -1, 31: +1, 37: -1}
N_PAT = 150
D = 30
TRUE_REP = 0
AMP = 3.0          # disease amplitude on module genes
BG = 1.0           # background expression noise std (realistic)
REV = 3.0          # drug reversal amplitude on module genes


def pearson(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def main():
    random.seed(SEED)
    # reversal target: drug should push module genes opposite to disease
    target = [-REV * s for g, s in sorted(MODULE.items())]
    mgenes = sorted(MODULE.keys())

    # ---- disease expression (patients) ----
    with open(os.path.join(HERE, "data", "disease_expr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"g{i}" for i in range(G)] + ["label", "split"])
        for i in range(N_PAT):
            is_dis = (i % 2 == 0)
            x = [random.gauss(0.0, BG) for _ in range(G)]
            if is_dis:
                for g, s in MODULE.items():
                    x[g] += s * random.gauss(AMP, 0.4)
            split = "train" if i < 100 else ("val" if i < 125 else "test")
            w.writerow([f"{v:.4f}" for v in x] + [1 if is_dis else 0, split])

    # ---- drug perturbation profiles ----
    with open(os.path.join(HERE, "data", "drug_profiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug_id"] + [f"g{i}" for i in range(G)] + ["split", "true_reversal"])
        for d in range(D):
            x = [random.gauss(0.0, BG) for _ in range(G)]
            if d == TRUE_REP:
                for g, s in MODULE.items():
                    x[g] = -s * random.gauss(REV, 0.4)
            elif d < 10:  # 9 validation drugs: reverse a random subset
                hit = random.sample(mgenes, random.randint(1, len(mgenes) - 1))
                for g in hit:
                    x[g] = -MODULE[g] * random.gauss(REV, 0.4)
            # ground-truth reversal strength = reversal measured on the TRUE
            # module genes only (the biological "answer key"). This cleanly
            # tracks how many module genes each drug reverses.
            mgenes_sorted = sorted(MODULE.keys())
            x_mod = [x[g] for g in mgenes_sorted]
            tgt_mod = [-REV * MODULE[g] for g in mgenes_sorted]
            oracle = pearson(x_mod, tgt_mod)
            split = "test" if d == TRUE_REP else ("val" if d < 10 else "test")
            w.writerow([f"drug_{d}"] + [f"{v:.4f}" for v in x] + [split, f"{oracle:.4f}"])

    print(f"[gen] disease_expr N={N_PAT} G={G} | drug_profiles D={D} "
          f"true_rep=drug_{TRUE_REP} module(mixed)={mgenes}")


if __name__ == "__main__":
    main()
