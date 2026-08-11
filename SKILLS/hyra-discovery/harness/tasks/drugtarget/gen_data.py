"""Generate a SYNTHETIC drug-target benchmark for the DrugTargetTask.

This is NOT real biochemical data -- it is a transparent, reproducible stand-in
that mimics a ligand-based SAR (structure-activity relationship) table so the
autonomous discovery loop can be demonstrated end-to-end without GPUs/docking.

Ground truth: activity is driven ONLY by features 2, 5, 11 plus a 2x5
interaction. ACTIVE features live in [0,1]; all other (decoy) features live in
a WIDE range [-4,4]. Because the decoys have large variance, a linear model
that includes them overfits and generalises worse on the validation set -- so
the loop is *rewarded* for redIScovering the sparse true subset {2,5,11}+inter.

Output:
  data/dataset.csv     -- header: id, f0..f{F-1}, activity, split
  data/reference.csv   -- one row: a "reference molecule" (olaparib-like) with a
                          partial active profile, used for "beat the reference".
"""
import csv
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "data")

F = 12
N = 90
ACTIVE = {2: 1.5, 5: -1.0, 11: 0.8}   # true feature -> weight
ACTIVE_IDX = set(ACTIVE)
INTER_PAIR = (2, 5)                  # true interaction
INTER_W = 0.6
NOISE_SD = 0.05


def gen_x(rng):
    x = [0.0] * F
    for i in range(F):
        if i in ACTIVE_IDX:
            x[i] = rng.random()        # well-behaved signal in [0,1]
        else:
            x[i] = rng.uniform(-4, 4)  # wide-range decoy -> hurts generalisation
    return x


def activity(x):
    a = 0.0
    for i, w in ACTIVE.items():
        a += w * x[i]
    a += INTER_W * x[INTER_PAIR[0]] * x[INTER_PAIR[1]]
    return a


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(20240722)
    rows = []
    for n in range(N):
        x = gen_x(rng)
        y = activity(x) + rng.gauss(0.0, NOISE_SD)
        rows.append((n, x, y))

    n_train = int(N * 0.45)
    n_val = int(N * 0.275)
    out = []
    for n, x, y in rows:
        sp = "train" if n < n_train else ("val" if n < n_train + n_val else "test")
        out.append((n, x, y, sp))

    with open(os.path.join(OUT_DIR, "dataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"f{i}" for i in range(F)] + ["activity", "split"])
        for n, x, y, sp in out:
            w.writerow([n] + [f"{v:.6f}" for v in x] + [f"{y:.6f}", sp])

    # reference molecule: partial active profile in [0,1], decoys at 0
    ref = [0.0] * F
    ref[2] = 0.9
    ref[5] = 0.5
    ref[11] = 0.7
    with open(os.path.join(OUT_DIR, "reference.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"f{i}" for i in range(F)])
        w.writerow([f"{v:.6f}" for v in ref])

    print(f"wrote {len(out)} rows ({n_train} train / {n_val} val / "
          f"{N - n_train - n_val} test) to data/dataset.csv")
    print(f"true active features = {sorted(ACTIVE)} (+ interaction {INTER_PAIR})")


if __name__ == "__main__":
    main()
