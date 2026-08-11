"""Generate a SYNTHETIC drug-combination synergy benchmark for DrugComboTask.

NOT real clinical data -- a transparent stand-in mirroring DrugComb / NCI-ALMANAC
combo screens so the autonomous synergy-discovery loop can be demonstrated
end-to-end without real PK/PD.

Representation of a drug PAIR (i, j):
  - A-features : mechanism bits of drug i        (M dims)
  - B-features : mechanism bits of drug j        (M dims)
  - interaction: f_i[k] * f_j[k]  (same-index)   (M dims)
  -> 3M = 48 feature columns.

Ground truth (sparse synergy law):
  synergy = 0.8*(f_i[3]*f_j[3])      # same-pathway AND interaction  -> idx 35
          + 0.6* f_i[1]              # drug i hits pathway 1         -> idx 1
          + 0.5* f_j[11]             # drug j hits pathway 11        -> idx 27
          + 0.4*(f_i[5]*f_j[5])      # same-pathway AND interaction  -> idx 37
          + noise

So the true relevant feature indices are {1, 27, 35, 37} out of 48 -- a sparse
subset the loop must rediscover, beating the all-features baseline.
"""
import csv
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "data")

D = 40          # number of drugs
M = 16          # mechanism features per drug  -> 3M = 48 pair-features
NOISE_SD = 0.03
SEED = 20240722


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    # drug mechanism bit-vectors
    drugs = []
    for d in range(D):
        drugs.append([rng.randint(0, 1) for _ in range(M)])

    # build all unordered pairs
    pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            fi, fj = drugs[i], drugs[j]
            a = list(fi)
            b = list(fj)
            inter = [fi[k] * fj[k] for k in range(M)]
            feat = a + b + inter           # 48 dims
            # ground-truth synergy
            y = (0.8 * (fi[3] * fj[3])
                 + 0.6 * fi[1]
                 + 0.5 * fj[11]
                 + 0.4 * (fi[5] * fj[5])
                 + rng.gauss(0.0, NOISE_SD))
            pairs.append((feat, y))

    # split pairs (not drugs) into train/val/test
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    out = []
    for idx, (feat, y) in enumerate(pairs):
        sp = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
        out.append((idx, feat, y, sp))

    with open(os.path.join(OUT_DIR, "dataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"f{k}" for k in range(3 * M)] + ["synergy", "split"])
        for idx, feat, y, sp in out:
            w.writerow([idx] + [f"{v:.6f}" for v in feat] + [f"{y:.6f}", sp])

    print(f"wrote {len(out)} drug pairs ({n_train} train / {n_val} val / "
          f"{n - n_train - n_val} test) to data/dataset.csv")
    print("true synergy-relevant feature indices = {1, 27, 35, 37} of 48")


if __name__ == "__main__":
    main()
