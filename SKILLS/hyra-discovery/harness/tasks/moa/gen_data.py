"""Generate a transparent mechanism-of-action (MOA) classification benchmark.

Hidden truth: each of the K mechanism classes is defined by ONE unique marker
feature with a class-specific mean; the other F-K features are noise (shared
distribution across classes). The agent's JOB (genome = which features to use)
is to REDISCOVER the K marker features = the mechanism signature.

Selection uses VALIDATION accuracy (only), penalized by feature count to
prefer the parsimonious signature; TEST accuracy is reported once.
"""
import os
import csv
import random

SEED = 11
HERE = os.path.dirname(os.path.abspath(__file__))
F = 20
N_CLASS = 6
N_PER = 40
MARKERS = [1, 3, 5, 7, 9, 11]   # one unique marker per class (total K=6)
SHIFT = 2.4


def main():
    random.seed(SEED)
    rows = []
    idx = 0
    for c in range(N_CLASS):
        mk = MARKERS[c]
        cmean = SHIFT * (c - (N_CLASS - 1) / 2.0)   # spread classes along axis
        for _ in range(N_PER):
            x = [random.gauss(0.0, 1.0) for _ in range(F)]
            x[mk] += random.gauss(cmean, 0.4)
            r = idx % 5
            split = "train" if r < 3 else ("val" if r == 3 else "test")
            rows.append((x, c, split))
            idx += 1
    with open(os.path.join(HERE, "data", "dataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"f{i}" for i in range(F)] + ["label", "split"])
        for x, c, split in rows:
            w.writerow([f"{v:.4f}" for v in x] + [c, split])
    print(f"[gen] moa dataset N={len(rows)} F={F} classes={N_CLASS} markers={MARKERS}")


if __name__ == "__main__":
    main()
