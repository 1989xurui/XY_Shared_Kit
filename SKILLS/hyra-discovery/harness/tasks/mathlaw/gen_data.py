"""Generate a transparent synthetic physics-law dataset for the mathlaw task.

Hidden ground truth: Kepler's third law  T = k * a**1.5  (k = 1 for a in AU,
T in years -- the actual solar-system relation). A nuisance feature z (uniform,
uncorrelated) and Gaussian observation noise are added so the discovery loop
must (a) pick the a**1.5 transform and (b) reject the useless z feature.

This is a LOCAL SYMBOLIC-REGRESSION PROXY for "math scientific discovery". It is
NOT the Hyra 15-parameter addition Transformer (that needs GPU / torch and is
out of scope for a pure-stdlib harness).
"""
import os
import csv
import math
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "dataset.csv")

N_TRAIN, N_VAL, N_TEST = 120, 40, 40
K = 1.0            # Kepler constant (solar-system units)
NOISE = 0.03       # relative Gaussian noise on T
SEED = 20240722


def main():
    rng = random.Random(SEED)
    rows = []
    for _ in range(N_TRAIN + N_VAL + N_TEST):
        a = rng.uniform(0.3, 30.0)              # semi-major axis (AU)
        z = rng.uniform(-1.0, 1.0)              # nuisance feature (uncorrelated)
        t_true = K * (a ** 1.5)
        t = t_true * (1.0 + rng.gauss(0, NOISE))
        rows.append((round(a, 6), round(z, 6), round(t, 6)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "z", "T"])
        for r in rows:
            w.writerow(r)
    print("wrote %d rows -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
