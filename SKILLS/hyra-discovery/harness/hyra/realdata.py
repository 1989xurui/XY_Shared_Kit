"""Real-data adapter for the medical tasks.

Each task normally runs on a BUNDLED SYNTHETIC dataset (transparent ground
truth). To point a task at REAL data instead, drop the matching CSV(s) into
``<task>/data_real/`` and the harness auto-switches (no code change needed).

Convention (schema must match the synthetic file -- same columns / feature
count / target column name):
  - drugtarget / moa / drugcombo : put ``dataset.csv`` in ``data_real/``.
  - drugrepurposing              : put ``disease_expr.csv`` + ``drug_profiles.csv``.
  - synthesis                    : put ``fragments.csv`` (idx,token) +
                                   ``library.csv`` (frag_set,activity); or rely
                                   on the built-in RDKit real fragment library.

If ``data_real/`` is absent/empty, the bundled synthetic file is used (default).

Honesty note: this only swaps the DATA SOURCE. The search loop, evaluation
metric and selection logic are unchanged. A real dataset therefore yields a
real predictive R^2 vs baseline -- it does NOT recover a "true cause" (real
data has no transparent ground-truth subset), and it is still NOT an official
Hyra / external-benchmark acceptance.
"""
import os


def discover_csv(task_here, synthetic_rel, real_name=None):
    """Return the real CSV path if present, else the synthetic one."""
    real_dir = os.path.join(task_here, "data_real")
    if os.path.isdir(real_dir):
        name = real_name or os.path.basename(synthetic_rel)
        rp = os.path.join(real_dir, name)
        if os.path.exists(rp):
            return rp
    return os.path.join(task_here, synthetic_rel)


def realdata_present(task_here, *names):
    """True if data_real/ exists and contains the named csv(s) (or any csv)."""
    real_dir = os.path.join(task_here, "data_real")
    if not os.path.isdir(real_dir):
        return False
    if not names:
        return bool([f for f in os.listdir(real_dir) if f.endswith(".csv")])
    return all(os.path.exists(os.path.join(real_dir, n)) for n in names)
