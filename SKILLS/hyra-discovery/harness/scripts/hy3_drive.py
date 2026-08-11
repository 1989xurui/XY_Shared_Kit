#!/usr/bin/env python3
"""hy3_drive.py — HY3-driven discovery loop executor (sandbox scorer + bank).

Design (per the "harness runs inside WorkBuddy, HY3 is the compute" model):
  * HY3 (the in-session LLM) does the REASONING: reads the experience bank,
    forms a hypothesis (a `genome` = a candidate solution), and passes it here.
  * This script ONLY: (1) renders the genome to solution.py, (2) runs the
    task's sandbox scorer, (3) parses the result, (4) persists it to the
    ExperienceBank. Repeat across turns to drive the loop autonomously.

This is the PRIMARY path. GuidedProposer/auto_loop.py is the headless fallback
when no LLM session is available.

Usage (in a WorkBuddy session, cwd = harness/):
  # score a hypothesis proposed by HY3 (human-readable subset form):
  python scripts/hy3_drive.py --task tasks.drugtarget.task --sel 2,5,11 --inter 1
  # or pass a raw genome dict:
  python scripts/hy3_drive.py --task tasks.drugtarget.task --genome '{"f2":1,...}'
  # evidence for HY3 reasoning — per-feature correlation with the target:
  python scripts/hy3_drive.py --task tasks.drugtarget.task --scan
  # inspect the loop state:
  python scripts/hy3_drive.py --task tasks.drugtarget.task --best
  python scripts/hy3_drive.py --task tasks.drugtarget.task --history
"""
import os
import sys
import json
import csv as _csv
import importlib
import argparse
import subprocess

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_importable():
    if HARNESS not in sys.path:
        sys.path.insert(0, HARNESS)


def load_task(module):
    ensure_importable()
    mod = importlib.import_module(module)
    task = mod.Task()
    eb_default = getattr(mod, "EB_DIR", os.path.join(HARNESS, "eb_default"))
    return task, eb_default, mod


def n_features(task):
    space = task.genome_space()
    return len([k for k in space if k.startswith("f")])


def genome_from_args(args, F):
    if args.genome:
        return json.loads(args.genome)
    if args.sel is not None:
        sel = [int(x) for x in args.sel.split(",") if x.strip() != ""]
        g = {f"f{i}": (1 if i in sel else 0) for i in range(F)}
        g["inter"] = 1 if args.inter else 0
        return g
    return None


def run_candidate(task, genome, eb_dir):
    code, summary = task.render(genome)
    wd = task.workdir("hy3")
    os.makedirs(wd, exist_ok=True)
    with open(os.path.join(wd, "solution.py"), "w") as f:
        f.write(code)
    runner = task.runner_code()
    run_path = os.path.join(wd, "runner.py")
    with open(run_path, "w") as f:
        f.write(runner)
    # soft sandbox: drop PYTHONPATH so the candidate can't import harness internals
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, run_path], cwd=wd,
                          capture_output=True, text=True, env=env, timeout=180)
    run = {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}
    parsed = task.parse_run(run)
    rec = {
        "genome": genome,
        "summary": summary,
        "task": task.__class__.__name__,
        "score": parsed.get("score"),
        "val_r2": parsed.get("val_r2"),
        "test_r2": parsed.get("test_r2"),
        "cv_adj": parsed.get("cv_adj"),
        "baseline_test_r2": parsed.get("baseline_test_r2"),
        "beats_baseline": parsed.get("beats_baseline"),
        "selected": parsed.get("selected"),
        "inter": parsed.get("inter"),
        "kind": parsed.get("kind"),
        "beats_ref": parsed.get("beats_ref"),
    }
    from hyra.experience_bank import ExperienceBank
    eb = ExperienceBank(eb_dir)
    eb.add(rec)
    return rec


def do_scan(task, mod):
    data = getattr(mod, "DATA", None)
    if not data or not os.path.exists(data):
        print("scan: no DATA on task module")
        return
    with open(data) as f:
        r = _csv.DictReader(f)
        feats = [c for c in r.fieldnames
                 if c not in ("id", "activity", "label", "split")]
        rows = list(r)
    yk = "label" if ("label" in (rows[0] if rows else {})) else "activity"
    ys = [float(row[yk]) for row in rows]

    def pearson(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** 0.5
        vy = sum((b - my) ** 2 for b in ys) ** 0.5
        return cov / (vx * vy) if vx and vy else 0.0

    print(f"# scan: n={len(rows)} target={yk}")
    ranked = []
    for fc in feats:
        xs = [float(row[fc]) for row in rows]
        ranked.append((fc, pearson(xs, ys)))
    ranked.sort(key=lambda t: abs(t[1]), reverse=True)
    for fc, cc in ranked:
        print(f"  {fc:5s} r={cc:+.3f}")
    print("# HY3: use the top-abs features as the initial hypothesis.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--sel", help="comma feature subset, e.g. 2,5,11")
    ap.add_argument("--inter", action="store_true", help="enable 2-way interactions")
    ap.add_argument("--genome", help="raw genome dict as JSON")
    ap.add_argument("--eb", help="experience-bank dir (default: task.EB_DIR)")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="clear the experience bank before scoring")
    args = ap.parse_args()

    task, eb_default, mod = load_task(args.task)
    # Real-data mode gets its OWN bank so a synthetic run's higher (in-sample)
    # scores can't mask the real-data best in --best/--history queries.
    eb_real = eb_default.replace("eb_hy3", "eb_real")
    if args.eb:
        eb_dir = args.eb
    else:
        eb_dir = eb_real if task.realdata_available() else eb_default
    if args.fresh and os.path.isdir(eb_dir):
        import shutil
        shutil.rmtree(eb_dir)

    if args.scan:
        do_scan(task, mod)
        return

    if args.best:
        from hyra.experience_bank import ExperienceBank
        eb = ExperienceBank(eb_dir)
        print(json.dumps(eb.best(), default=str, indent=2))
        return
    if args.history:
        from hyra.experience_bank import ExperienceBank
        eb = ExperienceBank(eb_dir)
        for r in eb.all():
            print(json.dumps({k: r.get(k) for k in
                              ["summary", "score", "val_r2", "test_r2",
                               "cv_adj", "beats_baseline"]}, default=str))
        print("STATS", json.dumps(eb.stats(), default=str))
        return

    F = n_features(task)
    genome = genome_from_args(args, F)
    if genome is None:
        genome = task.seed_genome()
    rec = run_candidate(task, genome, eb_dir)
    print(json.dumps(rec, default=str, indent=2))


if __name__ == "__main__":
    main()
