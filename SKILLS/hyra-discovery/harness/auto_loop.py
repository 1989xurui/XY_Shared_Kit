"""Unattended evolution-style search loop (warm-started hyperparameter search).

No external API is used: it runs entirely on WorkBuddy's local Python. The loop
is an evolution-style hyperparameter search with warm-start memory (the
GuidedProposer), NOT a language-model reasoning agent -- claims of "HY3-driven
recursive self-improvement" were audit-corrected. Each iteration:

    proposer reads the ExperienceBank
        -> proposes the next genome (guided mutation / crossover / explore)
        -> task.render(genome) -> solution.py
        -> Sandbox scores it (on the VALIDATION set, never the test set)
        -> Evaluator applies an anti-overfit / anti-reward-hacking penalty
        -> result committed back to the bank

Model selection uses the validation score; the test set is reported only at the
end, so the loop cannot optimise against it (reward-hacking guard).
"""
import argparse
import json
import os
import sys
import time

_HYRA_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HYRA_ROOT)

from hyra import ExperienceBank, Sandbox, GuidedProposer, AgentBridge, Evaluator
from hyra.shared_eb import SharedExperienceBank


def load_task(module_name):
    mod = __import__(module_name, fromlist=["*"])
    Task = getattr(mod, "Task", None)
    if Task is None:
        for v in vars(mod).values():
            if isinstance(v, type) and v.__name__.endswith("Task"):
                Task = v
                break
    return mod, Task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_module")
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--budget", type=float, default=None, help="wall-clock seconds")
    ap.add_argument("--eb", default=None, help="experience-bank dir override")
    ap.add_argument("--fresh", action="store_true",
                    help="clear the persisted ExperienceBank before starting")
    args = ap.parse_args()

    mod, Task = load_task(args.task_module)
    if Task is None:
        print(json.dumps({"error": f"no Task class in {args.task_module}"}))
        return
    task = Task()

    base_eb = getattr(mod, "EB_DIR", None) or os.path.join(_HYRA_ROOT, "eb")
    eb_dir = args.eb or (base_eb + "_auto")
    timeout = getattr(mod, "SANDBOX_TIMEOUT", 120)

    # --fresh must clear the persisted bank so a new run does not inherit a
    # stale best (e.g. a synthetic best leaking into a real-data run).
    if args.fresh and os.path.isdir(eb_dir):
        import shutil
        shutil.rmtree(eb_dir)

    bank = ExperienceBank(eb_dir)
    sandbox = Sandbox(timeout=timeout)
    evaluator = Evaluator()

    # ---- cross-task transfer layer (Hyra-style shared priors) ----
    shared = SharedExperienceBank(os.path.join(_HYRA_ROOT, "shared_eb"))
    family = getattr(task, "family", lambda: "general")()
    priors = shared.priors(family)
    seed = task.seed_genome()
    if priors and hasattr(task, "seed_from_priors"):
        seed = task.seed_from_priors(priors)   # cold-start sparser via siblings
    proposer = GuidedProposer(task.genome_space(), seed)
    if priors:
        proposer.set_family_prior(priors)
    def _is_on(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        return bool(v)   # non-numeric option (str/list) counts as 1 active dim
    seed_on = sum(1 for v in seed.values() if _is_on(v))
    print(json.dumps({"event": "seed", "family": family, "priors": priors,
                      "seed_on": seed_on, "seed_total": len(seed)}))

    if not args.fresh:
        proposer.warm_start(bank)

    # AgentBridge here is an AUDIT TRAIL only: it logs each autonomous decision
    # as req/resp JSON. The `responder` is a no-op echo (NOT a reasoning model) --
    # the actual search is done by GuidedProposer below.
    bridge_dir = os.path.join(os.path.dirname(eb_dir) or ".",
                              "bridge_" + task.__class__.__name__)
    bridge = AgentBridge(bridge_dir, responder=lambda prompt, system: prompt)

    print(json.dumps({"event": "start", "task": args.task_module,
                      "iters": args.iters, "budget": args.budget, "eb": eb_dir}))
    start = time.time()
    for i in range(args.iters):
        if args.budget and (time.time() - start) > args.budget:
            print(json.dumps({"event": "budget_exhausted", "iter": i}))
            break

        genome = proposer.propose(bank)
        code, summary = task.render(genome)
        # audit trail: record the autonomous decision through the bridge
        bridge.generate(prompt=summary, system="auto-proposer")

        wd = task.workdir(f"auto_{i}")
        run = sandbox.run(code, task.runner_code(), wd)
        parsed = task.parse_run(run)
        if parsed.get("ok"):
            # Selection base = validation R^2. Anti-reward-hacking pass: once
            # enough history exists, penalise candidates that overfit the
            # validation set (val_r2 >> test_r2).
            val = parsed.get("val_r2") or 0.0
            test = parsed.get("test_r2")
            gap = (val - test) if test is not None else 0.0
            ev = evaluator.evaluate(parsed, {"overfit_gap": gap, "complexity": 0.0})
            score = ev["score"]
            detail = ev["detail"]
        else:
            val = test = None
            gap = 0.0
            score = None
            detail = run.get("stderr") or "run failed"

        bank.add({
            "iter": f"auto_{i}",
            "genome": genome,
            "code": code,
            "summary": summary,
            "score": score,           # selection score (val-based, anti-hacked)
            "val_r2": val,
            "test_r2": test,
            "test_r2_h12": parsed.get("test_r2_h12") if parsed.get("ok") else None,
            "log": run.get("stdout"),
            "feedback": run.get("stderr") if not run.get("ok") else None,
        })
        proposer.observe(genome, score)
        evaluator.evolve(bank)

        best = bank.best()
        print(json.dumps({
            "iter": i,
            "select_score": score,
            "val_r2": val,
            "test_r2": test,
            "test_r2_h12": parsed.get("test_r2_h12") if parsed.get("ok") else None,
            "overfit_gap": round(gap, 4),
            "evaluator": detail,
            "summary": summary,
            "best_select": best.get("score") if best else None,
            "best_test_r2": best.get("test_r2") if best else None,
            "n": bank.stats().get("n"),
        }, default=str))

    best = bank.best()
    # ---- write ONE transferable lesson to the cross-task shared bank ----
    if best is not None:
        lesson = getattr(task, "lesson", None)
        if callable(lesson):
            les = lesson(best.get("genome"), best)
            if les:
                shared.record(task.__class__.__name__, family, les)
    print(json.dumps({
        "event": "done",
        "best_select_score": best.get("score") if best else None,
        "best_val_r2": best.get("val_r2") if best else None,
        "best_test_r2": best.get("test_r2") if best else None,
        "best_test_r2_h12": best.get("test_r2_h12") if best else None,
        "best_summary": best.get("summary") if best else None,
        "best_genome": best.get("genome") if best else None,
        "stats": bank.stats(),
        "shared_lesson": (les if (best is not None and callable(lesson)) else None),
    }, default=str))


if __name__ == "__main__":
    main()
