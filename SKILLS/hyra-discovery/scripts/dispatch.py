#!/usr/bin/env python3
"""Dispatch entry for the hyra-discovery skill.

Resolves the bundled harness root (this file lives in <skill>/scripts/, the
harness lives in <skill>/harness/) and runs the requested mode:

  auto     unattended evolution loop (GuidedProposer + Evaluator + Sandbox)
  step     single HY3-driven step: you supply a solution.py, harness scores it
  smoke    run the bundled smoke + discovery self-tests
  gen-data (re)generate the synthetic dataset for the chosen TASK

The harness uses paths relative to its own __file__, so this works no matter
where WorkBuddy invokes the skill from. Each medical task ships a gen_data
module, so `dispatch <task> --mode gen-data` regenerates its benchmark.
"""
import argparse
import os
import subprocess
import sys

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.abspath(os.path.join(SKILL_ROOT, "harness"))

# task module prefix -> its gen_data module (run to (re)create benchmark data)
GEN_DATA = {
    "tasks.drugtarget": "tasks.drugtarget.gen_data",
    "tasks.drugrepurposing": "tasks.drugrepurposing.gen_data",
    "tasks.moa": "tasks.moa.gen_data",
    "tasks.synthesis": "tasks.synthesis.gen_data",
    "tasks.drugcombo": "tasks.drugcombo.gen_data",
    "tasks.mathlaw": "tasks.mathlaw.gen_data",
    "tasks.quantum_routing": "tasks.quantum_routing.gen_data",
    # sunspot ships with a fixed historical CSV; documented but no generator
}


def _run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=HARNESS)


def ensure_data(task):
    gen = GEN_DATA.get(task)
    if not gen:
        return
    # crude "data exists?" check: each task stores data under tasks/<name>/data/
    name = task.split(".")[1]
    data_dir = os.path.join(HARNESS, "tasks", name, "data")
    if os.path.isdir(data_dir) and any(
            os.path.isfile(os.path.join(data_dir, f)) for f in os.listdir(data_dir)):
        return
    print(f"[dispatch] {name} dataset missing -> generating ...", flush=True)
    _run([sys.executable, "-m", gen])


def main():
    ap = argparse.ArgumentParser(prog="hyra-discovery")
    ap.add_argument("task", nargs="?", default="tasks.drugtarget.task")
    ap.add_argument("--mode",
                    choices=["auto", "step", "smoke", "gen-data"],
                    default="auto")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--fresh", action="store_true",
                    help="clear the persisted ExperienceBank and start a clean run")
    ap.add_argument("--solution", default=None,
                    help="path to a solution.py (step mode)")
    ap.add_argument("--summary", default="",
                    help="reasoning summary for the committed record (step mode)")
    ap.add_argument("--iter-id", default="0")
    args = ap.parse_args()

    if args.mode == "smoke":
        return _run([sys.executable, "test_smoke.py"]).returncode

    if args.mode == "gen-data":
        gen = GEN_DATA.get(args.task)
        if not gen:
            print(f"[dispatch] no gen_data for {args.task}", file=sys.stderr)
            return 2
        return _run([sys.executable, "-m", gen]).returncode

    ensure_data(args.task)

    if args.mode == "auto":
        cmd = [sys.executable, "auto_loop.py", args.task,
               "--iters", str(args.iters)]
        if args.fresh:
            cmd.append("--fresh")
        return _run(cmd).returncode

    if args.mode == "step":
        if not args.solution:
            print("step mode requires --solution <path-to-solution.py>",
                  file=sys.stderr)
            return 2
        cmd = [sys.executable, "hy3_step.py", args.task,
               args.solution, args.summary, args.iter_id]
        return _run(cmd).returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
