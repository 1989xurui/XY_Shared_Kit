"""HY3-driven single step.

This is the "HY3 as compute" execution path. Instead of MockBridge random
mutation, the ProposalAgent role is played by the running WorkBuddy/HY3 agent
(in-context). The agent reasons about the task, writes a candidate `solution.py`,
then calls this script which:
  1. runs the candidate in the Sandbox (isolated subprocess)
  2. parses the score
  3. commits the (code, score, reasoning) triple to a dedicated ExperienceBank
  4. prints a compact JSON state for the next reasoning step

The loop is: reason -> write solution -> hy3_step -> read score -> refine.
"""
import sys
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_HYRA_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HYRA_ROOT)

from hyra import ExperienceBank, Sandbox
from tasks.sunspot.task import SunspotTask

EB_DIR = os.path.join(_HERE, "eb_hy3")
WORK_ROOT = os.path.join(_HERE, "solutions")


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: hy3_step.py <sol_path> <summary> [iter_id]"}))
        return
    sol_path = sys.argv[1]
    summary = sys.argv[2]
    iter_id = sys.argv[3] if len(sys.argv) > 3 else "0"

    with open(sol_path, "r") as f:
        solution_code = f.read()

    bank = ExperienceBank(EB_DIR)
    sandbox = Sandbox(timeout=120)
    task = SunspotTask()

    workdir = os.path.join(WORK_ROOT, f"hy3_{iter_id}")
    runner_code = task.runner_code()
    run = sandbox.run(solution_code, runner_code, workdir)
    parsed = task.parse_run(run)

    score = parsed.get("score") if parsed.get("ok") else None
    bank.add({
        "iter": iter_id,
        "code": solution_code,
        "summary": summary,
        "score": score,
        "log": run.get("stdout"),
        "feedback": run.get("stderr") if not run.get("ok") else None,
    })

    best = bank.best()
    stats = bank.stats()
    print(json.dumps({
        "ok": run.get("ok"),
        "score": score,
        "best": best.get("score") if best else None,
        "best_summary": best.get("summary") if best else None,
        "n": stats.get("n"),
        "mean": stats.get("mean"),
        "stderr_tail": (run.get("stderr") or "")[-240:],
    }, default=str))


if __name__ == "__main__":
    main()
