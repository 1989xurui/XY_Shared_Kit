"""Generic HY3-driven single step (task-agnostic).

Usage:
    python hy3_step.py <task_module> <solution_path> <summary> [iter_id]

Examples:
    python hy3_step.py tasks.sunspot.task  tasks/sunspot/hy3_proposals/sol_h4.py "..." h4
    python hy3_step.py tasks.othello.task  tasks/othello/hy3_proposals/bot_h0.py   "..." h0

The running WorkBuddy/HY3 agent supplies the `solution.py` (the ProposalAgent
role). This script runs it in the Sandbox, parses the score, and commits
(code, score, reasoning) to the task's ExperienceBank. This is the concrete
"HY3 as compute" execution path shared by every acceptance test.
"""
import sys
import os
import json

_HYRA_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HYRA_ROOT)

from hyra import ExperienceBank, Sandbox

EB_SUFFIX = "eb_hy3"  # HY3-driven bank lives alongside any pre-existing bank


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"error": "usage: hy3_step.py <task_module> <sol_path> <summary> [iter_id]"}))
        return
    task_module_name = sys.argv[1]
    sol_path = sys.argv[2]
    summary = sys.argv[3]
    iter_id = sys.argv[4] if len(sys.argv) > 4 else "0"

    task_mod = __import__(task_module_name, fromlist=["Task", "EB_DIR", "SANDBOX_TIMEOUT"])
    Task = getattr(task_mod, "Task", None)
    if Task is None:
        for _v in vars(task_mod).values():
            if isinstance(_v, type) and _v.__name__.endswith("Task"):
                Task = _v
                break
    if Task is None:
        print(json.dumps({"error": f"no Task class found in {task_module_name}"}))
        return
    eb_dir = getattr(task_mod, "EB_DIR", None) or os.path.join(_HYRA_ROOT, "eb")
    timeout = getattr(task_mod, "SANDBOX_TIMEOUT", 120)

    with open(sol_path, "r") as f:
        solution_code = f.read()

    bank = ExperienceBank(eb_dir)
    sandbox = Sandbox(timeout=timeout)
    task = Task()

    workdir = task.workdir(f"hy3_{iter_id}")
    run = sandbox.run(solution_code, task.runner_code(), workdir)
    if "HY3_DEBUG" in os.environ:
        print("DBG run:", json.dumps({"ok": run.get("ok"), "rc": run.get("rc"),
                                       "stdout": run.get("stdout"),
                                       "stderr": run.get("stderr")[:300]}, default=str))
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
