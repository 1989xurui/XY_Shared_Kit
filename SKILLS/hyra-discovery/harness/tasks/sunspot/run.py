import sys
import os

_HYRA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _HYRA_ROOT)

from hyra import (
    ExperienceBank,
    Sandbox,
    MockBridge,
    ContextAgent,
    ProposalAgent,
    Evaluator,
    Harness,
)
from tasks.sunspot.task import SunspotTask

bank = ExperienceBank(os.path.join(os.path.dirname(__file__), "eb"))
sandbox = Sandbox(timeout=120)
llm = MockBridge()  # headless; swap for AgentBridge to inject HY3
ctx = ContextAgent(bank, llm)
prop = ProposalAgent(llm)
evaluator = Evaluator()
task = SunspotTask()

print(f"[init] train months={len(task.train)} test months={len(task.test)}")

harness = Harness(
    bank, sandbox, ctx, prop, evaluator, task, max_iters=30, budget=240
)
result = harness.run()

print("STATS:", result["stats"])
best = result["best"]
if best:
    print("BEST iter:", best.get("iter"), "score:", best.get("score"))
    print("BEST summary:", best.get("summary"))
