"""Harness: the recursive self-improvement loop driver.

Single iteration = plan -> write code -> run -> debug -> self-test -> deliver
(compact form executed by ContextAgent/ProposalAgent + Sandbox + Evaluator).
The outer loop periodically co-evolves the Evaluator.
"""
import os
import time


class Harness:
    def __init__(
        self,
        bank,
        sandbox,
        context_agent,
        proposal_agent,
        evaluator,
        task,
        max_iters=20,
        budget=None,
        evolve_every=5,
    ):
        self.bank = bank
        self.sandbox = sandbox
        self.ctx = context_agent
        self.prop = proposal_agent
        self.evaluator = evaluator
        self.task = task
        self.max_iters = max_iters
        self.budget = budget
        self.evolve_every = evolve_every
        self.start = time.time()

    def run(self):
        log = []
        for i in range(self.max_iters):
            if self.budget and (time.time() - self.start) > self.budget:
                log.append({"iter": i, "event": "budget_exhausted"})
                break

            insp = self.ctx.make_inspirations(k=3)
            proposal = self.prop.propose(insp, self.task.proposal_prompt())

            solution_code, summary = self.task.make_solution(proposal, insp, i)
            run = self.sandbox.run(
                solution_code, self.task.runner_code(), self.task.workdir(i)
            )
            parsed = self.task.parse_run(run)
            ev = self.evaluator.evaluate(parsed, context={"iter": i})
            score = ev.get("score")

            self.bank.add(
                {
                    "iter": i,
                    "code": solution_code,
                    "summary": summary,
                    "score": score,
                    "log": run.get("stdout"),
                    "feedback": ev.get("detail"),
                }
            )
            log.append({"iter": i, "score": score, "summary": summary})

            if i > 0 and i % self.evolve_every == 0:
                self.evaluator = self.evaluator.evolve(self.bank)

        best = self.bank.best()
        return {"log": log, "best": best, "stats": self.bank.stats()}
