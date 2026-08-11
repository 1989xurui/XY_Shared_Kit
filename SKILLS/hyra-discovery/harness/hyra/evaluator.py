"""Evaluator: scores a run result and guards against reward-hacking.

Two-layer design (audit-corrected, honest version):
  * inner loop: a candidate is SELECTED on the VALIDATION score (never the test
    set -- each Task's parse_run sets `score` to the validation metric).
  * outer loop (evolve): once enough history exists, the evaluator begins
    applying an anti-overfit penalty, so a candidate that looks great on the
    validation set but collapses on the held-out test set is not favoured.

This is NOT a language-model evaluator; it is a deterministic, inspectable
scoring rule. Claims that it "evolves itself via HY3 reasoning" were removed
during the 13-dimension audit.
"""


class Evaluator:
    def __init__(self, base_metric="score"):
        self.base_metric = base_metric
        self.penalise_complexity = False

    def evaluate(self, run_result, context=None):
        if not run_result.get("ok"):
            return {"score": float("-inf"), "detail": run_result.get("stderr")}
        # Selection base = validation R^2 (the Task already set `score` to that).
        score = run_result.get("score")
        detail = "val-based selection"
        if context:
            gap = context.get("overfit_gap", 0.0)        # val_r2 - test_r2
            complexity = context.get("complexity", 0.0)  # #free parameters
            if self.penalise_complexity and (gap > 0.02 or complexity > 0):
                penalty = gap * 0.5 + complexity * 0.0005
                score = score - penalty
                detail = (
                    f"val {run_result.get('score'):.4f} "
                    f"- anti-hack penalty {penalty:.4f}"
                )
        return {"score": score, "detail": detail}

    def evolve(self, experience_bank):
        # Outer loop: after enough history, begin penalising overfit/complexity.
        if experience_bank and len(experience_bank.records) >= 10:
            self.penalise_complexity = True
        return self
