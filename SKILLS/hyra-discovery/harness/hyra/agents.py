"""ContextAgent + ProposalAgent: the two agent roles of the Harness."""


class ContextAgent:
    """Maintains the ExperienceBank; distils inspirations for the queue."""

    def __init__(self, bank, llm=None):
        self.bank = bank
        self.llm = llm

    def make_inspirations(self, k=3):
        best = self.bank.inspirations(k)
        ctx = []
        for r in best:
            ctx.append(
                {
                    "id": r["id"],
                    "score": r.get("score"),
                    "summary": r.get("summary", ""),
                    "code_snippet": (r.get("code") or "")[:400],
                }
            )
        return ctx


class ProposalAgent:
    """Consumes inspirations and produces a new solution draft (code/text)."""

    def __init__(self, llm=None):
        self.llm = llm

    def propose(self, inspirations, task_prompt):
        if self.llm is None:
            return None
        insp_text = "\n".join(
            f"[{c['id']}] score={c['score']} :: {c['summary']}" for c in inspirations
        )
        prompt = f"{task_prompt}\n\nBest known inspirations:\n{insp_text}"
        return self.llm.generate(prompt)
