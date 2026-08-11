"""SharedExperienceBank: cross-task transfer of *meta* knowledge.

The per-task ExperienceBank (experience_bank.py) only remembers GENOMES of
ONE task. This module adds a SECOND memory that is shared across tasks of the
same family (e.g. all "medical" tasks). Each task, at the end of a run, writes
a STANDARDIZED *lesson* -- not its raw genome (genomes are task-specific) but
what ARCHETYPE won and how sparse the winning solution was.

A new task of that family can then cold-start with an Occam-flavoured prior
(sparser seed + biased mutations toward the family's typical sparsity)
instead of an all-features seed. That is the Hyra-style recursive
self-improvement ACROSS tasks, layered on top of the within-task bank.

Pure standard library. No external API.
"""
import json
import os
import time
import uuid


class SharedExperienceBank:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.records = []
        self._load()

    def _load(self):
        try:
            for fn in os.listdir(self.path):
                if not fn.endswith(".json"):
                    continue
                with open(os.path.join(self.path, fn)) as f:
                    self.records.append(json.load(f))
        except Exception:
            pass

    def record(self, task_name, family, lesson):
        """Persist one transferable lesson (a dict with at least 'sparsity')."""
        if not lesson or lesson.get("sparsity") is None:
            return None
        rec = {"id": uuid.uuid4().hex[:8], "ts": time.time(),
               "task": task_name, "family": family}
        rec.update(lesson)
        self.records.append(rec)
        with open(os.path.join(self.path, f"{rec['id']}.json"), "w") as f:
            json.dump(rec, f, indent=2, default=str)
        return rec["id"]

    def priors(self, family):
        """Aggregate the winning lessons of a family into a transferable prior."""
        recs = [r for r in self.records
                if r.get("family") == family
                and r.get("sparsity") is not None]
        if not recs:
            return None
        wins = [r for r in recs if r.get("score") is not None]
        pool = wins or recs
        spars = sorted(r["sparsity"] for r in pool)
        med = spars[len(spars) // 2]
        occam = sum(1 for r in pool if r.get("won_by_parsimony")) / len(pool)
        return {
            "n_tasks": len(set(r["task"] for r in recs)),
            "n_lessons": len(recs),
            "median_sparsity": med,
            "occam_win_rate": occam,
        }
