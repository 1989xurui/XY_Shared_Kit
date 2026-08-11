"""ExperienceBank: durable store of every explored solution + its outcome."""
import json
import os
import time
import uuid


class ExperienceBank:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.records = []
        self._load()

    def _load(self):
        """Reload persisted records so the bank accumulates across runs
        (required for recursive self-improvement across agent loop steps)."""
        try:
            for fn in os.listdir(self.path):
                if not fn.endswith(".json"):
                    continue
                with open(os.path.join(self.path, fn)) as f:
                    rec = json.load(f)
                self.records.append(rec)
        except Exception:
            pass

    def add(self, record):
        rec = {"id": uuid.uuid4().hex[:8], "ts": time.time()}
        rec.update(record)
        self.records.append(rec)
        self._save(rec)
        return rec["id"]

    def _save(self, rec):
        with open(os.path.join(self.path, f"{rec['id']}.json"), "w") as f:
            json.dump(rec, f, indent=2, default=str)

    def all(self):
        return self.records

    def best(self, key="score"):
        if not self.records:
            return None
        valid = [r for r in self.records if r.get(key) is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r.get(key, float("-inf")))

    def inspirations(self, k=3):
        scored = sorted(
            self.records,
            key=lambda r: r.get("score", float("-inf")),
            reverse=True,
        )
        return scored[:k]

    def stats(self):
        if not self.records:
            return {"n": 0}
        scores = [r["score"] for r in self.records if r.get("score") is not None]
        if not scores:
            return {"n": len(self.records), "best": None}
        return {
            "n": len(self.records),
            "best": max(scores),
            "mean": sum(scores) / len(scores),
        }
