"""GuidedProposer: the distilled-HY3 policy that drives the unattended loop.

No external API. This encodes, as an explicit search policy, the reasoning
pattern HY3 already demonstrated by hand on the sunspot test:

    exploit the best known idea -> perturb one/two dimensions -> occasionally
    recombine the two best ideas -> sometimes jump to an unexplored region.

Operating over a task-supplied *genome space* (a dict of {key: [options]}),
it reads the ExperienceBank each step, proposes the next genome to try, and
never repeats a genome it has already seen. This is what lets the AgentBridge
run the recursive-self-improvement loop with zero human intervention.
"""
import json
import random


class GuidedProposer:
    def __init__(self, space, seed, explore=0.30, crossover=0.35, seed_state=0):
        """
        space     : dict {key: [allowed values]} -- the search space.
        seed      : dict -- the starting genome (used before the bank has data).
        explore   : prob. of a random jump to an unexplored region.
        crossover : prob. (when >=2 elites exist) of recombining the top two.
        """
        self.space = space
        self.seed = seed
        self.explore = explore
        self.crossover = crossover
        self.tried = set()
        self.rng = random.Random(seed_state)
        # cross-task prior (set via set_family_prior); None => no bias
        self.prior_sparsity = None
        self.occam_bias = 0.0

    # -- cross-task transfer (Hyra-style shared priors) -------------------
    def set_family_prior(self, priors):
        """Bias search toward the family's typical winning sparsity.

        When a sibling medical task historically won with sparse subsets, a
        new medical task should cold-start sparser and let mutations lean
        Occam -- instead of defaulting to an all-features seed.
        """
        if not priors:
            return
        self.prior_sparsity = priors.get("median_sparsity")
        self.occam_bias = 0.5 if priors.get("occam_win_rate", 0) >= 0.5 else 0.25

    def _is_bool(self, vals):
        return isinstance(vals, list) and len(vals) == 2 and set(vals) == {0, 1}

    def _pick(self, k, vals):
        # apply Occam transfer only on boolean feature dims
        if self.prior_sparsity is not None and self._is_bool(vals):
            if self.rng.random() < self.occam_bias:
                return 0 if self.rng.random() > self.prior_sparsity else 1
        return self.rng.choice(vals)

    # -- helpers ------------------------------------------------------------
    def _key(self, g):
        return json.dumps(g, sort_keys=True, default=str)

    def _elites(self, bank):
        recs = [r for r in bank.all()
                if r.get("score") is not None and r.get("genome")]
        recs.sort(key=lambda r: r["score"], reverse=True)
        return recs

    def _mutate(self, g):
        ng = dict(g)
        keys = list(self.space.keys())
        n = self.rng.choice([1, 1, 2])
        for k in self.rng.sample(keys, min(n, len(keys))):
            ng[k] = self._pick(k, self.space[k])
        return ng

    def _crossover(self, a, b):
        return {k: self.rng.choice([a.get(k), b.get(k)]) for k in self.space}

    def _random(self):
        return {k: self._pick(k, v) for k, v in self.space.items()}

    # -- api ----------------------------------------------------------------
    def warm_start(self, bank):
        """Seed the 'tried' set from persisted genomes so a resumed run does
        not waste iterations re-evaluating what the bank already knows."""
        for r in bank.all():
            if r.get("genome"):
                self.tried.add(self._key(r["genome"]))

    def propose(self, bank):
        elites = self._elites(bank)
        for _ in range(60):
            if not elites:
                g = dict(self.seed)
            elif self.rng.random() < self.explore:
                g = self._random()
            elif len(elites) >= 2 and self.rng.random() < self.crossover:
                g = self._crossover(elites[0]["genome"], elites[1]["genome"])
            else:
                g = self._mutate(elites[0]["genome"])
            if self._key(g) not in self.tried:
                self.tried.add(self._key(g))
                return g
        # search space nearly exhausted -> force a fresh random genome
        g = self._random()
        self.tried.add(self._key(g))
        return g

    def observe(self, genome, score):
        self.tried.add(self._key(genome))
