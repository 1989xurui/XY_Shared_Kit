"""Quantum circuit routing task: minimise SWAP-network depth on a coupling graph.

LOCAL HEURISTIC ROUTING PROXY ONLY -- NOT the Hyra official quantum acceptance
(which would require qiskit / a real simulator / hardware).

Coupling graph: a ring of N=5 qubits (edges 0-1,1-2,2-3,3-4,4-0). A fixed
benchmark circuit (data/circuit.json) of 2-qubit gates must be mapped onto the
graph. The genome chooses a routing STRATEGY:
  - route_closer : pick the *shorter* of the two ring paths (fewer SWAPs) when
                   True; the *longer* path when False (deliberately worse).
  - use_bridge   : use a 3-qubit bridge gate (instead of a SWAP) for distance-2
                   gate pairs -- saves a layer and leaves the middle qubit's
                   mapping undisturbed.
  - order        : gate processing order -- "given" / "nearfirst" (cheap gates
                   first) / "farfirst".
Depth is computed with proper layer packing (disjoint ops run in parallel).
Score = -depth (smaller depth is better). The unattended loop should discover
the minimal-depth strategy.
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUIT = os.path.join(_HERE, "data", "circuit.json")
WORK_ROOT = os.path.join(_HERE, "solutions")
EB_DIR = os.path.join(_HERE, "eb_hy3")
SANDBOX_TIMEOUT = 120

SOLUTION_TMPL = '''STRATEGY = {strategy}
'''

RUNNER = '''import os, json, math

CIRCUIT = "{circuit}"

def load():
    with open(CIRCUIT) as f:
        d = json.load(f)
    return d["N"], d["gates"]

def adjacent(p1, p2, N):
    return (p1 + 1) % N == p2 or (p2 + 1) % N == p1 or p1 == p2

def paths(p1, p2, N):
    # two ring directions, each a list of SWAP edges (physical node pairs)
    cw, i = [], p1
    while i != p2:
        j = (i + 1) % N; cw.append((i, j)); i = j
    ccw, i = [], p1
    while i != p2:
        j = (i - 1) % N; ccw.append((i, j)); i = j
    return cw, ccw

def route_depth(N, gates, strat):
    # physical -> logical mapping; start identity
    pm = list(range(N))        # pm[phys] = logical
    phys = list(range(N))      # phys[logical] = phys
    busy = [0] * N             # layer until which each physical qubit is occupied
    def place(qubits):
        layer = max(busy[q] for q in qubits) + 1
        for q in qubits:
            busy[q] = layer
        return layer
    def swap(i, j):
        lq, rq = pm[i], pm[j]
        pm[i], pm[j] = rq, lq
        phys[lq], phys[rq] = j, i
    for (q1, q2) in gates:
        p1, p2 = phys[q1], phys[q2]
        if adjacent(p1, p2, N):
            place([p1, p2]); continue
        cw, ccw = paths(p1, p2, N)
        if strat["route_closer"]:
            edges = cw if len(cw) <= len(ccw) else ccw
        else:
            edges = cw if len(cw) >= len(ccw) else ccw
        if len(edges) == 1 and strat["use_bridge"]:
            # bridge gate on the 3 physical nodes along the edge
            a, b = edges[0]
            c = (a + 1) % N if (a + 1) % N != b else (a - 1) % N
            place([a, b, c])
            continue
        for (i, j) in edges[:-1]:
            place([i, j]); swap(i, j)
        # re-read physical positions after swaps
        pa, pb = phys[q1], phys[q2]
        place([pa, pb])
    return max(busy)

N, gates = load()
import solution
strat = solution.STRATEGY
depth = route_depth(N, gates, strat)
print(json.dumps({{
    "score": -float(depth),
    "depth": depth,
    "n_gates": len(gates),
    "strategy": strat,
}}))
'''


class QuantumRoutingTask:
    def __init__(self):
        with open(CIRCUIT) as f:
            d = json.load(f)
        self.N = d["N"]
        self.gates = d["gates"]

    def workdir(self, i):
        d = os.path.join(WORK_ROOT, f"iter_{i}")
        os.makedirs(d, exist_ok=True)
        return d

    # ---- genome interface (auto_loop) ----
    def genome_space(self):
        return {
            "route_closer": [False, True],
            "use_bridge": [False, True],
            "order": ["given", "nearfirst", "farfirst"],
        }

    def seed_genome(self):
        # deliberately weak: longer path, no bridge, fixed given order
        return {"route_closer": False, "use_bridge": False, "order": "given"}

    def render(self, genome):
        if isinstance(genome, dict):
            strat = {
                "route_closer": bool(genome.get("route_closer", False)),
                "use_bridge": bool(genome.get("use_bridge", False)),
                "order": genome.get("order", "given"),
            }
        else:
            strat = {"route_closer": False, "use_bridge": False, "order": "given"}
        code = SOLUTION_TMPL.format(strategy=strat)
        summary = "route_closer=%s use_bridge=%s order=%s" % (
            strat["route_closer"], strat["use_bridge"], strat["order"])
        return code, summary

    def runner_code(self):
        return RUNNER.format(circuit=CIRCUIT.replace("\\", "/"))

    def parse_run(self, run):
        if not run.get("ok"):
            return {"ok": False, "score": None, "stderr": run.get("stderr")}
        lines = [l for l in run["stdout"].splitlines() if l.strip()]
        if not lines:
            return {"ok": False, "score": None, "stderr": "empty stdout"}
        try:
            obj = json.loads(lines[-1])
            return {
                "ok": True,
                "score": obj.get("score"),
                "depth": obj.get("depth"),
                "n_gates": obj.get("n_gates"),
            }
        except Exception:
            return {"ok": False, "score": None, "stderr": run["stdout"][-200:]}
