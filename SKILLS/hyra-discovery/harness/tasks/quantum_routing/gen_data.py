"""Generate a fixed benchmark quantum circuit for the quantum_routing task.

A ring coupling graph of N=5 qubits (edges 0-1,1-2,2-3,3-4,4-0). The benchmark
is a deterministic sequence of 2-qubit gates between random logical-qubit pairs
(seeded for reproducibility). The harness must find a routing strategy that maps
this circuit onto the coupling graph with minimal *circuit depth* (SWAP-network
minimisation -- a real IBM Qiskit-style routing subproblem).

LOCAL HEURISTIC ROUTING PROXY ONLY -- NOT the Hyra official quantum acceptance
(which would require qiskit / real hardware or a high-fidelity simulator).
"""
import os
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "circuit.json")

N = 5
N_GATES = 14
SEED = 20240722


def main():
    rng = random.Random(SEED)
    gates = []
    for _ in range(N_GATES):
        a = rng.randrange(N)
        b = rng.randrange(N)
        while b == a:
            b = rng.randrange(N)
        gates.append([a, b])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"N": N, "gates": gates}, f, indent=2)
    print("wrote %d gates -> %s" % (len(gates), OUT))


if __name__ == "__main__":
    main()
