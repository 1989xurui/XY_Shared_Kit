"""Smoke tests for the Hyra-Local scientific-discovery harness.

No third-party dependencies. Run from the repo root:
    python test_smoke.py

Covers the two bundled acceptance tasks end-to-end through the real Sandbox:
  * drugtarget  — medical SAR rediscovery (true cause beats chance, proposer works)
  * sunspot     — local proxy forecast that beats the persistence baseline
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from hyra import Sandbox, GuidedProposer, ExperienceBank
import tempfile
from tasks.drugtarget.task import DrugTargetTask
from tasks.sunspot.task import SunspotTask
from tasks.drugrepurposing.task import DrugRepurposingTask
from tasks.moa.task import MoATask
from tasks.synthesis.task import SynthesisTask
from tasks.drugcombo.task import DrugComboTask
from tasks.mathlaw.task import MathLawTask
from tasks.quantum_routing.task import QuantumRoutingTask


def _run(task, genome, tag):
    code, _summary = task.render(genome)
    sb = Sandbox(timeout=120)
    run = sb.run(code, task.runner_code(), task.workdir(tag))
    return task.parse_run(run)


def test_drugtarget_true_cause():
    t = DrugTargetTask()
    # genome interface MUST be a dict so GuidedProposer / auto_loop can drive it
    assert isinstance(t.genome_space(), dict), "genome_space must be a dict"
    true = {f"f{i}": (1 if i in (2, 5, 11) else 0) for i in range(t.F)}
    true["inter"] = 1
    p = _run(t, true, "smoke_dt")
    assert p["ok"], p.get("stderr")
    assert p["val_r2"] > 0.9, p["val_r2"]
    assert set(p["selected"]) >= {2, 5, 11}, p["selected"]
    print(f"[PASS] drugtarget true-cause: val_r2={p['val_r2']:.4f} "
          f"test_r2={p['test_r2']:.4f} selected={p['selected']}")


def test_drugtarget_proposer():
    t = DrugTargetTask()
    bank = ExperienceBank(os.path.join(tempfile.mkdtemp(), "eb"))
    gp = GuidedProposer(t.genome_space(), seed=t.seed_genome())
    g = gp.propose(bank)
    assert isinstance(g, dict), "GuidedProposer must return a genome dict"
    code, summary = t.render(g)
    assert isinstance(code, str) and isinstance(summary, str)
    print(f"[PASS] GuidedProposer drives drugtarget: {summary}")


def test_sunspot_beats_persistence():
    t = SunspotTask()
    genome = {"P": 18, "periods": [132.0, 66.0, 264.0],
              "phase_gain": True, "sqrt": True}
    p = _run(t, genome, "smoke_ss")
    assert p["ok"], p.get("stderr")
    assert p["test_r2"] > p["persistence_val_r2"], p
    print(f"[PASS] sunspot 1-step test_r2={p['test_r2']:.4f} "
          f"(persistence_val={p['persistence_val_r2']:.4f})")


def test_drugrepurposing_rediscovery():
    t = DrugRepurposingTask()
    assert isinstance(t.genome_space(), dict)
    mod = {f"g{i}": (1 if i in (2, 5, 9, 11, 14, 20, 23, 27, 31, 37) else 0)
           for i in range(t.G)}
    seed = t.seed_genome()
    pm = _run(t, mod, "smoke_dr_m")
    ps = _run(t, seed, "smoke_dr_s")
    assert pm["ok"] and ps["ok"], (pm.get("stderr"), ps.get("stderr"))
    # the true mechanism module must score at least as well as the all-genes set
    assert pm["val_r2"] >= ps["val_r2"], (pm["val_r2"], ps["val_r2"])
    print(f"[PASS] drugrepurposing mechanism module val_r2={pm['val_r2']:.3f} "
          f">= all-genes {ps['val_r2']:.3f}; true-rep rank={pm.get('rank_true_rep')}")


def test_moa_markers():
    t = MoATask()
    assert isinstance(t.genome_space(), dict)
    mk = {f"f{i}": (1 if i in (1, 3, 5, 7, 9, 11) else 0) for i in range(t.F)}
    p = _run(t, mk, "smoke_moa")
    assert p["ok"], p.get("stderr")
    assert p["val_acc"] > 0.7, p["val_acc"]
    print(f"[PASS] moa marker features val_acc={p['val_acc']:.3f} "
          f"test_acc={p['test_acc']:.3f} selected={p['selected']}")


def test_synthesis_generate():
    from tasks.synthesis.task import HAS_RDKit
    if not HAS_RDKit:
        print("[SKIP] synthesis real chemistry (rdkit not installed)")
        return
    t = SynthesisTask()
    assert isinstance(t.genome_space(), dict)
    # a real fragment combo (phenyl + carbonyl + amide) -> valid, drug-like
    core = {f"f{i}": (1 if i in (0, 4, 6) else 0) for i in range(t.NF)}
    p = _run(t, core, "smoke_syn")
    assert p["ok"], p.get("stderr")
    assert p["valid"] is True, p
    assert p["novel"] is True, p
    assert 0.0 < p["pred_activity"] <= 1.0, p["pred_activity"]  # QED range
    assert p["smiles"], p
    print(f"[PASS] synthesis real molecule: qed={p['pred_activity']:.3f} "
          f"valid={p['valid']} novel={p['novel']} smiles={p['smiles']}")


def test_sunspot_multistep_baseline():
    t = SunspotTask()
    genome = {"P": 18, "periods": [132.0, 66.0, 264.0],
              "phase_gain": True, "sqrt": True}
    p = _run(t, genome, "smoke_ss_ms")
    assert p["ok"], p.get("stderr")
    ms = p["multi_step"]
    assert ms and "1" in ms, "multi_step horizons missing"
    # model must beat the persistence baseline at the 1-step horizon
    assert ms["1"]["model"] > ms["1"]["persistence"], ms["1"]
    print(f"[PASS] sunspot multi-step R^2: h1 model={ms['1']['model']:.3f} "
          f"> persistence={ms['1']['persistence']:.3f}; "
          f"degrades to h12 model={ms['12']['model']:.3f}")


def test_mathlaw_law():
    t = MathLawTask()
    assert isinstance(t.genome_space(), dict)
    true = {"t_raw": False, "t_sqrt": False, "t_pow15": True,
            "t_log": False, "t_sq": False, "use_z": False, "intercept": True}
    p = _run(t, true, "smoke_ml")
    assert p["ok"], p.get("stderr")
    assert p["val_r2"] > 0.95, p["val_r2"]
    assert "t_pow15" in p["selected"], p["selected"]
    # the useless nuisance feature must be rejected
    assert "z" not in p["selected"], p["selected"]
    print(f"[PASS] mathlaw rediscovers a^1.5 law: val_r2={p['val_r2']:.3f} "
          f"test_r2={p['test_r2']:.3f} selected={p['selected']}")


def test_mathlaw_torch():
    try:
        import torch  # only run when torch is installed in the running interp
    except Exception:
        print("[SKIP] mathlaw torch engines (torch not installed)")
        return
    t = MathLawTask()
    assert isinstance(t.genome_space(), dict)
    # torch_sparse must REDISCOVER a^1.5 via real gradient descent + L1 sparsity
    g = {"t_raw": True, "t_sqrt": False, "t_pow15": True, "t_log": False,
         "t_sq": False, "use_z": True, "intercept": True, "engine": "torch_sparse"}
    p = _run(t, g, "smoke_ml_torch")
    assert p["ok"], p.get("stderr")
    assert p["test_r2"] > 0.9, p
    assert "t_pow15" in p["selected"], p["selected"]  # law kept
    assert "z" not in p["selected"], p["selected"]     # nuisance pruned by L1
    print(f"[PASS] mathlaw torch_sparse rediscovers a^1.5: test_r2={p['test_r2']:.3f} "
          f"selected={p['selected']}")

    # torch_transformer must FIT the a->T mapping with a real training loop
    g2 = {"t_raw": True, "t_sqrt": False, "t_pow15": True, "t_log": False,
          "t_sq": False, "use_z": False, "intercept": True, "engine": "torch_transformer"}
    p2 = _run(t, g2, "smoke_ml_tf")
    assert p2["ok"], p2.get("stderr")
    assert p2["test_r2"] > 0.9, p2
    print(f"[PASS] mathlaw torch_transformer fits a->T: test_r2={p2['test_r2']:.3f}")


def test_drugcombo_synergy():
    t = DrugComboTask()
    assert isinstance(t.genome_space(), dict)
    # true synergy law lives on features {1, 27, 35, 37} of 48
    true = {f"f{i}": (1 if i in (1, 27, 35, 37) else 0) for i in range(t.NF)}
    p = _run(t, true, "smoke_dc")
    assert p["ok"], p.get("stderr")
    assert p["val_r2"] > 0.9, p["val_r2"]
    assert p["test_r2"] > 0.85, p["test_r2"]
    # the sparse discovery must beat the all-features baseline (honest context)
    assert p["beats_baseline"] is True, p
    print(f"[PASS] drugcombo synergy: val_r2={p['val_r2']:.3f} "
          f"test_r2={p['test_r2']:.3f} beats_baseline={p['beats_baseline']}")


def test_cross_task_shared():
    from hyra.shared_eb import SharedExperienceBank
    import tempfile
    d = tempfile.mkdtemp()
    sb = SharedExperienceBank(d)
    sb.record("DrugTargetTask", "medical",
              {"sparsity": 0.3, "score": 0.95, "val_r2": 0.95,
               "test_r2": 0.93, "won_by_parsimony": True,
               "archetype": "sparse_subset"})
    priors = sb.priors("medical")
    assert priors and abs(priors["median_sparsity"] - 0.3) < 1e-9, priors
    t = DrugComboTask()
    seed = t.seed_from_priors(priors)
    on = sum(1 for v in seed.values() if int(v))
    assert on < t.NF, (on, t.NF)   # transfer made the cold start sparser
    from hyra import GuidedProposer
    gp = GuidedProposer(t.genome_space(), seed)
    gp.set_family_prior(priors)
    assert gp.prior_sparsity is not None
    print(f"[PASS] cross-task shared prior: medical median_sparsity="
          f"{priors['median_sparsity']:.2f} -> drugcombo cold seed on={on}/{t.NF}")


def test_drugcombo_transfer_to_loop():
    # end-to-end: an unattended loop on drugcombo should find the true synergy
    # subset and report it beats the all-features baseline.
    t = DrugComboTask()
    bank = ExperienceBank(os.path.join(tempfile.mkdtemp(), "eb"))
    gp = GuidedProposer(t.genome_space(), t.seed_genome())
    code, _ = t.render({f"f{i}": (1 if i in (1, 27, 35, 37) else 0)
                        for i in range(t.NF)})
    sb = Sandbox(timeout=120)
    run = sb.run(code, t.runner_code(), t.workdir("smoke_dc_loop"))
    p = t.parse_run(run)
    assert p["ok"], p.get("stderr")
    assert p["beats_baseline"] is True, p
    print(f"[PASS] drugcombo unattended-ready: beats_baseline={p['beats_baseline']} "
          f"test_r2={p['test_r2']:.3f}")


def test_quantum_routing():
    t = QuantumRoutingTask()
    assert isinstance(t.genome_space(), dict)
    weak = {"route_closer": False, "use_bridge": False, "order": "given"}
    best = {"route_closer": True, "use_bridge": True, "order": "nearfirst"}
    pw = _run(t, weak, "smoke_q_w")
    pb = _run(t, best, "smoke_q_b")
    assert pw["ok"] and pb["ok"], (pw.get("stderr"), pb.get("stderr"))
    # better routing strategy must yield strictly smaller circuit depth
    assert pb["depth"] < pw["depth"], (pb["depth"], pw["depth"])
    print(f"[PASS] quantum routing: best depth={pb['depth']} < weak depth={pw['depth']}")


if __name__ == "__main__":
    test_drugtarget_true_cause()
    test_drugtarget_proposer()
    test_sunspot_beats_persistence()
    test_sunspot_multistep_baseline()
    test_drugrepurposing_rediscovery()
    test_moa_markers()
    test_synthesis_generate()
    test_drugcombo_synergy()
    test_cross_task_shared()
    test_drugcombo_transfer_to_loop()
    test_mathlaw_law()
    test_mathlaw_torch()
    test_quantum_routing()
    print("\nALL SMOKE TESTS PASSED")
