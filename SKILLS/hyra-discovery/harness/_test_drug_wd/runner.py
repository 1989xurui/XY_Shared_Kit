import os, sys, json, csv, importlib.util
from collections import Counter
DRUG_CSV = 'C:\\Users\\ZhuanZ\\.workbuddy\\skills\\hyra-discovery\\harness\\tasks\\asd_adhd\\data_real\\drug_candidates.csv'
PENALTY = 0.5

def load():
    d = {}
    if not os.path.exists(DRUG_CSV):
        print(json.dumps({"ok": False, "score": None, "err": "missing drug_candidates.csv"}))
        sys.exit(0)
    with open(DRUG_CSV, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("best_mech_gene_weight", "n_trials", "n_indications", "base_score"):
                try: r[k] = float(r[k])
                except (ValueError, KeyError): r[k] = 0.0
            for k in ("is_approved",):
                r[k] = str(r.get(k, "")).strip().lower() == "true"
            for k in ("clinical_asd",):
                r[k] = int(float(r.get(k, 0) or 0))
            d[r["drug"].strip().upper()] = r
    return d

drugs = load()
spec = importlib.util.spec_from_file_location("solution", os.path.join(os.getcwd(), "solution.py"))
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
SELECTED = [s.strip().upper() for s in getattr(solution, "SELECTED_DRUGS", [])]

def support(sel, drugs):
    total = 0.0
    covered = set(); classes = set(); cnt = Counter()
    kept = []
    for key in sel:
        rec = drugs.get(key)
        if rec is None:
            continue
        s = 0.0
        dg = [g for g in (rec.get("direct_genes") or "").split(";") if g]
        if dg:
            s += 10 + 2 * len(dg)
        s += 2.0 * float(rec.get("best_mech_gene_weight", 0) or 0)
        tag = rec.get("tag", "")
        if tag == "REPURPOSE": s += 4
        elif tag == "MECHANISM": s += 2
        if int(rec.get("clinical_asd", 0) or 0): s += 5
        s += 0.05 * min(float(rec.get("n_trials", 0) or 0), 30)
        total += s
        for g in dg: covered.add(g)
        for g in [g for g in (rec.get("mech_genes") or "").split(";") if g]: covered.add(g)
        for c in [c for c in (rec.get("mech_class") or "").split(";") if c]:
            classes.add(c); cnt[c] += 1
        kept.append(key)
    redundancy = sum(max(0, n - 2) * 3 for n in cnt.values())
    cluster_bonus = 4 * len(classes & {"EPI", "CAL", "RAS", "KIN"})
    score = total - redundancy + cluster_bonus - PENALTY * len(kept)
    return score, total, redundancy, cluster_bonus, sorted(classes), kept

score, total, redun, cbonus, classes, kept = support(SELECTED, drugs)
print(json.dumps({
    "ok": True,
    "score": score,
    "val_r2": score,
    "test_r2": None,
    "raw_total": total,
    "redundancy_penalty": redun,
    "cluster_bonus": cbonus,
    "mech_classes": classes,
    "n_sel": len(kept),
    "selected": kept,
}))
