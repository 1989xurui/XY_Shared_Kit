import os, sys, json, csv, random
import importlib.util
spec = importlib.util.spec_from_file_location("solution", os.path.join(os.getcwd(), "solution.py"))
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
DATA = 'C:\\Users\\ZhuanZ\\.workbuddy\\skills\\hyra-discovery\\harness\\tasks\\drugcombo\\data_real\\dataset.csv'

def load():
    feats=[]; rows=[]
    with open(DATA) as f:
        r=csv.DictReader(f)
        feats=[c for c in r.fieldnames if c not in ("id","synergy","split")]
        for row in r:
            x=[float(row[c]) for c in feats]
            rows.append((x, float(row["synergy"]), row.get("split","")))
    return feats, rows

def solve(A, b, ridge=1e-6):
    n=len(A); m=len(A[0])
    AtA=[[0.0]*m for _ in range(m)]; Atb=[0.0]*m
    for i in range(n):
        for p in range(m):
            Atb[p]+=A[i][p]*b[i]
            for q in range(m): AtA[p][q]+=A[i][p]*A[i][q]
    for p in range(m): AtA[p][p]+=ridge
    for col in range(m):
        piv=max(range(col,m), key=lambda rr: abs(AtA[rr][col]))
        AtA[col],AtA[piv]=AtA[piv],AtA[col]; Atb[col],Atb[piv]=Atb[piv],Atb[col]
        pv=AtA[col][col] or 1e-12
        for rr in range(col+1,m):
            fct=AtA[rr][col]/pv
            for c in range(col,m): AtA[rr][c]-=fct*AtA[col][c]
            Atb[rr]-=fct*Atb[col]
    x=[0.0]*m
    for col in range(m-1,-1,-1):
        s=Atb[col]
        for rr in range(col+1,m): s-=AtA[col][rr]*x[rr]
        x[col]=s/(AtA[col][col] or 1e-12)
    return x

def expand(x, sel):
    return [x[i] for i in sel]

def r2(yt, yp):
    m=sum(yt)/len(yt)
    sr=sum((a-b)**2 for a,b in zip(yt,yp))
    st=sum((a-m)**2 for a in yt)
    return 1-sr/st if st>0 else 0.0

feats, rows = load()
F = len(feats)
random.seed(0)
test=[r for r in rows if r[2]=="test"]
pool=[r for r in rows if r[2]!="test"]
if not test:
    random.shuffle(rows); k=len(rows)//5; test=rows[:k]; pool=rows[k:]

SELECTED = solution.SELECTED
sel = SELECTED

# K-fold CV on the training pool -> robust, parsimony-aware selection score
K = 5
idxs=list(range(len(pool))); random.shuffle(idxs)
folds=[idxs[p::K] for p in range(K)]
cv_adj=[]
for fidx in range(K):
    te=set(folds[fidx])
    tr=[pool[i] for i in range(len(pool)) if i not in te]
    va=[pool[i] for i in te]
    Xtr=[expand(r[0],sel) for r in tr]; ytr=[r[1] for r in tr]
    Xva=[expand(r[0],sel) for r in va]; yva=[r[1] for r in va]
    coef=solve(Xtr,ytr)
    pred=[sum(c*v for c,v in zip(coef,e)) for e in Xva]
    r2v=r2([r[1] for r in va],pred)
    p=len(sel); nv=len(va)
    adj=(1-(1-r2v)*(nv-1)/max(1,nv-p-1)) if nv>p+1 else r2v
    cv_adj.append(adj)
cv_score=sum(cv_adj)/len(cv_adj)

Xall=[expand(r[0],sel) for r in pool]; yall=[r[1] for r in pool]
COEF=solve(Xall,yall)
tp=[sum(c*v for c,v in zip(COEF,e)) for e in [expand(r[0],sel) for r in test]]
test_metric=r2([r[1] for r in test],tp)

# honest baseline: ALL-features least-squares on the same test set
sel_all=list(range(F))
Xtr_b=[expand(r[0],sel_all) for r in pool]; coef_b=solve(Xtr_b,yall)
tp_b=[sum(c*v for c,v in zip(coef_b,e)) for e in [expand(r[0],sel_all) for r in test]]
baseline_test_r2=r2([r[1] for r in test],tp_b)
beats_baseline = test_metric > baseline_test_r2
score=cv_score

print(json.dumps({
    "score": score,
    "val_r2": cv_score,
    "test_r2": test_metric,
    "cv_adj": cv_score,
    "baseline_test_r2": baseline_test_r2,
    "beats_baseline": beats_baseline,
    "selected": sel,
    "n_sel": len(sel),
    "n_train": len(pool),
    "n_test": len(test)
}))
