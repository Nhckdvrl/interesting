"""Which coordinates are minimally sufficient?

We do not assume the answer.  Candidate coordinate sets are scored by
leave-one-out cross-validation *on the atlas alone*, so the comparison is
honest about overfitting with 18 designed points.

Candidates, all measurable at initialisation with no training:
    S     tr(A Sigma A^T)                  data-space scale
    D     r_eff(A Sigma A^T)               spectral dimension, activation metric
    rho   captured whitened-gradient energy
    D_g   r_eff(A C_g A^T)                 spectral dimension, GRADIENT metric
    Cdis  r_eff of the captured-energy distribution over the T eigenmodes
    cos1  cos(G, GP), the measured first-order descent efficiency
"""
import glob, json, math, os, statistics as st, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_atlas import atlas_points

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")


def join(tag="atlas"):
    A = atlas_points(tag)
    E = json.load(open(os.path.join(RES, "extra_coords.json")))
    ekey = {(round(v["S"], 2), round(v["D"], 2), round(v["rho"], 2)): v
            for v in E.values()}
    out = []
    for p in A.values():
        if not p["bracketed"]:
            continue
        k = (round(p["S"], 2), round(p["D"], 2), round(p["rho"], 2))
        e = ekey.get(k)
        if e is None:                      # nearest match on (S, D, rho)
            e = min(E.values(), key=lambda v: (math.log(v["S"]/p["S"])**2 +
                                               math.log(v["D"]/p["D"])**2 +
                                               math.log(max(v["rho"],1e-3) /
                                                        max(p["rho"],1e-3))**2))
        out.append(dict(S=p["S"], D=p["D"], rho=p["rho"], D_g=e["D_g"],
                        Cdis=e["Cdis"], cos1=e["cos1"],
                        L=p["L_star"], lr=math.log(p["lr_star"])))
    return out


TR = {"S": lambda r: math.log(r["S"]),
      "S2": lambda r: math.log(r["S"]) ** 2,
      "D": lambda r: math.log(r["D"]),
      "invsqrtD": lambda r: 1 / math.sqrt(r["D"]),
      "rho": lambda r: math.log(max(r["rho"], 1e-3)),
      "D_g": lambda r: math.log(r["D_g"]),
      "invsqrtDg": lambda r: 1 / math.sqrt(r["D_g"]),
      "Cdis": lambda r: math.log(r["Cdis"]),
      "cos1": lambda r: math.log(r["cos1"]),
      }

SETS = [
    ("S", ["S", "S2"]),
    ("S + D", ["S", "S2", "invsqrtD"]),
    ("S + D + rho", ["S", "S2", "invsqrtD", "rho"]),
    ("S + D_g", ["S", "S2", "invsqrtDg"]),
    ("S + D + D_g", ["S", "S2", "invsqrtD", "invsqrtDg"]),
    ("S + D + rho + D_g", ["S", "S2", "invsqrtD", "rho", "invsqrtDg"]),
    ("S + Cdis", ["S", "S2", "Cdis"]),
    ("S + D + Cdis", ["S", "S2", "invsqrtD", "Cdis"]),
    ("S + cos1", ["S", "S2", "cos1"]),
    ("S + D + cos1", ["S", "S2", "invsqrtD", "cos1"]),
]


def loo(rows, names, target):
    X = np.array([[1.0] + [TR[n](r) for n in names] for r in rows])
    y = np.array([r[target] for r in rows])
    errs = []
    for i in range(len(rows)):
        m = np.ones(len(rows), bool); m[i] = False
        c, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        errs.append(y[i] - float(X[i] @ c))
    errs = np.array(errs)
    return float(np.sqrt(np.mean(errs ** 2))), 1 - np.var(errs) / np.var(y)


def main(tag="atlas"):
    rows = join(tag)
    print(f"{len(rows)} bracketed atlas points\n")
    for target, unit in (("L", "nats"), ("lr", "log lr")):
        y = [r[target] for r in rows]
        print(f"## target = {target}   (spread {max(y)-min(y):.4f} {unit}, "
              f"sd {st.pstdev(y):.4f})\n")
        print(f"  {'coordinate set':24s} {'k':>2s} {'LOO rms':>10s} "
              f"{'LOO R^2':>9s}")
        base = float(np.std(y))
        print(f"  {'(predict the mean)':24s} {0:2d} {base:10.5f} "
              f"{0.0:9.3f}")
        best = None
        for nm, names in SETS:
            if len(names) + 1 >= len(rows):
                continue
            rms, r2 = loo(rows, names, target)
            flag = ""
            if best is None or rms < best[1]:
                best = (nm, rms); flag = ""
            print(f"  {nm:24s} {len(names):2d} {rms:10.5f} {r2:9.3f}{flag}")
        print(f"\n  best: {best[0]}  (LOO rms {best[1]:.5f})\n")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
