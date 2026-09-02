"""Fit the response surface on the synthetic atlas, then predict the held-out
published initializers.

The atlas points are *constructions*, not methods: each one sits at an exactly
specified (S, D, rho).  The published initializers are never used to fit
anything; they are located in the same coordinates by `intrinsic_table.py` and
predicted, which makes this an out-of-distribution test rather than a fit.
"""
import glob, json, math, os, statistics as st, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")


def atlas_points(tag="atlas"):
    """-> {point_key: dict(S, D, rho, curve={lr: loss}, trP, A_fro)}"""
    pts = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        key = (a["S"], a["D"], a["rho"])
        m = list(r["init_stats"].values())
        p = pts.setdefault(key, dict(
            S=st.mean(x["S_rel"] for x in m),
            D=st.mean(x["D"] for x in m),
            rho=st.mean(x["rho_rel"] for x in m),
            trP=st.mean(x["tr_P"] for x in m),
            A_fro=st.mean(x["A_fro"] for x in m),
            curve={}, traj=r.get("traj")))
        p["curve"].setdefault(a["lr"], []).append(r["log"]["final_eval_loss"])
    for p in pts.values():
        p["curve"] = {lr: st.mean(v) for lr, v in p["curve"].items()}
        lrs = sorted(p["curve"])
        p["lr_star"] = min(p["curve"], key=p["curve"].get)
        p["L_star"] = p["curve"][p["lr_star"]]
        p["bracketed"] = p["lr_star"] not in (lrs[0], lrs[-1])
    return pts


def ood_points(tag="ood", coords="intrinsic_table.json"):
    C = json.load(open(os.path.join(RES, coords)))
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        k = f"{a['cond']}|{a.get('match','trace')}"
        if k not in C:
            continue
        o = out.setdefault(k, dict(**{q: C[k][q] for q in
                                      ("S_rel", "D", "rho", "tr_P", "B0")},
                                   curve={}))
        o["curve"].setdefault(a["lr"], []).append(r["log"]["final_eval_loss"])
    for o in out.values():
        o["curve"] = {lr: st.mean(v) for lr, v in o["curve"].items()}
        lrs = sorted(o["curve"])
        o["lr_star"] = min(o["curve"], key=o["curve"].get)
        o["L_star"] = o["curve"][o["lr_star"]]
        o["bracketed"] = o["lr_star"] not in (lrs[0], lrs[-1])
    return out


def feats(S, D, rho):
    lS, lD, lr_ = math.log(max(S, 1e-6)), math.log(max(D, 1e-6)), \
        math.log(max(rho, 1e-3))
    return [1.0, lS, lS * lS, 1.0 / math.sqrt(max(D, 1e-6)), lD, lr_]


FNAMES = ["1", "log S", "(log S)^2", "1/sqrt(D)", "log D", "log rho"]


def fit(X, y):
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return coef


def main(atlas_tag="atlas", ood_tag="ood"):
    A = atlas_points(atlas_tag)
    print(f"# Atlas: {len(A)} points\n")
    print(f"  {'S':>8s} {'D':>7s} {'rho':>8s} {'trP':>9s} | "
          f"{'L*':>9s} {'lr*':>8s} {'ok':>3s}")
    for k in sorted(A, key=lambda k: (k[0], str(k[1]), str(k[2]))):
        p = A[k]
        print(f"  {p['S']:8.3f} {p['D']:7.3f} {p['rho']:8.3f} {p['trP']:9.2f} | "
              f"{p['L_star']:9.5f} {p['lr_star']:8.0e} "
              f"{'y' if p['bracketed'] else 'EDGE':>3s}")

    use = [p for p in A.values() if p["bracketed"]]
    if len(use) < 6:
        print("\n(not enough bracketed atlas points yet)"); return
    X = [feats(p["S"], p["D"], p["rho"]) for p in use]
    yL = [p["L_star"] for p in use]
    ylr = [math.log(p["lr_star"]) for p in use]
    cL, clr = fit(X, yL), fit(X, ylr)
    for nm, c, y in (("L*", cL, yL), ("log lr*", clr, ylr)):
        pred = np.array(X) @ c
        r2 = 1 - np.var(np.array(y) - pred) / np.var(y)
        print(f"\n  in-sample fit of {nm}: R^2 = {r2:.3f}")
        print("    " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(FNAMES, c)))

    try:
        O = ood_points(ood_tag)
    except FileNotFoundError:
        print("\n(no intrinsic_table.json yet -- OOD test pending)"); return
    if not O:
        print("\n(no OOD runs yet)"); return
    print(f"\n# Out-of-distribution: {len(O)} published initializers, "
          f"never used to fit\n")
    print(f"  {'initializer|match':30s} {'S':>8s} {'D':>6s} {'rho':>7s} "
          f"{'L* obs':>9s} {'L* pred':>9s} {'err':>9s} {'lr* obs':>8s} "
          f"{'lr* pred':>9s}")
    errs, errlr = [], []
    for k in sorted(O):
        o = O[k]
        if not o["bracketed"]:
            continue
        x = np.array(feats(o["S_rel"], o["D"], o["rho"]))
        pL, plr = float(x @ cL), math.exp(float(x @ clr))
        errs.append(o["L_star"] - pL)
        errlr.append(math.log(o["lr_star"] / plr))
        print(f"  {k:30s} {o['S_rel']:8.2f} {o['D']:6.2f} {o['rho']:7.2f} "
              f"{o['L_star']:9.5f} {pL:9.5f} {o['L_star']-pL:+9.5f} "
              f"{o['lr_star']:8.0e} {plr:9.1e}")
    if errs:
        rms = float(np.sqrt(np.mean(np.square(errs))))
        base = float(np.std([O[k]["L_star"] for k in O if O[k]["bracketed"]]))
        print(f"\n  OOD rms error on L*      = {rms:.5f} nats "
              f"({rms/2.7e-4:.1f}x the gauge null)")
        print(f"  spread of the OOD targets = {base:.5f} nats "
              f"-> variance explained = {1-(rms/base)**2:.3f}")
        print(f"  OOD median |log lr* ratio| = "
              f"{np.median(np.abs(errlr)):.3f} "
              f"({math.exp(float(np.median(np.abs(errlr)))):.2f}x)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
