"""Does the fourth coordinate close the out-of-distribution gap?

Wave 1 fitted (S, D, rho) on synthetic constructions and mispredicted the tuned
loss of the 13 published initializers by a systematic +0.005 nats.  Wave 3
causally confirmed W.  The question this script answers is the one that matters:

    is W the piece that was missing?

The law is fitted on synthetic atlas points ONLY.  Published initializers are
never touched by the fit.  They are split into the B_0 = 0 family, which the
theory covers, and the B_0 != 0 family, which is expected to remain outside it.

Two further refinements the earlier analysis lacked:
  * lr* is obtained by a quadratic interpolation in log-lr around the grid
    minimum, so it is continuous rather than snapped to a grid point;
  * the theoretical exponent log lr* = a - (1/2) log S is tested against the
    fitted one, because a theoretical constant is worth more than a fitted
    exponent if it predicts as well.
"""
import glob, json, math, os, statistics as st, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_atlas import atlas_points, ood_points

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NULL = 2.7e-4
NONZERO_B = ("pissa", "pissa_minor", "olora", "lora_one")


def refine(curve):
    """Continuous (lr*, L*) by a quadratic through the grid minimum and its
    neighbours in log-lr."""
    lrs = sorted(curve)
    ys = [curve[l] for l in lrs]
    i = int(np.argmin(ys))
    if i == 0 or i == len(lrs) - 1:
        return lrs[i], ys[i], False
    x = np.log([lrs[i - 1], lrs[i], lrs[i + 1]])
    y = np.array([ys[i - 1], ys[i], ys[i + 1]])
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return lrs[i], ys[i], True
    xs = -c[1] / (2 * c[0])
    xs = min(max(xs, x[0]), x[2])
    return float(np.exp(xs)), float(np.polyval(c, xs)), True


def feats(S, D, W):
    lw = math.log(max(W, 1e-9))
    return [1.0, 1.0 / math.sqrt(max(D, 1e-9)), lw, lw * lw]


FN = ["1", "1/sqrt(D)", "log w", "(log w)^2"]


def main(atlas_tag="atlas", ood_tag="ood"):
    A = atlas_points(atlas_tag)
    rows = []
    for p in A.values():
        lr, L, ok = refine(p["curve"])
        if ok:
            rows.append(dict(S=p["S"], D=p["D"], W=p["W"], lr=lr, L=L))
    print(f"# Law fitted on {len(rows)} SYNTHETIC atlas points only\n")

    # --- learning-rate law: theory vs fit
    ls, ll = np.array([math.log(r["S"]) for r in rows]), \
        np.array([math.log(r["lr"]) for r in rows])
    a_fit = np.polyfit(ls, ll, 1)
    a_half = float(np.mean(ll + 0.5 * ls))
    r_fit = ll - np.polyval(a_fit, ls)
    r_half = ll - (a_half - 0.5 * ls)
    print(f"  log lr* = a + b log S :  fitted b = {a_fit[0]:+.3f}, "
          f"rms {float(np.sqrt(np.mean(r_fit**2))):.3f}")
    print(f"  theory  b = -1/2      :  rms {float(np.sqrt(np.mean(r_half**2))):.3f}"
          f"   -> {'theory is as good' if np.sqrt(np.mean(r_half**2)) < 1.15*np.sqrt(np.mean(r_fit**2)) else 'fitted is better'}\n")

    # --- tuned-loss law on (D, W), at matched S the S-dependence is weak
    X = np.array([feats(r["S"], r["D"], r["W"]) for r in rows])
    y = np.array([r["L"] for r in rows])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"  L* = " + " + ".join(f"{v:+.5f}·{n}" for v, n in zip(c, FN)))
    lo = np.array([X[i] @ np.linalg.lstsq(np.delete(X, i, 0),
                                          np.delete(y, i), rcond=None)[0]
                   for i in range(len(rows))])
    print(f"  atlas LOO rms = {float(np.sqrt(np.mean((y-lo)**2))):.5f} nats\n")

    O = ood_points(ood_tag)
    for group, keep in (("B_0 = 0  (the regime the theory covers)",
                         lambda k: not k.split("|")[0] in NONZERO_B),
                        ("B_0 != 0  (expected to lie outside)",
                         lambda k: k.split("|")[0] in NONZERO_B)):
        sel = {k: v for k, v in O.items() if keep(k)}
        print(f"## {group}   ({len(sel)} held-out initializers)\n")
        print(f"  {'initializer':28s} {'S':>8s} {'D':>6s} {'w':>7s} "
              f"{'L* obs':>9s} {'pred':>9s} {'err':>9s} | "
              f"{'lr* obs':>9s} {'lr* pred':>9s} {'ratio':>6s}")
        errs, lrr = [], []
        for k in sorted(sel):
            o = sel[k]
            lr, L, ok = refine(o["curve"])
            if not ok:
                continue
            pL = float(np.array(feats(o["S_rel"], o["D"], o["W"])) @ c)
            plr = math.exp(a_half - 0.5 * math.log(max(o["S_rel"], 1e-9)))
            errs.append(L - pL); lrr.append(lr / plr)
            print(f"  {k:28s} {o['S_rel']:8.2f} {o['D']:6.2f} {o['W']:7.3f} "
                  f"{L:9.5f} {pL:9.5f} {L-pL:+9.5f} | {lr:9.1e} {plr:9.1e} "
                  f"{lr/plr:6.2f}")
        if errs:
            rms = float(np.sqrt(np.mean(np.square(errs))))
            print(f"\n  OOD rms on L*        = {rms:.5f} nats "
                  f"({rms/NULL:.1f}x null)")
            print(f"  systematic offset    = {st.mean(errs):+.5f} nats "
                  f"({sum(1 for e in errs if e > 0)}/{len(errs)} positive)")
            print(f"  wave-1 comparison    = +0.00500 offset, 12/13 positive")
            print(f"  lr* median |ratio|   = "
                  f"{math.exp(float(np.median(np.abs(np.log(lrr))))):.2f}x "
                  f"(theory exponent -1/2, no fitting)\n")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
