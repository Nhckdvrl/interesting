"""Does the atlas collapse onto a master learning-rate curve?

Wave-1 model selection says the optimal learning rate is predicted by the
data-space scale S alone, and that conditional on the learning rate the rest of
the intrinsic space is nearly flat.  The sharp version of that statement is a
collapse: if S acts purely as a reparameterisation of the learning rate, then
plotting every point's loss against

    eta_eff = eta * S^p

for a single exponent p should put all of them on one curve.

The first-order theory predicts p = 1/2, because the size of the function change
made by the adapter scales as eta * sqrt(tr(A Sigma A^T)).  We fit p rather than
assume it.
"""
import json, math, os, sys, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_atlas import atlas_points, ood_points

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PL = os.path.join(REPO, "01-hidden-preconditioner", "plots")


def spread_at(pts, p, lmax=0.50):
    """Total vertical scatter of all curves after rescaling eta -> eta * S^p,
    measured as the rms residual to a common smooth master curve."""
    xs, ys = [], []
    for q in pts:
        for lr, L in q["curve"].items():
            if L > lmax:          # exclude the divergent tail, which carries no
                continue          # information about the master curve
            xs.append(math.log(lr) + p * math.log(max(q["S"], 1e-9)))
            ys.append(L)
    if len(xs) < 8:
        return float("inf"), None
    X = np.array(xs); Y = np.array(ys)
    # master curve = cubic in the rescaled log learning rate
    A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T
    c, *_ = np.linalg.lstsq(A, Y, rcond=None)
    r = Y - A @ c
    return float(np.sqrt(np.mean(r ** 2))), c


def main(atlas_tag="atlas", ood_tag="ood"):
    pts = [p for p in atlas_points(atlas_tag).values()]
    print(f"{len(pts)} atlas points, "
          f"{sum(len(p['curve']) for p in pts)} (point, lr) cells\n")
    best = min(((spread_at(pts, p)[0], p) for p in np.arange(0, 1.01, 0.01)))
    r0, _ = spread_at(pts, 0.0)
    print(f"  rms residual to a common master curve")
    print(f"    p = 0    (no rescaling)      {r0:.5f} nats")
    for p in (0.25, 0.4, 0.5, 0.6):
        print(f"    p = {p:<4}                    {spread_at(pts, p)[0]:.5f}")
    print(f"    p = {best[1]:.2f}  (fitted)            {best[0]:.5f}")
    print(f"\n  collapse factor vs no rescaling: {r0/best[0]:.1f}x")

    # the sharpest form of the same statement: is eta* * S^p constant?
    print(f"\n  sharper form -- is the OPTIMUM location constant after "
          f"rescaling?")
    for p in (0.0, 0.41, 0.5):
        v = [math.log(q["lr_star"]) + p * math.log(max(q["S"], 1e-9))
             for q in pts if q["bracketed"]]
        print(f"    p = {p:<5} sd of log(eta* S^p) = {st.pstdev(v):.4f} "
              f"({math.exp(st.pstdev(v)):.2f}x)")

    p_star = best[1]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for q in pts:
        lrs = sorted(q["curve"])
        ax[0].plot(lrs, [q["curve"][l] for l in lrs], "o-", ms=3, lw=.9,
                   alpha=.7)
        ax[1].plot([l * q["S"] ** p_star for l in lrs],
                   [q["curve"][l] for l in lrs], "o-", ms=3, lw=.9, alpha=.7)
    try:
        O = ood_points(ood_tag)
    except Exception:
        O = {}
    for k, o in O.items():
        lrs = sorted(o["curve"])
        ax[1].plot([l * o["S_rel"] ** p_star for l in lrs],
                   [o["curve"][l] for l in lrs], "k^--", ms=3.5, lw=.8,
                   alpha=.55)
    for a_, t in ((ax[0], "raw learning rate"),
                  (a_ := ax[1], rf"$\eta\,S^{{{p_star:.2f}}}$")):
        a_.set_xscale("log"); a_.set_ylim(0.435, 0.52); a_.grid(alpha=.3)
        a_.set_xlabel(t); a_.set_ylabel("eval loss")
    ax[0].set_title("(a) 18 intrinsic-state points")
    ax[1].set_title(f"(b) collapsed  ({r0/best[0]:.1f}x tighter)"
                    + ("; ▲ = held-out published initializers" if O else ""))
    plt.tight_layout()
    os.makedirs(PL, exist_ok=True)
    f = os.path.join(PL, "master_curve.png")
    plt.savefig(f, dpi=150)
    print("wrote", f)
    if O:
        ro, _ = spread_at([dict(S=o["S_rel"], curve=o["curve"]) for o in O.values()],
                          p_star)
        print(f"\n  held-out published initializers, same exponent: "
              f"rms {ro:.5f} nats  ({len(O)} methods)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
