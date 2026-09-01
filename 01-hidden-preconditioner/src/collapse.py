"""Topic 01 -- the magnitude-collapse law.

Claim under test:
    once the realised update scale is accounted for, the final loss does not
    depend on WHICH initialisation produced it.

We plot final eval loss against the realised merged-update norm ||dW||_F
(summed over adapters, measured at the end of training) for every
(condition, lr) cell.  If the claim holds, all conditions lie on a single
curve, and an initialisation's only effect is to move a run along it.

We also test the *a priori* version of the predictor, lr * sqrt(tr P), which
requires no training run at all.
"""
import glob, json, math, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
PLOTS = os.path.join(REPO, "01-hidden-preconditioner", "plots")


def load(tags):
    out = []
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
            r = json.load(open(f))
            ps = list(r["init_pstats"].values())
            a = r["args"]
            s = a["alpha"] / a["r"]
            out.append(dict(
                tag=tag, cond=a["cond"], lr=a["lr"], seed=a["seed"],
                r=a["r"], alpha=a["alpha"],
                trP=st.mean(p["tr_P"] for p in ps) * s * s,
                eff_rank=st.mean(p["eff_rank"] for p in ps),
                diag_imb=st.mean(p["diag_imbalance"] for p in ps),
                dW=r["log"]["diag"][-1]["dW_norm"],
                loss=r["log"]["final_eval_loss"]))
    return out


def interp_curve(ref, xkey):
    """Piecewise-linear interpolant through the reference condition's own
    points, so the reference has exactly zero residual by construction and any
    other condition's residual is a pure between-condition discrepancy."""
    pts = sorted([(r[xkey], r["loss"]) for r in ref])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]

    def f(x):
        if x <= xs[0] or x >= xs[-1]:
            return None
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                t = (x - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] * (1 - t) + ys[i + 1] * t
        return None
    return f


def main(tags):
    rows = load(tags)
    print(f"{len(rows)} runs\n")
    base = [r for r in rows if r["r"] == 16 and r["alpha"] == 32]
    print("=== best-tuned loss per condition (LR swept) " + "=" * 26)
    print(f"  {'condition':22s} {'best loss':>10s} {'at lr':>9s} "
          f"{'||dW||':>8s} {'tr P':>10s} {'r_eff':>7s} {'diag_imb':>9s}")
    for cond in sorted({r["cond"] for r in base}):
        rs = [r for r in base if r["cond"] == cond]
        b = min(rs, key=lambda r: r["loss"])
        print(f"  {cond:22s} {b['loss']:10.5f} {b['lr']:9.1e} {b['dW']:8.2f} "
              f"{b['trP']:10.3f} {b['eff_rank']:7.2f} {b['diag_imb']:9.1e}")
    print()

    ref_rows = [r for r in base if r["cond"] == "kaiming"]
    for xkey, label in [("dW", "realised ||dW||_F"),
                        ("lr", "learning rate")]:
        f = interp_curve(ref_rows, xkey)
        print(f"=== residual vs the kaiming curve, predictor = {label} " + "=" * 12)
        print(f"  {'condition':22s} {'n':>2s} {'max |dev|':>10s} {'rms dev':>10s}")
        for cond in sorted({r["cond"] for r in base}):
            rs = [r for r in base if r["cond"] == cond]
            res = [r["loss"] - f(r[xkey]) for r in rs if f(r[xkey]) is not None]
            if len(res) < 2:
                continue
            rms = (sum(x * x for x in res) / len(res)) ** 0.5
            print(f"  {cond:22s} {len(res):2d} {max(abs(x) for x in res):10.5f} "
                  f"{rms:10.5f}")
        # rank/alpha families
        for (rr, aa) in [(4, 32), (64, 32), (16, 8), (16, 128)]:
            rs = [r for r in rows if r["r"] == rr and r["alpha"] == aa]
            res = [r["loss"] - f(r[xkey]) for r in rs if f(r[xkey]) is not None]
            if len(res) < 2:
                continue
            rms = (sum(x * x for x in res) / len(res)) ** 0.5
            print(f"  {'kaiming r=%d a=%g' % (rr, aa):22s} {len(res):2d} "
                  f"{max(abs(x) for x in res):10.5f} {rms:10.5f}")
        print()

    # plot
    os.makedirs(PLOTS, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, xkey, label in [(axes[0], "lr", "learning rate"),
                            (axes[1], "dW", r"realised $\|\Delta W\|_F$")]:
        for cond in sorted({r["cond"] for r in rows}):
            rs = sorted([r for r in rows if r["cond"] == cond and r["r"] == 16
                         and r["alpha"] == 32], key=lambda r: r[xkey])
            if not rs:
                continue
            ax.plot([r[xkey] for r in rs], [r["loss"] for r in rs], "o-",
                    ms=4, lw=1.2, label=cond, alpha=.85)
        for (rr, aa), mk in [((4, 32), "s"), ((64, 32), "^"), ((16, 8), "v"),
                             ((16, 128), "D")]:
            rs = sorted([r for r in rows if r["r"] == rr and r["alpha"] == aa],
                        key=lambda r: r[xkey])
            if rs:
                ax.plot([r[xkey] for r in rs], [r["loss"] for r in rs],
                        mk + "--", ms=4, lw=.9, alpha=.6,
                        label=f"kaiming r={rr} a={aa:g}")
        ax.set_xscale("log"); ax.set_xlabel(label)
        ax.set_ylabel("final eval loss"); ax.grid(alpha=.3)
        ax.set_ylim(0.435, 0.56)
    axes[1].legend(fontsize=6.5, ncol=2)
    axes[0].set_title("collapse in LR: conditions separate")
    axes[1].set_title("collapse in update magnitude")
    plt.tight_layout()
    p = os.path.join(PLOTS, "magnitude_collapse.png")
    plt.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1:] or ["g1b", "g1b_r4a32", "g1b_r64a32", "g1b_r16a8",
                          "g1b_r16a128"])
