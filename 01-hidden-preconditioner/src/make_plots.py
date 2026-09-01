"""Topic 01 -- figures."""
import glob, json, math, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
PL = os.path.join(REPO, "01-hidden-preconditioner", "plots")
os.makedirs(PL, exist_ok=True)
ZEROB = {"kaiming", "left_gauge", "nora", "nora_unit", "etf",
         "flatspec_flatdiag", "eva", "gradsub"}


def load(tag, key="cond"):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]; ps = list(r["init_pstats"].values())
        rows.append(dict(cond=a["cond"], lr=a["lr"], seed=a["seed"],
                         match=a.get("match", "trace"),
                         gs=a.get("gauge_seed", 0),
                         r_eff=st.mean(p["eff_rank"] for p in ps),
                         act=st.mean(p.get("rel_tr_act", 1) for p in ps),
                         B0=st.mean(p.get("B0_norm", 0) for p in ps),
                         loss=r["log"]["final_eval_loss"]))
    return rows


# ---------------------------------------------------------------- fig 1
def fig_reff():
    rows = []
    for tag in ("g1c",):
        for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
            r = json.load(open(f)); a = r["args"]
            ps = list(r["init_pstats"].values())
            rows.append((st.mean(p["eff_rank"] for p in ps), a["lr"],
                         r["log"]["final_eval_loss"], a["cond"]))
    gc = json.load(open(os.path.join(RES, "grad_capture.json")))
    conds = sorted({x[3] for x in rows},
                   key=lambda c: -st.mean([x[0] for x in rows if x[3] == c]))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    # (a) cos law
    xs = [gc[c]["r_eff"] for c in gc]
    ax[0].plot(xs, [gc[c]["cos_sgd"] for c in gc], "o", label=r"measured $\cos(G,GP)$")
    grid = sorted(xs)
    ax[0].plot(grid, [0.0281 * math.sqrt(x) for x in grid], "-",
               label=r"$0.0281\sqrt{r_{\rm eff}}$")
    ax[0].plot(xs, [gc[c]["cos_adam"] for c in gc], "s",
               label=r"measured $\cos(G,\Delta W_1^{\rm Adam})$")
    ax[0].set_xlabel(r"$r_{\rm eff}(P)$"); ax[0].set_ylabel("first-step efficiency")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[0].set_title("(a) the first-order law, real gradients")
    # (b) dose response
    for c in conds:
        pts = sorted([(x[1], x[2]) for x in rows if x[3] == c])
        ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, lw=1,
                   label=f"{st.mean([x[0] for x in rows if x[3]==c]):.1f}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("learning rate")
    ax[1].set_ylabel("final eval loss"); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=6, title=r"$r_{\rm eff}$", ncol=2)
    ax[1].set_title("(b) matched trace + flat diagonal")
    # (c) best-tuned vs 1/sqrt(r_eff)
    pts = []
    for c in conds:
        cr = [x for x in rows if x[3] == c]
        pts.append((st.mean(x[0] for x in cr), min(x[2] for x in cr)))
    ax[2].plot([1 / math.sqrt(p[0]) for p in pts], [p[1] for p in pts], "o")
    ax[2].set_xlabel(r"$1/\sqrt{r_{\rm eff}(P)}$")
    ax[2].set_ylabel("best-tuned eval loss"); ax[2].grid(alpha=.3)
    ax[2].set_title("(c) $r=+0.95$")
    plt.tight_layout(); p = os.path.join(PL, "reff_law.png")
    plt.savefig(p, dpi=150); print("wrote", p)


# ---------------------------------------------------------------- fig 2
def fig_audit(tag="lit"):
    rows = load(tag)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, match in zip(axes, ("trace", "trace_act")):
        sub = [r for r in rows if r["match"] == match]
        conds = sorted({r["cond"] for r in sub})
        for c in conds:
            cr = sorted([r for r in sub if r["cond"] == c], key=lambda r: r["lr"])
            if not cr:
                continue
            byl = {}
            for r in cr:
                byl.setdefault(r["lr"], []).append(r["loss"])
            xs = sorted(byl); ys = [st.mean(byl[x]) for x in xs]
            style = "o-" if c in ZEROB else "s--"
            ax.plot(xs, ys, style, ms=3.5, lw=1.1, label=c, alpha=.85)
        ax.set_xscale("log"); ax.set_ylim(0.44, 0.50)
        ax.set_xlabel("learning rate"); ax.set_ylabel("final eval loss")
        ax.grid(alpha=.3); ax.set_title(f"matching: {match}")
    axes[1].legend(fontsize=6, ncol=2)
    plt.tight_layout(); p = os.path.join(PL, f"audit_{tag}.png")
    plt.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    fig_reff()
    for t in sys.argv[1:] or ["lit"]:
        fig_audit(t)
