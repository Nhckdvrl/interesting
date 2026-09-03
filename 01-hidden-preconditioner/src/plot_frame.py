"""The main figure: one gauge orbit, three optimizers.

Left  -- tuned loss against the frame coordinate, for SGD, Muon and AdamW.
         Every point shares B A, P = s^2 A^T A and all nine gauge invariants;
         only the frame differs.
Right -- the effect against the size of the symmetry quotient, with the
         training-free reach prediction overlaid.
"""
import glob, json, math, os, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
FLOOR = 2e-4          # measured reproducibility floor of a 300-step run


def main(out=None):
    SO = json.load(open(os.path.join(RES, "second_order.json")))
    D = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RES, "frame", "*.json")):
        r = json.load(open(f)); a = r["args"]
        if a["seed"] != 0:
            continue
        D[(a.get("optimizer", "adamw"), a["cond"])][a["lr"]] = \
            r["log"]["eval_loss"][-1]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.1))
    LAD = ["frame0", "frame0.25", "frame0.5", "frame0.75", "frame1"]
    style = (("adamw", "AdamW  (diagonal)", "#c0392b", "o", "-"),
             ("lion", "Lion  (sign, diagonal)", "#e67e22", "v", "-"),
             ("muon", "Muon  (spectral norm)", "#2980b9", "s", "--"),
             ("matprec", "matrix-precond Adam", "#8e44ad", "D", "--"),
             ("sgd", "SGD  (Frobenius norm)", "#27ae60", "^", ":"))
    for opt, lab, col, mk, ls in style:
        pts = []
        for c in LAD:
            cur = D.get((opt, c))
            so = SO.get(f"lit:{c}|trace")
            if cur and so:
                pts.append((so["Lam1"], min(cur.values())))
        if len(pts) < 2:
            continue
        pts.sort()
        base = min(p[1] for p in pts)
        ax.plot([p[0] for p in pts], [(p[1] - base) * 1e3 for p in pts],
                marker=mk, color=col, ls=ls, label=lab, lw=1.8, ms=6)
    ax.axhspan(-FLOOR * 1e3, FLOOR * 1e3, color="0.85", zorder=0)
    ax.text(0.42, FLOOR * 1e3 * 1.4, "reproducibility floor", fontsize=7.5,
            color="0.4")
    ax.set_xlabel(r"frame coordinate  $\Lambda_1$"
                  "\n(eigenframe of $M_g$  $\\rightarrow$  flat diagonal)")
    ax.set_ylabel("tuned eval loss above the best frame  (millinats)")
    ax.set_title("One gauge orbit: $BA$, $P$ and all nine invariants fixed",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # The reach law was falsified at r = 128 (PREDICTIONS_r128.md), so plotting
    # it would show a fitted line with its own counterexample sitting 2
    # millinats below it.  This panel carries a live claim instead: the same
    # ladder on three model families, each against ITS OWN floor measured from
    # SGD, which is exactly gauge-covariant.
    FAMS = [("Qwen3-0.6B", "frame", 1.0e-05), ("Llama-3.2-3B", "llama_fp32", 2.6e-04),
            ("OLMo-2-1B", "olmo", 2.6e-08)]
    xs, names = [], []
    for i, (fam, tag, fl) in enumerate(FAMS):
        E = collections.defaultdict(dict)
        base = None
        for f in glob.glob(os.path.join(RES, tag, "*.json")):
            r = json.load(open(f)); a = r["args"]
            if a.get("seed", 0) or a.get("r", 16) != 16:
                continue
            base = r["base_eval_loss"]
            E[(a.get("optimizer", "adamw"), a["cond"])][a["lr"]] = \
                r["log"]["eval_loss"][-1]
        vs = []
        for c in ("kaiming", "frame0", "frame1"):
            cur = {l: v for l, v in E.get(("adamw", c), {}).items() if v < base}
            if len(cur) >= 2:
                vs.append(min(cur.values()))
        if len(vs) < 2:
            continue
        xs.append(((max(vs) - min(vs)) * 1e3, fl * 1e3)); names.append(fam)
    w = 0.34
    bx.bar([i - w/2 for i in range(len(xs))], [a for a, _ in xs], w,
           color="#c0392b", label="AdamW spread")
    bx.bar([i + w/2 for i in range(len(xs))], [b for _, b in xs], w,
           color="#27ae60", label="SGD floor (this panel)")
    bx.set_yscale("log")
    bx.set_xticks(range(len(xs)))
    bx.set_xticklabels(names, fontsize=8.5)
    for i, (a, b) in enumerate(xs):
        bx.annotate(f"{a/b:.0f}x", (i, max(a, b)), ha="center",
                    textcoords="offset points", xytext=(0, 4), fontsize=9)
    bx.set_ylabel("millinats  (log scale)")
    bx.set_title("Each family against its own floor\n"
                 "(SGD is exactly gauge-covariant, so its spread IS the floor)",
                 fontsize=10)
    bx.legend(fontsize=8, frameon=False)
    bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = out or os.path.join(REPO, "paper", "fig_frame.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
