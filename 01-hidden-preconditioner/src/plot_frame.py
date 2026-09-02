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
    style = (("adamw", "AdamW  (elementwise max norm)", "#c0392b", "o", "-"),
             ("muon", "Muon  (spectral norm)", "#2980b9", "s", "--"),
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

    R = json.load(open(os.path.join(RES, "frame_reach.json")))
    E = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RES, "rank", "*.json")):
        r = json.load(open(f)); a = r["args"]
        E[(a["r"], a["cond"])][a["lr"]] = r["log"]["eval_loss"][-1]
    xs, ys, rs = [], [], []
    for rk in sorted({k[0] for k in E}):
        a0, a1 = E.get((rk, "frame0")), E.get((rk, "frame1"))
        if not (a0 and a1 and 1e-4 in a0 and 1e-4 in a1):
            continue
        xs.append(math.log(R[str(rk)]["reach"]))
        ys.append((a1[1e-4] - a0[1e-4]) * 1e3)
        rs.append(rk)
    if xs:
        sl = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        gx = [0, max(xs) * 1.15]
        bx.plot(gx, [sl * x for x in gx], color="0.55", lw=1.2, ls="--",
                label=f"$0.00184\\,\\log\\,$reach   ($R^2$ = 0.975)")
        bx.scatter(xs, ys, s=52, color="#c0392b", zorder=3)
        for x, y, rk in zip(xs, ys, rs):
            bx.annotate(f"r={rk}", (x, y), textcoords="offset points",
                        xytext=(7, -3), fontsize=8)
    bx.axhline(0, color="0.8", lw=0.8)
    bx.set_xlabel(r"$\log$ of the training-free frame reach"
                  "\n(one probe pass, no training)")
    bx.set_ylabel("frame1 $-$ frame0 at lr = 1e-4  (millinats)")
    bx.set_title(r"Zero when $O(r)$ is Adam's own symmetry group",
                 fontsize=10)
    bx.legend(fontsize=8, frameon=False, loc="upper left")
    bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = out or os.path.join(REPO, "paper", "fig_frame.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
