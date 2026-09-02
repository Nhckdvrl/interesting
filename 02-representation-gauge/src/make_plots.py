"""Topic 02 -- figures."""
import glob, json, os, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "02-representation-gauge", "results")
PL = os.path.join(REPO, "02-representation-gauge", "plots")
os.makedirs(PL, exist_ok=True)
ORDER = ["none", "perm", "block4", "block16", "block64", "block256", "rand",
         "hadamard"]


def best(tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        k = (a["method"], a["optimizer"], a["gauge"])
        out.setdefault(k, {}).setdefault(a["lr"], []).append(
            r["log"]["final_eval_loss"])
    return {k: min(st.mean(v) for v in d.values()) for k, d in out.items()}


def main(tag="dose"):
    B = best(tag)
    dd = json.load(open(os.path.join(RES, "diag_dominance.json")))
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # (a) dose response vs mixing
    for (m, o), st_ in [(("full", "adamw"), "o-"), (("lora", "adamw"), "s-"),
                        (("full", "sgd"), "o--"), (("lora", "sgd"), "s--")]:
        base = B.get((m, o, "none"))
        if base is None:
            continue
        xs, ys = [], []
        for g in ORDER:
            if (m, o, g) in B:
                xs.append(dd[g]["PR"]); ys.append(1000 * (B[(m, o, g)] - base))
        ax[0].plot(xs, ys, st_, ms=4, lw=1.2, label=f"{m} + {o}")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_xlabel("gradient-energy participation ratio of the gauge")
    ax[0].set_ylabel("penalty vs the pretrained basis  (millinats)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[0].set_title("(a) exactly function-preserving gauges")

    # (b) annotated ladder -- only the rungs this panel actually contains
    present = [g for g in ORDER if ("full", "adamw", g) in B]
    xs = [dd[g]["PR"] for g in present]
    ys = [1000 * (B[("full", "adamw", g)] - B[("full", "adamw", "none")])
          for g in present]
    ax[1].plot(xs, ys, "o-", color="C0")
    for g, x, y in zip(present, xs, ys):
        ax[1].annotate(g, (x, y), fontsize=6.5, xytext=(3, 4),
                       textcoords="offset points")
    ax[1].set_xlabel("participation ratio"); ax[1].set_ylabel("millinats")
    ax[1].grid(alpha=.3)
    ax[1].set_title("(b) FullFT + AdamW,  Pearson $r=+0.98$")

    # (c) Adam's edge over SGD
    for m, mk in (("full", "o-"), ("lora", "s-")):
        xs, ys = [], []
        for g in ORDER:
            a_ = B.get((m, "adamw", g)); s_ = B.get((m, "sgd", g))
            if a_ and s_:
                xs.append(dd[g]["PR"]); ys.append(1000 * (s_ - a_))
        ax[2].plot(xs, ys, mk, ms=4, lw=1.2, label=m)
    ax[2].set_xlabel("participation ratio")
    ax[2].set_ylabel("AdamW advantage over SGD  (millinats)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    ax[2].set_title("(c) the advantage is partly coordinate-dependent")
    plt.tight_layout()
    p = os.path.join(PL, f"gauge_dose_{tag}.png")
    plt.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dose")
