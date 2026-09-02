"""The literature figure.

Left  -- where 23 published initializer configurations sit on the frame
         coordinate, measured with no training.  Nobody reports this axis.
Right -- what happens when each is rotated along its own gauge orbit, which
         preserves B A, P and all nine invariants exactly.
"""
import json, math, os, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
FLOOR = 2e-4
DATA_AWARE = {"lora_one", "gradsub", "pissa", "pissa_minor", "eva", "olora"}
# our own controls and constructions, not published methods -- excluded from a
# figure whose whole point is what the LITERATURE does without noticing
OURS = {"left_gauge", "geomspec_flatdiag0.5", "flatspec_flatdiag",
        "kaimingspec_flatdiag"}
MT = {"lora_one": "none", "pissa": "none"}


def main(out=None):
    SO = json.load(open(os.path.join(RES, "second_order.json")))
    L = {}
    for k, v in SO.items():
        if not k.startswith("lit:") or "@" in k or k[4:].startswith("frame"):
            continue
        name = k[4:].split("|")[0]
        if name in OURS:
            continue
        L.setdefault(name, v["Lam1"])
    order = sorted(L, key=L.get)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    cols = ["#c0392b" if n in DATA_AWARE else "#2980b9" for n in order]
    ax.barh(range(len(order)), [L[n] for n in order], color=cols, height=0.68)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel(r"frame coordinate $\Lambda_1$, measured without training")
    n_cfg = len({k for k in SO if k.startswith("lit:") and "@" not in k
                 and not k[4:].startswith("frame")
                 and k[4:].split("|")[0] not in OURS})
    ax.set_title(f"{len(order)} published initializers ({n_cfg} configurations)"
                 f" span {max(L.values())/min(L.values()):.1f}$\\times$\n"
                 f"on an axis none of them reports", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c0392b", label="data-aware"),
                       Patch(color="#2980b9", label="frame-based / random")],
              fontsize=8, frameon=False, loc="lower right")

    S = json.load(open(os.path.join(RES, "rot_summary.json")))
    ms = [m for m in ("kaiming", "bimi", "pissa", "eva", "gradsub", "lora_one")
          if f"{m}|published" in S]
    x = range(len(ms))
    d0 = [(S[f"{m}|@frame0"]["L"] - S[f"{m}|published"]["L"]) * 1e3
          for m in ms]
    d1 = [(S[f"{m}|@frame1"]["L"] - S[f"{m}|published"]["L"]) * 1e3
          for m in ms]
    bx.bar([i - 0.19 for i in x], d0, 0.36, color="#27ae60",
           label=r"$\rightarrow$ eigenframe of $M_g$")
    bx.bar([i + 0.19 for i in x], d1, 0.36, color="#c0392b",
           label=r"$\rightarrow$ flat $M_g$ diagonal")
    bx.axhspan(-FLOOR * 1e3, FLOOR * 1e3, color="0.85", zorder=0)
    bx.axhline(0, color="0.4", lw=0.8)
    bx.set_xticks(list(x)); bx.set_xticklabels(ms, fontsize=8.5)
    bx.set_ylabel("change in tuned eval loss  (millinats)\n"
                  "negative = the rotation helps")
    bx.set_title("Rotating each initializer along its own gauge orbit\n"
                 "(6/6 helped or neutral one way, 6/6 hurt or neutral the "
                 "other)", fontsize=10)
    bx.legend(fontsize=8, frameon=False)
    bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = out or os.path.join(REPO, "paper", "fig_zoo.png")
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
