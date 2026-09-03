"""B: does the gauge frame shift the optimal learning rate, and does the shift
grow with rank?

LoRA's optimal learning rate transfers poorly across ranks -- LoRA-Muon (ICLR
2026) names it as a practical annoyance.  The gauge redundancy of the
factorisation is r(r-1)/2 dimensions, growing quadratically in rank, carries
zero function-space information, and AdamW reads it.  If the frame moves lr*,
and the movement grows with rank, part of that annoyance is gauge.

lr* is located by a quadratic fit through the grid minimum and its neighbours in
log-lr, so it is continuous rather than snapped to a rung -- necessary, because
the effect being looked for is smaller than the grid spacing.
"""
import glob, json, math, os, sys, collections
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")


def refine(curve):
    """Continuous (lr*, L*) by a quadratic through the minimum in log-lr."""
    lrs = sorted(curve)
    ys = [curve[l] for l in lrs]
    i = int(np.argmin(ys))
    if i == 0 or i == len(lrs) - 1:
        return lrs[i], ys[i], False
    x = np.log(np.array([lrs[i - 1], lrs[i], lrs[i + 1]]))
    y = np.array([ys[i - 1], ys[i], ys[i + 1]])
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return lrs[i], ys[i], True
    xs = float(np.clip(-c[1] / (2 * c[0]), x[0], x[2]))
    return float(np.exp(xs)), float(np.polyval(c, xs)), True


def main():
    D = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in glob.glob(os.path.join(RES, "rank", "*.json")) + \
             glob.glob(os.path.join(RES, "frame", "*.json")):
        r = json.load(open(f)); a = r["args"]
        if a.get("optimizer", "adamw") != "adamw" or a.get("seed", 0) != 0:
            continue
        if a["cond"] not in ("kaiming", "frame0", "frame1"):
            continue
        if a["r"] != 16 and abs(a["alpha"] - 2 * a["r"]) > 1e-9:
            continue                       # keep s = alpha/r fixed
        D[a["r"]][a["cond"]][a["lr"]] = r["log"]["eval_loss"][-1]

    print("Optimal learning rate by rank and frame (quadratic in log-lr)\n")
    print(f"{'r':>5s} {'dim gauge':>10s} | " +
          " ".join(f"{c:>11s}" for c in ("kaiming", "frame0", "frame1")) +
          f" | {'kai/frame0':>11s} {'spread':>8s}")
    rows = []
    for rk in sorted(D):
        vals, ok = {}, True
        for c in ("kaiming", "frame0", "frame1"):
            cur = D[rk].get(c)
            if not cur or len(cur) < 3:
                vals[c] = None; ok = False; continue
            lr, L, br = refine(cur)
            vals[c] = (lr, L, br)
            if not br:
                ok = False
        cells = []
        for c in ("kaiming", "frame0", "frame1"):
            v = vals[c]
            cells.append("-" if v is None else
                         f"{v[0]:.2e}{'' if v[2] else '*'}")
        if vals["kaiming"] and vals["frame0"]:
            ratio = vals["kaiming"][0] / vals["frame0"][0]
            lrsall = [v[0] for v in vals.values() if v]
            spread = max(lrsall) / min(lrsall)
            rows.append((rk, ratio, spread, ok))
            rs, ss = f"{ratio:11.2f}", f"{spread:8.2f}"
        else:
            rs, ss = " " * 11, " " * 8
        print(f"{rk:5d} {rk*(rk-1)//2:10d} | " +
              " ".join(f"{x:>11s}" for x in cells) + f" | {rs} {ss}")
    print("\n(* = grid edge, lr* not bracketed -- not usable)")

    good = [(rk, sp) for rk, ra, sp, ok in rows if ok]
    if len(good) >= 3:
        xs = [math.log(max(rk * (rk - 1) // 2, 1)) for rk, _ in good]
        ys = [math.log(sp) for _, sp in good]
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        r_ = sxy / math.sqrt(sxx * syy + 1e-30)
        print(f"\n  lr* spread across frames vs gauge dimension "
              f"({len(good)} usable ranks):")
        print(f"    log-log slope {sxy/sxx:+.3f}, r = {r_:+.3f}")
        print(f"    spread by rank: " +
              ", ".join(f"r={rk}: {sp:.2f}x" for rk, sp in good))
        if max(sp for _, sp in good) < 1.15:
            print("    -> the frame does NOT measurably move lr*; "
                  "the hypothesis fails and we say so.")
    else:
        print(f"\n  only {len(good)} ranks bracketed; no verdict.")


if __name__ == "__main__":
    main()
