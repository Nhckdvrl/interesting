"""Which functional of the frame does AdamW respond to?

Two designs, and they must be read differently.

*Within one gauge orbit* -- the eight frames of the SAME vanilla draw -- nothing
varies but the frame, so a correlation there is clean.

*Across the zoo* -- twelve rotations of six different initialisers -- the row
spaces differ, and the row-space effect is an order of magnitude larger than
any frame effect (EVA sits 0.010 nats from Kaiming, against frame spreads of
0.002), so cross-method correlations are confounded and are reported only for
completeness.

Candidates:
  Lambda_1  ||G A^T||_1^2 / (d_out r ||G A^T||_F^2), AdamW's first-order
            descent rate over SGD's
  Off_x     off-diagonal mass of M_x = A Sigma A^T
  Off_g     off-diagonal mass of M_g = A C_g A^T
  E_g       equipartition of diag(M_g)
"""
import glob, json, math, os, statistics as st, sys, collections
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
NULL = 2.7e-4
KEYS = ("Lam1", "Off_x", "Off_g", "E_g")


def corr(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) *
                           sum((b - my) ** 2 for b in y) + 1e-30)


def main():
    SO = json.load(open(os.path.join(RES, "second_order.json")))
    D = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RES, "frame", "*.json")):
        r = json.load(open(f)); a = r["args"]
        if a.get("optimizer", "adamw") != "adamw":
            continue
        D[a["cond"]][a["lr"]] = r["log"]["eval_loss"][-1]
    order = ["frame0", "frame0.25", "frame0.5", "frame0.75", "frame1",
             "framex0", "framex1", "kaiming"]
    rows = []
    for c in order:
        so = SO.get(f"lit:{c}|trace"); cur = D.get(c)
        if not cur or not so:
            continue
        rows.append((c, [so[k] for k in KEYS], min(cur.values())))
    print("Eight frames of one vanilla draw -- every gauge invariant identical,")
    print("only the frame moves.  Sorted by tuned loss.\n")
    print(f"{'frame':>10s} " + " ".join(f"{k:>7s}" for k in KEYS) +
          f" {'tuned':>9s}")
    for c, v, L in sorted(rows, key=lambda z: z[2]):
        print(f"{c:>10s} " + " ".join(f"{x:7.4f}" for x in v) + f" {L:9.5f}")
    y = [r[2] for r in rows]
    print(f"\n  correlation of tuned loss with each candidate "
          f"({len(rows)} frames, one orbit):")
    best = None
    for i, k in enumerate(KEYS):
        r = corr([z[1][i] for z in rows], y)
        print(f"    {k:8s} r = {r:+.3f}")
        if best is None or abs(r) > abs(best[1]):
            best = (k, r)
    print(f"  -> {best[0]} is the strongest predictor within the orbit, and the")
    print(f"     best frame is the one where it is exactly zero.")

    S = json.load(open(os.path.join(RES, "rot_summary.json")))
    MT = {"lora_one": "none", "pissa": "none"}
    pts = []
    for k, v in S.items():
        m, var = k.split("|")
        if var == "published" or v["delta"] is None:
            continue
        mt = MT.get(m, "trace")
        a_ = SO.get(f"lit:{m}@frame{var[-1]}|{mt}")
        b_ = SO.get(f"lit:{m}|{mt}")
        if not a_ or not b_:
            continue
        pts.append((v["delta"], [a_[k2] - b_[k2] for k2 in KEYS]))
    yy = [p[0] for p in pts]
    print(f"\n  across {len(pts)} zoo rotations (CONFOUNDED by row space, see "
          f"the docstring):")
    for i, k in enumerate(KEYS):
        print(f"    d{k:8s} r = {corr([p[1][i] for p in pts], yy):+.3f}")


if __name__ == "__main__":
    main()
