"""The gauge-frame ladder: a coordinate AdamW sees and SGD provably cannot.

Predictions were committed in 01-hidden-preconditioner/PREDICTIONS_frame.md
before the panel ran.  The SGD arm is not a flat expectation but a null with a
predicted value of zero: SGD's whole trajectory is gauge-covariant, so the six
frame conditions are the same run in rotated coordinates.
"""
import glob, json, math, os, statistics as st, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
NULL_ADAM, NULL_SGD = 2.7e-4, 1.5e-6


def load(tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        opt = a.get("optimizer", "adamw")
        out.setdefault((opt, a["cond"]), {})[a["lr"]] = \
            r["log"]["eval_loss"][-1]
    return out


def main(tag="frame", order=None):
    SO = json.load(open(os.path.join(RES, "second_order.json")))
    D = load(tag)
    conds = order or sorted({c for _, c in D})
    print(f"{'condition':>14s} {'Lam1':>7s} {'E_g':>7s} | "
          f"{'AdamW L*':>9s} {'lr*':>8s} | {'SGD L*':>9s} {'lr*':>7s}")
    rows = []
    for c in conds:
        so = SO.get(f"lit:{c}|trace") or SO.get(f"lit:{c}|none") or {}
        r = dict(cond=c, Lam1=so.get("Lam1"), E_g=so.get("E_g"))
        for opt in ("adamw", "sgd"):
            cur = D.get((opt, c))
            if not cur:
                r[opt] = r[opt + "_lr"] = None
                continue
            lrs = sorted(cur)
            b = min(cur, key=cur.get)
            r[opt], r[opt + "_lr"] = cur[b], b
            r[opt + "_edge"] = b in (lrs[0], lrs[-1])
        rows.append(r)
        f2 = lambda v, w, p="": f"{v:{w}.5f}" if isinstance(v, float) else " " * w
        print(f"{c:>14s} {r['Lam1'] if r['Lam1'] else float('nan'):7.4f} "
              f"{r['E_g'] if r['E_g'] else float('nan'):7.4f} | "
              f"{f2(r['adamw'], 9)} {r['adamw_lr'] or 0:8.0e}"
              f"{'*' if r.get('adamw_edge') else ' '}| "
              f"{f2(r['sgd'], 9)} {r['sgd_lr'] or 0:7.2g}"
              f"{'*' if r.get('sgd_edge') else ''}")
    print("  (* = the tuned learning rate sits at the edge of the grid)")

    # the ordering at EACH learning rate, which is the honest test: a tuned-loss
    # ordering can be a grid artefact, an ordering that holds at every rung
    # cannot.
    print("\n  ordering at each learning rate (AdamW), conditions sorted by "
          "Lambda_1:")
    by_lam = sorted([r for r in rows if r["Lam1"] is not None
                     and r["cond"].startswith("frame")],
                    key=lambda r: r["Lam1"])
    lrs = sorted({lr for (o, c), cur in D.items() if o == "adamw"
                  for lr in cur})
    for lr in lrs:
        v = [D.get(("adamw", r["cond"]), {}).get(lr) for r in by_lam]
        if any(x is None for x in v) or len(v) < 3:
            continue
        up = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
        dn = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
        print(f"    lr={lr:<8.0e} " + " ".join(f"{x:.5f}" for x in v)
              + f"   spread {max(v)-min(v):.5f}"
              + ("   monotone: low Lambda_1 better" if up else
                 "   monotone: high Lambda_1 better" if dn else "   mixed"))

    print()
    for opt, null, name in (("adamw", NULL_ADAM, "AdamW"),
                            ("sgd", NULL_SGD, "SGD")):
        v = [r[opt] for r in rows if r[opt] is not None]
        if len(v) < 2:
            continue
        spread = max(v) - min(v)
        print(f"  {name:5s} spread over the ladder: {spread:.5f} nats "
              f"= {spread/NULL_ADAM:.1f}x the AdamW measurement null "
              f"({NULL_ADAM:.1e})")
    lad = [r for r in rows if r["cond"].startswith("frame")
           and r["adamw"] is not None and r["Lam1"] is not None]
    if len(lad) >= 3:
        lad.sort(key=lambda r: r["Lam1"])
        up = all(lad[i]["adamw"] <= lad[i + 1]["adamw"]
                 for i in range(len(lad) - 1))
        dn = all(lad[i]["adamw"] >= lad[i + 1]["adamw"]
                 for i in range(len(lad) - 1))
        mono = ("monotone INCREASING in Lambda_1 (concentration wins)" if up
                else "monotone decreasing in Lambda_1 (spreading wins)" if dn
                else "not monotone")
        n = len(lad)
        mx = sum(r["Lam1"] for r in lad) / n
        my = sum(r["adamw"] for r in lad) / n
        sxy = sum((r["Lam1"] - mx) * (r["adamw"] - my) for r in lad)
        sxx = sum((r["Lam1"] - mx) ** 2 for r in lad)
        syy = sum((r["adamw"] - my) ** 2 for r in lad)
        print(f"\n  AdamW loss vs Lambda_1 over the ladder: "
              f"slope {sxy/sxx:+.4f} nats per unit Lambda_1, "
              f"r = {sxy/math.sqrt(sxx*syy+1e-30):+.3f}, "
              f"{mono}")
        print(f"  the left-gauge null implies {NULL_ADAM/0.0105:.4f} nats per "
              f"unit Lambda_1 near the vanilla frame "
              f"(2.7e-4 nats over dLambda_1 = 0.0105)")
    json.dump(rows, open(os.path.join(RES, f"{tag}_summary.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
