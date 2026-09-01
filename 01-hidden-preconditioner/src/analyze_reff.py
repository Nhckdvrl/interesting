"""Topic 01 -- r_eff(P) dose-response at matched trace and flat diagonal."""
import glob, json, math, os, sys, statistics as st
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
GC = json.load(open(os.path.join(RES, "grad_capture.json")))


def main(tags):
    rows = []
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
            r = json.load(open(f)); a = r["args"]
            ps = list(r["init_pstats"].values())
            rows.append(dict(cond=a["cond"], lr=a["lr"], seed=a["seed"],
                             steps=a["steps"],
                             opt=a.get("optimizer", "adamw"),
                             r_eff=st.mean(p["eff_rank"] for p in ps),
                             dW=r["log"]["diag"][-1]["dW_norm"],
                             loss=r["log"]["final_eval_loss"]))
    conds = sorted({r["cond"] for r in rows},
                   key=lambda c: -st.mean([x["r_eff"] for x in rows
                                           if x["cond"] == c]))
    for opt in sorted({r["opt"] for r in rows}):
        for steps in sorted({r["steps"] for r in rows}):
            sub = [r for r in rows if r["opt"] == opt and r["steps"] == steps]
            if not sub:
                continue
            lrs = sorted({r["lr"] for r in sub})
            print(f"\n=== optimizer={opt}  steps={steps} " + "=" * 40)
            print(f"  {'condition':24s} {'r_eff':>6s} {'cos_sgd':>8s} "
                  + "".join(f"{('lr=%.0e' % l):>11s}" for l in lrs)
                  + f"{'best':>10s}")
            ref = None
            for c in conds:
                cr = [r for r in sub if r["cond"] == c]
                if not cr:
                    continue
                cells = []
                for l in lrs:
                    v = [x["loss"] for x in cr if x["lr"] == l]
                    cells.append(f"{st.mean(v):11.5f}" if v else f"{'-':>11s}")
                best = min(x["loss"] for x in cr)
                if ref is None:
                    ref = best
                re_ = st.mean(x["r_eff"] for x in cr)
                cs = GC.get(c, {}).get("cos_sgd", float("nan"))
                print(f"  {c:24s} {re_:6.2f} {cs:8.5f}" + "".join(cells)
                      + f"{best:10.5f}")
            # correlation of penalty with 1/sqrt(r_eff)
            pts = []
            for c in conds:
                cr = [r for r in sub if r["cond"] == c]
                if not cr or c == "kaiming":
                    continue
                pts.append((st.mean(x["r_eff"] for x in cr),
                            min(x["loss"] for x in cr)))
            if len(pts) > 3:
                xs = [1 / math.sqrt(p[0]) for p in pts]
                ys = [p[1] for p in pts]
                mx, my = st.mean(xs), st.mean(ys)
                num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                den = (sum((x - mx) ** 2 for x in xs) *
                       sum((y - my) ** 2 for y in ys)) ** .5
                slope = num / max(sum((x - mx) ** 2 for x in xs), 1e-30)
                print(f"\n  best-tuned loss vs 1/sqrt(r_eff):  "
                      f"pearson r = {num/max(den,1e-30):+.3f}, "
                      f"slope = {slope:+.4f} nats per unit 1/sqrt(r_eff)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["g1c", "g1c_long"])
