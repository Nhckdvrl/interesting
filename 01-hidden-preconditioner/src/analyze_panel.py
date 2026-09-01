"""Summarise a topic-01 panel.

The key comparison is:
    spread across `left_gauge` runs  (IDENTICAL P0 -> zero preconditioner
    content; pure AdamW non-covariance)
versus
    spread across methods (different P0).
A method effect that does not exceed the gauge null is not a preconditioner
effect at all.
"""
import glob, json, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load(tag, base=None):
    d = base or os.path.join(REPO, "01-hidden-preconditioner", "results", tag)
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            out.append(json.load(open(f)))
        except Exception as e:
            print("bad", f, e)
    return out


def main(tag="g1"):
    recs = load(tag)
    print(f"{len(recs)} runs in {tag}\n")
    by = {}
    for r in recs:
        a = r["args"]
        by.setdefault(a["lr"], {}).setdefault(a["cond"], []).append(r)

    for lr in sorted(by):
        print(f"--- lr = {lr:g} " + "-" * 56)
        print(f"  {'condition':24s} {'n':>2s} {'final eval loss':>26s} "
              f"{'|dW|':>8s} {'trP':>9s} {'diag_imb':>9s}")
        rows = {}
        for cond in ["kaiming", "left_gauge", "nora", "nora_unit",
                     "kaimingspec_flatdiag"]:
            rs = by[lr].get(cond, [])
            if not rs:
                continue
            fe = [x["log"]["final_eval_loss"] for x in rs]
            dw = [x["log"]["diag"][-1]["dW_norm"] for x in rs]
            ps = list(rs[0]["init_pstats"].values())
            trp = st.mean(p["tr_P"] for p in ps)
            di = st.mean(p["diag_imbalance"] for p in ps)
            sd = st.pstdev(fe) if len(fe) > 1 else 0.0
            rows[cond] = (fe, sd)
            print(f"  {cond:24s} {len(fe):2d}  {st.mean(fe):.5f} "
                  f"+- {sd:.5f}  [{min(fe):.5f},{max(fe):.5f}]  "
                  f"{st.mean(dw):8.3f} {trp:9.4f} {di:9.2e}")
        if "left_gauge" in rows and "kaiming" in rows:
            gauge_sd = rows["left_gauge"][1]
            gauge_rng = max(rows["left_gauge"][0]) - min(rows["left_gauge"][0])
            base = st.mean(rows["kaiming"][0])
            print(f"\n  GAUGE NULL (identical P0):  sd={gauge_sd:.5f}  "
                  f"range={gauge_rng:.5f}")
            for cond, (fe, sd) in rows.items():
                if cond in ("kaiming", "left_gauge"):
                    continue
                eff = st.mean(fe) - base
                z = eff / gauge_sd if gauge_sd > 0 else float('inf')
                print(f"    {cond:24s} effect vs kaiming = {eff:+.5f}  "
                      f"= {z:+.1f} x gauge-null sd")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "g1")
