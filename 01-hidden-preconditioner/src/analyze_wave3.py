"""Wave 3 -- the exactly matched causal ladder in W.

S and D are fixed by construction (M_x = Lambda identically) and the
spectrum-weighted alignment R_g is driven to the vanilla draw's value, so W is
the only coordinate that moves.  This is the test the wave-2 whitening sweep
could not be, because there S was held but D and the row space drifted.
"""
import glob, json, os, statistics as st, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")


def main(tag="atlas"):
    pts = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        if a.get("matchW") is None:
            continue
        m = list(r["init_stats"].values())
        p = pts.setdefault(a["matchW"], dict(
            S=st.mean(x["S_rel"] for x in m), D=st.mean(x["D"] for x in m),
            W=st.mean(x["W"] / x["W_ref"] for x in m),
            Rg=st.mean(x["R_g"] / x["R_g_ref"] for x in m),
            trP=st.mean(x["tr_P"] for x in m), curve={}))
        p["curve"][a["lr"]] = r["log"]["final_eval_loss"]
    print("Exact causal ladder in W = tr(AA^T)/tr(A Sigma A^T)\n")
    print(f"  {'W/W0':>7s} {'S_rel':>7s} {'D':>6s} {'Rg/Rg0':>7s} {'tr P':>9s} | "
          f"{'L*':>9s} {'lr*':>8s} {'ok':>4s}")
    L, Wv = {}, []
    for k in sorted(pts):
        p = pts[k]; lrs = sorted(p["curve"])
        b = min(p["curve"], key=p["curve"].get)
        ok = "y" if b not in (lrs[0], lrs[-1]) else "EDGE"
        print(f"  {p['W']:7.3f} {p['S']:7.4f} {p['D']:6.3f} {p['Rg']:7.3f} "
              f"{p['trP']:9.3f} | {p['curve'][b]:9.5f} {b:8.0e} {ok:>4s}")
        L[p["W"]] = p["curve"][b]; Wv.append((p["W"], p["curve"][b], b))
    if len(Wv) >= 3:
        lo = min(Wv, key=lambda x: x[1])
        print(f"\n  optimum of the ladder at W/W0 = {lo[0]:.2f} "
              f"(the vanilla draw sits at 1.00 by definition)")
        print(f"  total effect of W over the ladder: "
              f"{max(x[1] for x in Wv) - min(x[1] for x in Wv):.5f} nats "
              f"= {(max(x[1] for x in Wv)-min(x[1] for x in Wv))/2.7e-4:.1f}x "
              f"the measurement null")
        lrs = {x[2] for x in Wv}
        print(f"  tr P spans {max(p['trP'] for p in pts.values())/min(p['trP'] for p in pts.values()):.0f}x "
              f"across the ladder while lr* takes {len(lrs)} distinct value(s): "
              f"{sorted(lrs)}")
        print(f"  -> tr P is not the learning-rate coordinate; S is.")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
