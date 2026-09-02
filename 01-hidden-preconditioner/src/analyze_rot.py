"""Rotating the published zoo: is the frame a free gain, and which functional
governs it?

Predictions were committed in PREDICTIONS_mechanism.md before these runs
finished.  Each method appears three times -- as published, rotated to the
gradient-metric eigenframe (`@frame0`), and rotated to a flat gradient-metric
diagonal (`@frame1`).  All three have the same `B A` to 1e-15, the same
`P = s^2 A^T A` to 1e-15, and the same nine gauge invariants, so SGD cannot
tell them apart at all.

NOTE ON THE FIRST READING.  With the original grid (1e-4 .. 5e-4) four of the
six methods tuned to the bottom rung, and `eva@frame0` read as +0.00243 nats
WORSE than `eva`, which was recorded as favouring the diagonal-preconditioner
hypothesis.  Extending the grid to 2e-5 and 5e-5 gave every method an interior
optimum and reversed that cell to -0.00079, i.e. better.  The first reading was
a grid truncation, not a result; it is left in the commit history rather than
quietly dropped.
"""
import glob, json, math, os, statistics as st, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
NULL = 2.7e-4
MATCH = {"lora_one": "none", "pissa": "none", "gradsub": "trace",
         "eva": "trace", "bimi": "trace", "kaiming": "trace"}


def main(tag="rot"):
    SO = json.load(open(os.path.join(RES, "second_order.json")))
    C = {}
    for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
        r = json.load(open(f)); a = r["args"]
        C.setdefault(a["cond"], {})[a["lr"]] = r["log"]["eval_loss"][-1]

    def best(c):
        cur = C.get(c)
        if not cur or len(cur) < 3:
            return None, None, False
        lrs = sorted(cur); b = min(cur, key=cur.get)
        return cur[b], b, b not in (lrs[0], lrs[-1])

    print(f"{'method':>10s} {'variant':>9s} {'Lam1':>7s} {'Off_x':>7s} "
          f"{'L*':>9s} {'lr*':>7s}  {'delta vs published':>19s}")
    rows = {}
    for m, mt in MATCH.items():
        base, _, _ = best(m)
        for v, c in (("published", m), ("@frame0", f"{m}@frame0"),
                     ("@frame1", f"{m}@frame1")):
            L, lr, ok = best(c)
            so = SO.get(f"lit:{c}|{mt}", {})
            if L is None:
                continue
            d = "" if v == "published" or base is None else \
                f"{L - base:+.5f} ({(L-base)/NULL:+.1f}x null)"
            print(f"{m if v=='published' else '':>10s} {v:>9s} "
                  f"{so.get('Lam1', float('nan')):7.4f} "
                  f"{so.get('Off_x', float('nan')):7.4f} {L:9.5f} {lr:7.0e}"
                  f"{'' if ok else '*'}  {d:>19s}")
            rows[(m, v)] = dict(L=L, Lam1=so.get("Lam1"),
                                Off_x=so.get("Off_x"), delta=None if base is
                                None or v == "published" else L - base)
        print()
    print("  (* = tuned learning rate at the edge of the grid)\n")

    # --- the pre-registered discriminating cases -------------------------
    print("pre-registered discriminating predictions (PREDICTIONS_mechanism.md):")
    for m, h1, h2 in (
            ("eva", "@frame0 BEATS published", "@frame0 is WORSE; eva is already"
             " at the optimum (Off_x = 0.000 by construction)"),
            ("lora_one", "@frame0 helps", "@frame0 is a near-null (Off_x"
             " unchanged 0.370 -> 0.370)"),
            ("gradsub", "@frame0 helps", "@frame0 is a near-null (Off_x"
             " unchanged 0.570 -> 0.570)")):
        d = rows.get((m, "@frame0"), {}).get("delta")
        if d is None:
            continue
        verdict = ("H1 (descent rate)" if d < -NULL else
                   "H2 (diagonal preconditioner)" if d > NULL else
                   "null -- consistent with H2's near-null cases")
        print(f"  {m:9s} @frame0 - published = {d:+.5f} nats "
              f"({d/NULL:+.1f}x null)  -> favours {verdict}")
        print(f"            H1 said: {h1}")
        print(f"            H2 said: {h2}")

    # --- the practical claim ---------------------------------------------
    kb = rows.get(("kaiming", "published"), {}).get("L")
    kf = rows.get(("kaiming", "@frame0"), {}).get("L")
    if kb and kf:
        best_pub = min((v["L"], m) for (m, vv), v in rows.items()
                       if vv == "published")
        print(f"\npractical: vanilla Kaiming {kb:.5f} -> rotated {kf:.5f} "
              f"({kf-kb:+.5f} nats)")
        print(f"           best published initialiser in this panel: "
              f"{best_pub[1]} at {best_pub[0]:.5f}")
        if best_pub[0] < kb:
            frac = (kb - kf) / (kb - best_pub[0])
            print(f"           a free rotation of the vanilla draw recovers "
                  f"{100*frac:.0f}% of the gap between Kaiming and the best "
                  f"published method")
    # --- how big is the frame next to what these papers compare? ----------
    pub = {m: rows[(m, "published")]["L"] for m in MATCH
           if (m, "published") in rows}
    fr = {}
    for m in MATCH:
        v = [rows[(m, x)]["L"] for x in ("published", "@frame0", "@frame1")
             if (m, x) in rows]
        if len(v) == 3:
            fr[m] = max(v) - min(v)
    if len(pub) >= 3 and len(fr) >= 3:
        tot = max(pub.values()) - min(pub.values())
        adj = sorted(pub.values())
        gaps = [adj[i + 1] - adj[i] for i in range(len(adj) - 1)]
        mf = st.median(list(fr.values()))
        print(f"\nscale: the six published initialisers span {tot:.5f} nats "
              f"between them.")
        print(f"       the frame alone moves ONE method by {mf:.5f} nats "
              f"(median), {max(fr.values()):.5f} (max)")
        print(f"       = {100*mf/tot:.0f}% to {100*max(fr.values())/tot:.0f}% "
              f"of that entire range,")
        print(f"       and {mf/st.median(gaps):.1f}x the median gap between "
              f"ADJACENT methods in the ranking ({st.median(gaps):.5f} nats).")
        print(f"       It preserves B A, P = s^2 A^T A and all nine gauge "
              f"invariants to 1e-15, and no paper reports it.")

    json.dump({f"{m}|{v}": r for (m, v), r in rows.items()},
              open(os.path.join(RES, f"{tag}_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
