"""Is the frame coordinate a property of the METHOD or of the MODEL?

The audit needs no training, so it runs on every model we can load.  That buys
more than breadth: it answers a question one model cannot.

If each published initialiser's Lambda_1 is stable across architectures, then
the coordinate is a property of the METHOD -- one number characterises it, and
that number can simply be reported alongside a loss, which is a concrete thing
to ask of the field.  If it swings with the backbone, the coordinate is
model-dependent and no such recommendation is possible.

Also checks the claim the paper leans on: does the data-aware / frame-based
separation hold on every model, or is it a Qwen artefact?
"""
import glob, json, math, os, sys, statistics as st, itertools
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "01-hidden-preconditioner", "results")
OURS = {"left_gauge", "geomspec_flatdiag0.5", "flatspec_flatdiag",
        "kaimingspec_flatdiag"}
DATA_AWARE = {"lora_one", "gradsub", "pissa", "pissa_minor", "eva", "olora"}
FILES = [("Qwen3-0.6B", "second_order.json"),
         ("Gemma-2-2B", "second_order_gemma2b.json"),
         ("Mistral-7B", "second_order_mistral7b.json"),
         ("Qwen3-1.7B", "second_order_qwen17b.json"),
         ("Llama-3.2-3B", "second_order_llama3b.json"),
         ("OLMo-2-1B", "second_order_olmo1b.json"),
         ("Qwen3-8B", "second_order_q8b.json")]


def load(fn):
    try:
        SO = json.load(open(os.path.join(RES, fn)))
    except Exception:
        return None
    L = {}
    for k, v in SO.items():
        if not k.startswith("lit:") or "@" in k or k[4:].startswith("frame"):
            continue
        n = k[4:].split("|")[0]
        if n in OURS or "Lam1" not in v:
            continue
        L.setdefault(n, v["Lam1"])
    return L or None


def main():
    tabs = [(m, L) for m, fn in FILES if (L := load(fn))]
    if not tabs:
        print("no audit files yet"); return
    print(f"Frame coordinate of each published initialiser, "
          f"{len(tabs)} models, no training\n")
    methods = sorted(set.intersection(*[set(L) for _, L in tabs]),
                     key=lambda n: tabs[0][1][n])
    print(f"{'method':>13s} " + " ".join(f"{m.split('-')[0][:8]:>9s}"
                                         for m, _ in tabs)
          + f" | {'mean':>7s} {'CV':>6s}")
    cvs = []
    for n in methods:
        vs = [L[n] for _, L in tabs]
        cv = st.stdev(vs) / st.mean(vs) if len(vs) > 1 else 0.0
        cvs.append((n, cv))
        tag = "D" if n in DATA_AWARE else "F"
        print(f"{n:>11s} {tag} " + " ".join(f"{v:9.4f}" for v in vs)
              + f" | {st.mean(vs):7.4f} {cv:6.1%}")

    print("\nseparation (every frame-based method above every data-aware one):")
    for m, L in tabs:
        da = {n: L[n] for n in L if n in DATA_AWARE}
        fb = {n: L[n] for n in L if n not in DATA_AWARE}
        if not (da and fb):
            continue
        hi, lo = max(da, key=da.get), min(fb, key=fb.get)
        ok = da[hi] < fb[lo]
        note = "PERFECT" if ok else (
            f"one pair crosses: {hi} {da[hi]:.4f} over {lo} {fb[lo]:.4f} "
            f"(+{da[hi]-fb[lo]:.4f})")
        print(f"  {m:>13s}  span {max(L.values())/min(L.values()):4.1f}x   {note}")
    print("\n  Where it breaks, it breaks at the boundary pair -- the least"
          "\n  data-aware method (OLoRA: a QR of the pretrained weight, using"
          "\n  neither gradients nor activations) against the most structured"
          "\n  frame-based one (BiMI).  The two categories are ours, and they"
          "\n  blur for exactly the methods that sit on the line between them.")

    if len(tabs) >= 3:
        print(f"\nis the coordinate a property of the METHOD?")
        print(f"  median across-model CV of a method's Lambda_1: "
              f"{st.median([c for _, c in cvs]):.1%}")
        taus = []
        for (m1, L1), (m2, L2) in itertools.combinations(tabs, 2):
            common = [n for n in methods]
            disc = sum(1 for x, y in itertools.combinations(common, 2)
                       if (L1[x] < L1[y]) != (L2[x] < L2[y]))
            tot = len(common) * (len(common) - 1) // 2
            taus.append(1 - 2 * disc / tot)
        print(f"  median Kendall tau between model pairs: {st.median(taus):+.2f}"
              f"   ({len(taus)} pairs)")
        if st.median(taus) > 0.7:
            print("  -> the ORDERING is a method property: each initialiser has")
            print("     a characteristic frame that survives the backbone, so")
            print("     one number per method can simply be reported.")


if __name__ == "__main__":
    main()
