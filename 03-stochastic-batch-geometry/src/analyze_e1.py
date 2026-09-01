"""Topic 03 E1 -- best-tuned loss vs logical batch size, per method."""
import glob, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "03-stochastic-batch-geometry", "results")


def load(tags):
    rows = []
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(RES, tag, "*.json"))):
            r = json.load(open(f)); a = r["args"]
            rows.append(dict(tag=tag, method=a["method"], r=a["r"], bs=a["bs"],
                             lr=a["lr"], seed=a["seed"], steps=r["steps"],
                             loss=r["log"]["final_eval_loss"],
                             base=r["base_eval_loss"]))
    return rows


def main(tags):
    rows = load(tags)
    keys = sorted({(r["method"], r["tag"], r["r"]) for r in rows})
    batches = sorted({r["bs"] for r in rows})
    print(f"{len(rows)} runs\n")
    print("best-tuned final eval loss (LR swept per cell)\n")
    hdr = "  " + f"{'method':22s}" + "".join(f"{('bs=%d' % b):>12s}" for b in batches)
    print(hdr); print("  " + "-" * (22 + 12 * len(batches)))
    table = {}
    for k in keys:
        m, tag, rk = k
        name = f"{m}" + (f" r={rk}" if m != "full" else "")
        cells = []
        for b in batches:
            rs = [x for x in rows if (x["method"], x["tag"], x["r"]) == k
                  and x["bs"] == b]
            if rs:
                best = min(rs, key=lambda x: x["loss"])
                table[(name, b)] = best
                cells.append(f"{best['loss']:12.5f}")
            else:
                cells.append(f"{'-':>12s}")
        print(f"  {name:22s}" + "".join(cells))
    print("\nbest LR per cell")
    print(hdr); print("  " + "-" * (22 + 12 * len(batches)))
    for k in keys:
        m, tag, rk = k
        name = f"{m}" + (f" r={rk}" if m != "full" else "")
        cells = [f"{table[(name,b)]['lr']:12.1e}" if (name, b) in table
                 else f"{'-':>12s}" for b in batches]
        print(f"  {name:22s}" + "".join(cells))

    full = {b: table[("full", b)]["loss"] for b in batches if ("full", b) in table}
    print("\nGAP vs FullFT (positive = LoRA worse)")
    print(hdr); print("  " + "-" * (22 + 12 * len(batches)))
    for k in keys:
        m, tag, rk = k
        if m == "full":
            continue
        name = f"{m} r={rk}"
        cells = []
        for b in batches:
            if (name, b) in table and b in full:
                cells.append(f"{table[(name,b)]['loss'] - full[b]:+12.5f}")
            else:
                cells.append(f"{'-':>12s}")
        print(f"  {name:22s}" + "".join(cells))

    print("\nfull LR sweeps")
    for k in keys:
        m, tag, rk = k
        name = f"{m}" + (f" r={rk}" if m != "full" else "")
        print(f"\n  {name}")
        for b in batches:
            rs = sorted([x for x in rows if (x["method"], x["tag"], x["r"]) == k
                         and x["bs"] == b], key=lambda x: x["lr"])
            if rs:
                print(f"    bs={b:4d} steps={rs[0]['steps']:5d}  " +
                      "  ".join(f"{x['lr']:.0e}:{x['loss']:.4f}" for x in rs))


if __name__ == "__main__":
    main(sys.argv[1:] or ["e1", "e1_r128"])
