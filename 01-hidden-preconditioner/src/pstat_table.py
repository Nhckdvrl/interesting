"""Topic 01 -- the complete init-time statistic table for every initializer.

For each initializer we report, averaged over adapted modules:

  tr P                     parameter-space scale   (what LoRAM / NoRA control)
  tr(P Sigma_x)            data-space scale        (size of the function change)
  tr(P C_g)                gradient-metric scale   (first-order descent rate)
  r_eff(P)                 conditioning in parameter space
  cos(G, GP)               SGD  first-step descent efficiency
  cos(G, sign(GA^T)A)      Adam first-step descent efficiency

The last two are the quantities the first-order theory says should matter, and
they collapse "alignment" and "conditioning" into one number each.  For a P
whose eigenvectors are unrelated to C_g, cos(G,GP) = sqrt(r_eff/d_in); a
data-aware initializer breaks that identity by aligning P with C_g.
"""
import argparse, hashlib, json, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from common.pinit import make_A, kaiming_A
from common.pstats import p_stats
from common import initializers as IN
from common.data import build_sft, FixedOrderLoader
from common.train import load_model
sys.path.insert(0, os.path.dirname(__file__))
from run_lit import collect_grads, collect_act_cov, ACT_GROUP, cached_make_A

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONDS = (["kaiming", "left_gauge", "nora", "nora_unit", "etf",
          "flatspec_flatdiag"]
         + [f"geomspec_flatdiag{d}" for d in (0.8, 0.6, 0.5, 0.4, 0.3)]
         + ["eva", "gradsub", "pissa", "pissa_minor", "olora", "lora_one"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--match", default="trace")
    ap.add_argument("--every_layer", type=int, default=4)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(REPO, "01-hidden-preconditioner", "results",
                                f"pstat_table_{a.match}.json")

    model, tok = load_model(a.model, dtype=torch.float32)
    tr, _ = build_sft(tok, a.task, 6000, 256, 384, seed=0)
    ld = FixedOrderLoader(tr, 16, tok.pad_token_id, seed=0)
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
               "down_proj")
    allmods = {n: m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n.split(".")[-1] in targets}
    G = collect_grads(model, allmods, ld, a.probe_batches)
    ACT = collect_act_cov(model, allmods, ld, a.probe_batches)
    mods = {n: m for n, m in allmods.items()
            if int(n.split("layers.")[1].split(".")[0]) % a.every_layer == 0}
    print(f"{len(mods)} sampled modules of {len(allmods)}")
    s = a.alpha / a.r
    rows = {}

    for cond in CONDS:
        acc = {}
        for name, mod in mods.items():
            d_in = mod.weight.shape[1]
            h = int(hashlib.md5(f"{a.seed}:{name}".encode()).hexdigest()[:12], 16)
            g = torch.Generator().manual_seed(h)
            base = kaiming_A(a.r, d_in, g, "cpu")
            ref_tr = float(base.pow(2).sum())
            W = mod.weight.detach().float()
            A, B = None, None
            if cond == "kaiming":
                A = base
            elif cond == "left_gauge":
                g2 = torch.Generator().manual_seed(
                    int(hashlib.md5(f"gauge1:{name}".encode()).hexdigest()[:12], 16))
                A = make_A("left_gauge", a.r, d_in, g2, "cpu", ref_A=base)
            elif cond == "etf":
                A = IN.init_etf(a.r, d_in, g, ref_tr)
            elif cond == "eva":
                key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
                A = IN.init_eva(ACT[key].cuda(), a.r, d_in, ref_tr).cpu().double()
            elif cond == "gradsub":
                A = IN.init_gradsubspace(G[name].cuda(), a.r, ref_tr).cpu().double()
            elif cond in ("pissa", "pissa_minor"):
                A, B = IN.init_pissa(W, a.r, s, minor=(cond == "pissa_minor"))
                A, B = A.cpu().double(), B.cpu().double()
            elif cond == "olora":
                A, B = IN.init_olora(W, a.r, s)
                A, B = A.cpu().double(), B.cpu().double()
            elif cond == "lora_one":
                A, B = IN.init_lora_one(G[name].cuda(), a.r, s, W=W)
                A, B = A.cpu().double(), B.cpu().double()
            else:
                A = cached_make_A(cond, a.r, d_in, f"{a.seed}:{name}", base)

            key = name.rsplit(".", 1)[0] + "." + ACT_GROUP[name.split(".")[-1]]
            Sig = ACT[key].cuda(); Gc = G[name].cuda()

            def wt(M):
                Md = M.float().cuda()
                return (float(Md.pow(2).sum()),
                        float(((Md @ Sig) * Md).sum()),
                        float((Gc @ Md.T).pow(2).sum()))
            if a.match != "none":
                idx = {"trace": 0, "trace_act": 1, "trace_grad": 2}[a.match]
                k = (wt(base)[idx] / max(wt(A)[idx], 1e-30)) ** 0.5
                A = A * k
                if B is not None:
                    B = B / k
            tp, ta, tg = wt(A); bp, ba, bg = wt(base)
            Ad = A.float().cuda()
            GP = (Gc @ Ad.T) @ Ad
            U = torch.sign(Gc @ Ad.T) @ Ad
            gn = Gc.norm()
            stt = p_stats(A, s=1.0)
            row = dict(r_eff=stt["eff_rank"], diag_imb=stt["diag_imbalance"],
                       rel_trP=tp / bp, rel_act=ta / ba, rel_grad=tg / bg,
                       cos_sgd=float((Gc * GP).sum() / (gn * GP.norm() + 1e-30)),
                       cos_adam=float((Gc * U).sum() / (gn * U.norm() + 1e-30)),
                       B0=float(B.norm()) if B is not None else 0.0)
            for k2, v in row.items():
                acc.setdefault(k2, []).append(v)
        rows[cond] = {k2: st.mean(v) for k2, v in acc.items()}
        r = rows[cond]
        print(f"  {cond:22s} r_eff={r['r_eff']:6.2f} trP={r['rel_trP']:7.2f} "
              f"trPS={r['rel_act']:8.2f} trPCg={r['rel_grad']:9.2f} "
              f"cos_sgd={r['cos_sgd']:.5f} cos_adam={r['cos_adam']:.5f} "
              f"B0={r['B0']:6.2f}")
    json.dump(rows, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
