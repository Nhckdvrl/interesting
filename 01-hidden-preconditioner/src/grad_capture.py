"""Topic 01 -- init-time diagnostics, no training required.

For each candidate A-init we measure, on real minibatch gradients of the
pretrained model, the quantities that the first-step theory says should control
descent:

  SGD:   dW_1 = -eta G P.  The loss decrease per unit update norm is
             cos(G, GP) = <G,GP> / (||G|| ||GP||).
         For an isotropic G and P with spectrum lam this equals
             sqrt(r_eff(P) / d_in),      r_eff = (tr P)^2/||P||_F^2,
         i.e. the *effective rank* of the hidden preconditioner, and NOT its
         diagonal, is the quantity that controls first-order efficiency.

  Adam:  at B_0 = 0 the bias-corrected first step is
             dW_1 = -lr * s * sign(G A^T) A,
         which is NOT -eta G P.  We measure cos(G, sign(GA^T)A) as the Adam
         analogue.  Any difference between the two columns is a direct estimate
         of how much of NoRA's mother proposition survives the optimizer that
         is actually used in practice.
"""
import argparse, hashlib, json, os, sys, statistics as stat
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from common.pinit import make_A, kaiming_A
from common.pstats import p_stats
from common.data import build_sft, FixedOrderLoader
from common.train import load_model

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONDS = ["kaiming", "nora", "nora_unit", "kaimingspec_flatdiag",
         "flatspec_flatdiag"] + [f"geomspec_flatdiag{d}" for d in
                                 (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)]


@torch.no_grad()
def stats_for(G, A, s):
    G = G.double(); A = A.double()
    P_apply = lambda M: (M @ A.T) @ A * (s * s)
    GP = P_apply(G)
    gn = G.norm(); gpn = GP.norm()
    cos_sgd = float((G * GP).sum() / (gn * gpn + 1e-30))
    # Adam first step at B=0 (bias corrected): -lr * s * sign(G A^T) A
    U = torch.sign(G @ A.T) @ A * s
    un = U.norm()
    cos_adam = float((G * U).sum() / (gn * un + 1e-30))
    return cos_sgd, cos_adam, float(gpn / gn), float(un / gn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        REPO, "01-hidden-preconditioner", "results", "grad_capture.json"))
    a = ap.parse_args()

    model, tok = load_model(a.model, dtype=torch.float32)
    tr, _ = build_sft(tok, a.task, 2000, 64, 384, seed=0)
    ld = FixedOrderLoader(tr, a.bs, tok.pad_token_id, seed=0)
    targets = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
    mods = {n: m for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in targets}
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods.values():
        m.weight.requires_grad_(True)

    # accumulate the full-weight gradient over a few batches
    for i in range(a.n_batches):
        b = {k: v.to("cuda") for k, v in ld.get(i).items()}
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        lg = out.logits[:, :-1].float(); lb = b["labels"][:, 1:]
        msk = lb != -100
        loss = nn.functional.cross_entropy(lg[msk], lb[msk])
        (loss / a.n_batches).backward()
    Gs = {n: m.weight.grad.detach().float().cpu() for n, m in mods.items()}
    print(f"collected gradients for {len(Gs)} modules")

    s = a.alpha / a.r
    rows = {}
    for cond in CONDS:
        acc = {k: [] for k in ("cos_sgd", "cos_adam", "gp_ratio", "u_ratio",
                               "r_eff", "tr_P", "diag_imb")}
        for n, G in Gs.items():
            d_in = G.shape[1]
            h = int(hashlib.md5(f"{a.seed}:{n}".encode()).hexdigest()[:12], 16)
            g = torch.Generator().manual_seed(h)
            base = kaiming_A(a.r, d_in, g, "cpu")
            A = base if cond == "kaiming" else make_A(cond, a.r, d_in, g, "cpu",
                                                      ref_A=base)
            cs, ca, gpr, ur = stats_for(G, A, s)
            st = p_stats(A, s=s)
            for k, v in (("cos_sgd", cs), ("cos_adam", ca), ("gp_ratio", gpr),
                         ("u_ratio", ur), ("r_eff", st["eff_rank"]),
                         ("tr_P", st["tr_P"]),
                         ("diag_imb", st["diag_imbalance"])):
                acc[k].append(v)
        rows[cond] = {k: stat.mean(v) for k, v in acc.items()}
        rows[cond]["cos_sgd_over_sqrt_reff"] = (
            rows[cond]["cos_sgd"] / (rows[cond]["r_eff"] ** 0.5))
        print(f"  {cond:24s} r_eff={rows[cond]['r_eff']:6.2f} "
              f"trP={rows[cond]['tr_P']:9.3f} "
              f"cos_sgd={rows[cond]['cos_sgd']:.5f} "
              f"cos_adam={rows[cond]['cos_adam']:.5f} "
              f"cos_sgd/sqrt(r_eff)={rows[cond]['cos_sgd_over_sqrt_reff']:.6f}")
    json.dump(rows, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
