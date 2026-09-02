"""Training-free test of the mechanism's own scaling prediction.

Both topics reduce their effect size to statistics of the pretrained model that
cost one forward+backward pass to measure:

  topic 02: the penalty of an exactly function-preserving gauge tracks how much
            that gauge homogenises the per-coordinate gradient energy.  The
            headroom is set by the participation ratio of the *pretrained*
            basis, PR = (sum E_j)^2 / (d * sum E_j^2).  PR = 1 means the basis
            is already homogeneous and there is nothing for a rotation to
            destroy; the smaller PR, the more Adam has to lose.

  topic 01: the two causal channels are the data-metric scale tr(P Sigma) and
            the data-metric effective rank r_eff^Sigma = (tr A Sigma A^T)^2 /
            ||A Sigma A^T||_F^2.  Both are governed by the anisotropy of the
            activation covariance Sigma, which is also a property of the
            pretrained model alone.

It also measures, without training, how much room the ADAPTER GAUGE leaves at
each scale: the span of Lambda_1 over the gauge orbit of a fixed random A, from
the gradient-metric eigenframe through the flat-diagonal frame to the direct
argmax.  That span is the dose available to the frame intervention, so it is a
prediction of how much the frame can matter at 8B, made from forward passes
alone before any 8B training.

If the outlier-feature structure of LLMs sharpens with scale -- as the
quantisation literature reports -- then PR falls and the anisotropy of Sigma
rises, and BOTH effects are predicted to grow.  If instead these statistics are
scale-invariant, the effects will not grow and the paper's claim has to be an
identifiability claim rather than a phenomenon claim.  Either way we learn it
here, for the price of a few forward passes.
"""
import argparse, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from common.gauge import make_R, fold_rmsnorm_gains
from common.pinit import kaiming_A
from common.intrinsic import (frame_ladder, l1_flatness, offdiag_mass,
                              max_l1_frame)
from common.data import build_sft, FixedOrderLoader
from common.train import load_model

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_SIDE = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


def pr(E):
    s1 = float(E.sum()); s2 = float((E * E).sum())
    return (s1 * s1) / (len(E) * s2 + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen3-0.6B-Base,"
                                        "Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-4B,"
                                        "Qwen/Qwen3-8B")
    ap.add_argument("--task", default="metamath")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(REPO, "paper",
                                                  "scaling_predictor.json"))
    a = ap.parse_args()
    rows = {}
    for mid in a.models.split(","):
        try:
            model, tok = load_model(mid, dtype=torch.float32)
        except Exception as e:                                   # noqa: BLE001
            print(f"{mid}: SKIP ({type(e).__name__})", flush=True)
            continue
        d = model.config.hidden_size
        fold_rmsnorm_gains(model)
        _, te = build_sft(tok, a.task, 64, 256, 384, seed=0)
        ld = FixedOrderLoader(te, a.bs, tok.pad_token_id, seed=0)
        mods = {n: m for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and n.split(".")[-1] in IN_SIDE}
        for p in model.parameters():
            p.requires_grad_(False)
        for m in mods.values():
            m.weight.requires_grad_(True)
        # activation covariance of the residual stream (shared by q/k/v)
        cov, hooks = {}, []
        reps = {n: m for n, m in mods.items() if n.endswith("q_proj")}

        def mk(n):
            def hook(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                cov[n] = cov.get(n, 0) + x.T @ x
            return hook
        for n, m in reps.items():
            hooks.append(m.register_forward_hook(mk(n)))
        for i in range(a.n_batches):
            b = {k: v.to("cuda") for k, v in ld.get(i).items()}
            o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            lg = o.logits[:, :-1].float(); lb = b["labels"][:, 1:]
            msk = lb != -100
            (nn.functional.cross_entropy(lg[msk], lb[msk]) / a.n_batches).backward()
        for h in hooks:
            h.remove()

        # --- topic 02: gradient-energy participation ratio, and how much an
        #     exactly function-preserving rotation raises it.  Hadamard needs a
        #     power-of-two width (Qwen3-4B has d = 2560), so use a random
        #     orthogonal rotation, which is defined for every width and is the
        #     rung the ladder shares across all models.
        H = make_R("rand", d, torch.Generator().manual_seed(1000)).float().cuda()
        pr0, pr1 = [], []
        for n, m in mods.items():
            G = m.weight.grad.detach().float()
            pr0.append(pr(G.pow(2).sum(0)))
            pr1.append(pr((G @ H).pow(2).sum(0)))

        # --- topic 01: anisotropy of Sigma, and the statistics of a random A
        g = torch.Generator().manual_seed(0)
        A = kaiming_A(a.r, d, g, "cpu").float().cuda()
        reff, aniso, evar = [], [], []
        for n, C in cov.items():
            C = C.cuda()
            M = A @ C @ A.T
            t = float(torch.diagonal(M).sum())
            reff.append(t * t / (float((M * M).sum()) + 1e-30))
            ev = torch.linalg.eigvalsh(C).clamp_min(0)
            aniso.append(float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30)))
            # tr(P Sigma) of the top-r eigen-subspace vs a random A, at matched
            # tr P: exactly the amplification a data-aware initializer buys
            V = torch.linalg.eigh(C)[1][:, -a.r:].T.contiguous()
            V = V * (A.norm() / V.norm())
            evar.append(float(((V @ C) * V).sum()) / max(float(((A @ C) * A).sum()), 1e-30))

        # --- the frame: how much room the gauge orbit leaves at this scale.
        #     Purely a property of the pretrained model plus a random A, so it
        #     is a PREDICTION of how much the frame can matter at 8B, made
        #     before any 8B training.
        lam_eig, lam_flat, lam_opt, off_g, off_x = [], [], [], [], []
        for n, C in cov.items():
            gn = n  # the q_proj whose input covariance this is
            G = mods[gn].weight.grad.detach().double().cuda()
            Ad = A.double()
            Mg = Ad @ (G.T @ G) @ Ad.T
            Q0, Q1 = frame_ladder(Mg, [0.0, 1.0])
            lam_eig.append(l1_flatness(Q0 @ Ad, G)[0])
            lam_flat.append(l1_flatness(Q1 @ Ad, G)[0])
            lam_opt.append(max_l1_frame(G @ Ad.T, iters=200)[1])
            off_g.append(offdiag_mass(Mg))
            off_x.append(offdiag_mass(Ad @ C.double().cuda() @ Ad.T))

        rows[mid] = dict(d_model=d,
                         lam1_eigframe=st.mean(lam_eig),
                         lam1_flatframe=st.mean(lam_flat),
                         lam1_maxframe=st.mean(lam_opt),
                         lam1_reach=st.mean(lam_opt) / st.mean(lam_eig),
                         offdiag_x_random_frame=st.mean(off_x),
                         offdiag_g_random_frame=st.mean(off_g),
                         PR_pretrained=st.mean(pr0),
                         PR_rotated=st.mean(pr1),
                         PR_ratio=st.mean(pr1) / st.mean(pr0),
                         n_modules=len(mods),
                         sigma_participation=st.mean(aniso) / d,
                         r_eff_sigma_random_A=st.mean(reff),
                         trPSigma_topr_over_random=st.mean(evar))
        r = rows[mid]
        print(f"{mid:26s} d={d:5d}  PR={r['PR_pretrained']:.4f} -> "
              f"{r['PR_rotated']:.4f} ({r['PR_ratio']:.1f}x)  "
              f"Sigma-PR={r['sigma_participation']:.4f}  "
              f"r_eff^Sigma(rand A)={r['r_eff_sigma_random_A']:.2f}  "
              f"tr(PSigma) top-r/rand={r['trPSigma_topr_over_random']:.1f}x",
              flush=True)
        print(f"{'':26s}    frame reach: Lambda_1 "
              f"{r['lam1_eigframe']:.3f} (eig) .. {r['lam1_flatframe']:.3f} "
              f"(flat) .. {r['lam1_maxframe']:.3f} (max) = "
              f"{r['lam1_reach']:.2f}x;  Off_x at a random frame "
              f"{r['offdiag_x_random_frame']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
