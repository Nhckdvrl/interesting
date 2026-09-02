"""Shared SFT trainer with optimization diagnostics.

Everything a matched comparison needs is logged: not just endpoint accuracy but
loss trajectories, merged-update norms, factor movement, and P-statistics.
"""
import json, math, os, time, contextlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora import apply_lora, lora_parameters, LoRALinear, DEFAULT_TARGETS
from .pstats import p_stats


AMP = [contextlib.nullcontext()]


def set_amp(mode):
    """mode: 'none' (weights carry the compute dtype) or 'bf16' (fp32 master
    weights, bf16 matmuls).  FullFT and LoRA must use the SAME setting."""
    AMP[0] = (torch.autocast("cuda", torch.bfloat16) if mode == "bf16"
              else contextlib.nullcontext())


def load_model(model_id, dtype=torch.bfloat16, device="cuda", attn="sdpa"):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype,
                                             attn_implementation=attn)
    m.to(device)
    m.config.use_cache = False
    return m, tok


@torch.no_grad()
def eval_loss(model, loader, n_batches, device):
    model.eval()
    tot, ntok = 0.0, 0
    for i in range(n_batches):
        b = {k: v.to(device) for k, v in loader.get(i).items()}
        with AMP[0]:
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = out.logits[:, :-1].float()
        labels = b["labels"][:, 1:]
        mask = labels != -100
        ls = nn.functional.cross_entropy(
            logits[mask], labels[mask], reduction="sum")
        tot += float(ls); ntok += int(mask.sum())
    model.train()
    return tot / max(ntok, 1)


@torch.no_grad()
def adapter_diagnostics(adapters, sample_layers=None):
    """Aggregate + per-layer P / movement statistics."""
    names = list(adapters.keys())
    if sample_layers is None:
        sample_layers = names
    agg = dict(dW_norm=0.0, B_norm=0.0, dA_norm=0.0, A_norm=0.0)
    per = {}
    for n in names:
        ad = adapters[n]
        dW = ad.delta_w()
        agg["dW_norm"] += float(dW.pow(2).sum())
        agg["B_norm"] += float(ad.lora_B.pow(2).sum())
        agg["dA_norm"] += float((ad.lora_A - ad.A0).pow(2).sum())
        agg["A_norm"] += float(ad.lora_A.pow(2).sum())
        if n in sample_layers:
            st = p_stats(ad.lora_A.detach().float().cpu(), s=ad.s)
            st.pop("spec_top4", None)
            st["dW_norm"] = float(dW.norm())
            st["rel_A_move"] = float((ad.lora_A - ad.A0).norm() / ad.A0.norm())
            per[n] = st
    for k in agg:
        agg[k] = agg[k] ** 0.5
    return agg, per


def make_optimizer(params, kind, lr, wd=0.0, betas=(0.9, 0.999), eps=1e-8,
                   momentum=0.0, b_lr_ratio=1.0, adapters=None,
                   a_lr_ratio=1.0):
    """b_lr_ratio > 1 reproduces LoRA+ (Hayou et al., ICML 2024).
    a_lr_ratio scales the DOWN-projection's learning rate independently, which
    is the knob that sets how fast Adam rewrites the initial A scaffold:
    ||dA||_F/||A||_F ~ eta_A sqrt(r d) / sqrt(S W), so the persistence timescale
    of the initialisation is tau_A ~ sqrt(S W)/eta_A.  Setting a_lr_ratio = 0
    freezes A entirely (LoRA-FA), which is the cleanest control for whether an
    effect is carried by A-remodelling at all."""
    if (b_lr_ratio != 1.0 or a_lr_ratio != 1.0) and adapters:
        bs = {id(a.lora_B) for a in adapters.values()}
        gb = [p for p in params if id(p) in bs]
        ga = [p for p in params if id(p) not in bs]
        params = [{"params": ga, "lr": lr * a_lr_ratio},
                  {"params": gb, "lr": lr * b_lr_ratio}]
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(kind)


def train(model, adapters, params, train_loader, eval_loader, cfg, device="cuda",
          log_every=10, eval_every=50, eval_batches=16, sample_layers=None,
          callback=None):
    """cfg keys: steps, accum, optimizer, lr, wd, betas, warmup, sched, grad_clip"""
    opt = make_optimizer(params, cfg.get("optimizer", "adamw"), cfg["lr"],
                         wd=cfg.get("wd", 0.0),
                         betas=tuple(cfg.get("betas", (0.9, 0.999))),
                         eps=cfg.get("eps", 1e-8),
                         momentum=cfg.get("momentum", 0.0),
                         b_lr_ratio=cfg.get("b_lr_ratio", 1.0),
                         a_lr_ratio=cfg.get("a_lr_ratio", 1.0),
                         adapters=adapters)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps = cfg["steps"]; accum = cfg.get("accum", 1)
    warmup = cfg.get("warmup", 0)
    sched_kind = cfg.get("sched", "constant")

    def lr_at(t):
        if t < warmup:
            return (t + 1) / max(warmup, 1)
        if sched_kind == "cosine":
            p = (t - warmup) / max(steps - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * p))
        return 1.0

    log = {"step": [], "train_loss": [], "lr": [], "grad_norm": [],
           "eval_step": [], "eval_loss": [], "diag_step": [], "diag": [],
           "per_layer_step": [], "per_layer": []}
    model.train()
    t0 = time.time()
    bi = 0
    for t in range(steps):
        for gparam, bl in zip(opt.param_groups, base_lrs):
            gparam["lr"] = bl * lr_at(t)
        opt.zero_grad(set_to_none=True)
        tot_loss, tot_tok = 0.0, 0
        for _ in range(accum):
            b = {k: v.to(device, non_blocking=True)
                 for k, v in train_loader.get(bi).items()}
            bi += 1
            with AMP[0]:
                out = model(input_ids=b["input_ids"],
                            attention_mask=b["attention_mask"])
            logits = out.logits[:, :-1].float()
            labels = b["labels"][:, 1:]
            mask = labels != -100
            ntok = int(mask.sum())
            loss_sum = nn.functional.cross_entropy(logits[mask], labels[mask],
                                                   reduction="sum")
            (loss_sum / max(ntok, 1) / accum).backward()
            tot_loss += float(loss_sum.detach()); tot_tok += ntok
        gn = float(torch.nn.utils.clip_grad_norm_(
            params, cfg.get("grad_clip", 1e9)))
        opt.step()
        if callback is not None:
            callback(t, model, adapters)
        if t % max(steps // 10, 1) == 0 or t == steps - 1:
            print(f"    step {t}/{steps} loss={tot_loss/max(tot_tok,1):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if t % log_every == 0 or t == steps - 1:
            log["step"].append(t)
            log["train_loss"].append(tot_loss / max(tot_tok, 1))
            log["lr"].append(cfg["lr"] * lr_at(t))
            log["grad_norm"].append(gn)
        if adapters and (t % eval_every == 0 or t == steps - 1):
            agg, per = adapter_diagnostics(adapters, sample_layers)
            log["diag_step"].append(t); log["diag"].append(agg)
            if per:
                log["per_layer_step"].append(t); log["per_layer"].append(per)
        if (t % eval_every == 0 and t > 0) or t == steps - 1:
            el = eval_loss(model, eval_loader, eval_batches, device)
            log["eval_step"].append(t); log["eval_loss"].append(el)
    log["wall_time"] = time.time() - t0
    log["final_eval_loss"] = eval_loss(model, eval_loader, eval_batches, device)
    return log
