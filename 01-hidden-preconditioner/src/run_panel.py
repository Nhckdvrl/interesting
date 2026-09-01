"""Topic 01 -- matched-control panel runner.

One process = one (condition, lr, seed) cell.  Results are appended as JSON to
results/<tag>/<cell>.json so the panel is restartable and parallelisable.
"""
import argparse, json, os, sys, math, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from common.lora import apply_lora, lora_parameters, DEFAULT_TARGETS
from common.pinit import make_A, kaiming_A
from common.pstats import p_stats
from common.data import build_sft, FixedOrderLoader
from common.train import load_model, train, eval_loss

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def build_factory(kind, r, seed, gauge_seed=0, device="cpu"):
    """Returns a_factory.  Every layer gets its own generator seeded by
    (seed, layer name) so that conditions sharing `seed` share the *same*
    underlying random draw -- that is what makes the controls matched."""
    stats_store = {}

    def factory(name, r_, d_in, d_out):
        h = int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest()[:12], 16)
        g = torch.Generator(device=device).manual_seed(h)
        base = kaiming_A(r_, d_in, g, device)
        if kind == "kaiming":
            A = base
        elif kind == "left_gauge":
            g2 = torch.Generator(device=device).manual_seed(
                int(hashlib.md5(f"gauge{gauge_seed}:{name}".encode()).hexdigest()[:12], 16))
            A = make_A("left_gauge", r_, d_in, g2, device, ref_A=base)
        else:
            A = make_A(kind, r_, d_in, g, device, ref_A=base)
        st = p_stats(A, s=1.0)
        st.pop("spec_top4", None)
        stats_store[name] = st
        return A.float()

    factory.stats = stats_store
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gauge_seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--scaling", default="standard")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sched", default="constant")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--train_A", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cell = (f"{args.cond}_lr{args.lr:g}_s{args.seed}"
            + (f"_g{args.gauge_seed}" if args.cond == "left_gauge" else "")
            + (f"_{args.optimizer}" if args.optimizer != "adamw" else "")
            + ("" if args.train_A else "_frozenA"))
    outdir = args.out or os.path.join(REPO, "01-hidden-preconditioner", "results", args.tag)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, cell + ".json")
    if os.path.exists(outfile):
        print("skip (exists)", outfile); return

    torch.manual_seed(args.seed)
    model, tok = load_model(args.model)
    tr, te = build_sft(tok, args.task, args.n_train, args.n_eval,
                       args.max_len, seed=args.data_seed)
    trl = FixedOrderLoader(tr, args.bs, tok.pad_token_id, seed=args.data_seed)
    tel = FixedOrderLoader(te, args.bs, tok.pad_token_id, seed=999)

    fac = build_factory(args.cond, args.r, args.seed, args.gauge_seed)
    adapters = apply_lora(model, args.r, args.alpha, fac,
                          targets=tuple(args.targets.split(",")),
                          scaling=args.scaling, train_A=bool(args.train_A))
    params = lora_parameters(adapters)
    ntr = sum(p.numel() for p in params)
    print(f"[{cell}] {len(adapters)} adapters, {ntr/1e6:.2f}M trainable params")

    base_eval = eval_loss(model, tel, args.eval_batches, "cuda")
    cfg = dict(steps=args.steps, accum=args.accum, optimizer=args.optimizer,
               lr=args.lr, wd=args.wd, warmup=args.warmup, sched=args.sched,
               grad_clip=args.grad_clip, momentum=args.momentum)
    sample = [n for n in adapters if n.endswith(("layers.0.self_attn.q_proj",
                                                 "layers.13.mlp.down_proj",
                                                 "layers.27.self_attn.v_proj"))]
    log = train(model, adapters, params, trl, tel, cfg, log_every=5,
                eval_every=args.eval_every, eval_batches=args.eval_batches,
                sample_layers=sample)
    rec = dict(cell=cell, args=vars(args), base_eval_loss=base_eval,
               init_pstats={k: v for k, v in list(fac.stats.items())},
               n_trainable=ntr, log=log)
    json.dump(rec, open(outfile, "w"))
    print(f"[{cell}] base={base_eval:.4f} final_eval={log['final_eval_loss']:.4f} "
          f"({log['wall_time']:.0f}s)")


if __name__ == "__main__":
    main()
