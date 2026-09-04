"""The exact staircase: GroupAdam_b is blind to O(k) rotations iff k <= b.

This is a theorem check, not an experiment, so it runs in float64 on a synthetic
problem where the gauge structure is exact and nothing else can contaminate it:

    L(A, B) = 1/2 || s B A - T ||_F^2,     G^A = s B^T R,  G^B = s R A^T,
                                           R = s B A - T.

L depends on (A, B) only through the product BA, so it is exactly gauge
invariant, and the gradients transform as G^A -> Q G^A, G^B -> G^B Q^T, which is
how they transform in the real model too.  Running the same optimizer from
(A0, B0) and from (Q A0, B0 Q^T) therefore isolates the optimizer's own
equivariance with no model, no data and no float32 noise.

Prediction, registered before running: with contiguous power-of-two blocks a
rotation confined to k-blocks lies inside H_b = O(b)^{r/b} exactly when k <= b,
so the loss trajectories must agree to the float64 floor there and diverge for
k > b.  A monotone "more averaging = smoother" story predicts no such sharp
dependence on k, which is what makes the staircase the identifying test rather
than the b axis alone.

Also checks GroupAdam_1 == torch.optim.AdamW to machine precision, which is what
lets the ranking phase diagram claim it never leaves one optimizer family.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from common.groupadam import GroupAdam, block_rotation


def grads(A, B, T, s):
    R = s * (B @ A) - T
    return s * (B.T @ R), s * (R @ A.T), 0.5 * float((R * R).sum())


def run(A0, B0, T, s, b, steps, lr, betas, eps):
    """Trajectory of GroupAdam_b from a given start; returns losses and A path."""
    A = A0.clone().requires_grad_(False)
    B = B0.clone().requires_grad_(False)
    opt = GroupAdam([A, B], lr=lr, b=b, betas=betas, eps=eps, wd=0.0)
    losses, path = [], []
    for _ in range(steps):
        gA, gB, L = grads(A, B, T, s)
        A.grad, B.grad = gA, gB
        opt.step()
        losses.append(L)
        path.append(A.clone())
    return losses, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--d_in", type=int, default=64)
    ap.add_argument("--d_out", type=int, default=48)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dt = torch.float64
    r, s = a.r, 2.0
    blocks = [k for k in (1, 2, 4, 8, 16) if k <= r and r % k == 0]

    A0 = torch.randn(r, a.d_in, dtype=dt) / a.d_in ** 0.5
    B0 = torch.randn(a.d_out, r, dtype=dt) * 0.0 + \
        torch.randn(a.d_out, r, dtype=dt) * 1e-2
    T = torch.randn(a.d_out, a.d_in, dtype=dt) * 0.1
    betas = (0.9, 0.999)

    # --- anchor: GroupAdam_1 must equal torch AdamW ---------------------------
    Aa, Ba = A0.clone(), B0.clone()
    ref = torch.optim.AdamW([Aa, Ba], lr=a.lr, betas=betas, eps=a.eps,
                            weight_decay=0.0)
    ref_losses = []
    for _ in range(a.steps):
        gA, gB, L = grads(Aa, Ba, T, s)
        Aa.grad, Ba.grad = gA, gB
        ref.step()
        ref_losses.append(L)
    g1_losses, _ = run(A0, B0, T, s, 1, a.steps, a.lr, betas, a.eps)
    anchor = max(abs(x - y) / max(abs(y), 1e-300)
                 for x, y in zip(g1_losses, ref_losses))
    print(f"anchor  GroupAdam_1 vs torch.AdamW : max rel loss dev {anchor:.3e}"
          f"   {'OK' if anchor < 1e-12 else 'MISMATCH'}\n")

    # --- the staircase --------------------------------------------------------
    base = {b: run(A0, B0, T, s, b, a.steps, a.lr, betas, a.eps)
            for b in blocks}
    grid = {}
    print("max relative loss deviation between (A0,B0) and (Q_k A0, B0 Q_k^T)")
    print("rows = GroupAdam b, cols = rotation block k;  "
          "theorem: floor iff k <= b\n")
    hdr = "  b\\k " + "".join(f"{k:>12d}" for k in blocks)
    print(hdr)
    for b in blocks:
        L0, P0 = base[b]
        row = f"{b:>5d} "
        for k in blocks:
            g = torch.Generator().manual_seed(1000 + k)
            Q = block_rotation(r, k, dtype=dt, generator=g)
            Lq, Pq = run(Q @ A0, B0 @ Q.T, T, s, b, a.steps, a.lr, betas, a.eps)
            dev = max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(Lq, L0))
            # parameter-level check: A'_t should equal Q A_t
            pdev = max(float((pq - Q @ p0).norm() / p0.norm().clamp_min(1e-300))
                       for p0, pq in zip(P0, Pq))
            grid[(b, k)] = dict(loss_dev=dev, param_dev=pdev,
                                predicted="invisible" if k <= b else "visible")
            row += f"{dev:>12.2e}"
        print(row)

    print("\nverdict per cell (floor = rel dev < 1e-10):")
    ok = True
    for b in blocks:
        line = f"{b:>5d} "
        for k in blocks:
            at_floor = grid[(b, k)]["loss_dev"] < 1e-10
            want = (k <= b)
            good = (at_floor == want)
            ok &= good
            line += f"{('0' if at_floor else 'X'):>12s}" if good else \
                    f"{('0' if at_floor else 'X') + '!':>12s}"
        print(line)
    print(f"\nstaircase {'HOLDS exactly as predicted' if ok else 'VIOLATED'}"
          f"   (0 = invisible/at floor, X = visible, ! = contradicts theorem)")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(args=vars(a), anchor_rel_dev=anchor, staircase_ok=ok,
                       blocks=blocks,
                       grid={f"b{b}_k{k}": v for (b, k), v in grid.items()}),
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
