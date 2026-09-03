"""The symmetry hierarchy: which subgroup of GL(r) does each optimizer see?

LoRA's factorisation ambiguity is GL(r): (A, B) -> (SA, B S^{-1}) leaves BA
fixed for any invertible S.  By polar decomposition it factors into a ROTATION
part O(r) and a SCALING part, and optimizers do not treat the two alike.

This matters because a concurrent ICLR 2026 paper (LoRA meets Riemannion)
motivates its construction by asserting that per-factor Muon is
"non-reparameterization-invariant: its per-factor orthogonalization depends on
arbitrary scalings OR ROTATIONS".  The scaling half is right; the rotation half
is not, and the difference is the whole point -- it is exactly the rotation part
that AdamW alone can see.

Run:  .venv/bin/python 01-hidden-preconditioner/src/hierarchy.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from common.muon import Muon
from common.matprec import MatPrecAdam, Lion

R, DIN, DOUT, STEPS = 8, 64, 48, 25


def trajectory(A0, B0, kind, X, T, lr=1e-2):
    A = A0.clone().requires_grad_(True)
    B = B0.clone().requires_grad_(True)
    opt = {"sgd": lambda: torch.optim.SGD([A, B], lr=lr),
           "adamw": lambda: torch.optim.AdamW([A, B], lr=lr, weight_decay=0.0),
           "muon": lambda: Muon([A, B], lr=lr, momentum=0.9),
           "matprec": lambda: MatPrecAdam([A, B], lr=lr),
           "lion": lambda: Lion([A, B], lr=lr),
           # more diagonal methods: all predicted to sit with AdamW and Lion
           "rmsprop": lambda: torch.optim.RMSprop([A, B], lr=lr),
           "adagrad": lambda: torch.optim.Adagrad([A, B], lr=lr),
           "adadelta": lambda: torch.optim.Adadelta([A, B], lr=lr),
           # momentum-only: no per-coordinate scaling at all, predicted to be
           # exactly O(r)-covariant like plain SGD
           "sgdm": lambda: torch.optim.SGD([A, B], lr=lr, momentum=0.9),
           }[kind]()
    out = []
    for _ in range(STEPS):
        loss = ((X @ (B @ A).T - T) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        out.append(float(loss.detach()))
    return out


LRS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def best_lr(A0, B0, kind, X, T):
    """Each optimizer gets its own learning rate.

    Without this the table is not honest: at a shared lr, SGD and Adadelta
    barely move, and an optimizer that does not move looks perfectly invariant
    for a reason that has nothing to do with symmetry.  Every row is therefore
    measured where that optimizer actually trains.
    """
    best, blr = None, LRS[0]
    for lr in LRS:
        try:
            t = trajectory(A0, B0, kind, X, T, lr=lr)
        except Exception:
            continue
        v = t[-1]
        if v == v and (best is None or v < best):
            best, blr = v, lr
    return blr


def main(seed=0):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    X, T = torch.randn(256, DIN), torch.randn(256, DOUT)
    A0 = torch.randn(R, DIN) / DIN ** 0.5
    B0 = torch.randn(DOUT, R) * 0.05          # nonzero, so both factors move
    Q = torch.linalg.qr(torch.randn(R, R))[0]
    S = torch.eye(R) + 0.3 * torch.randn(R, R); S = S @ S.T
    P = torch.eye(R)[torch.randperm(R)] * torch.sign(torch.randn(R, 1))
    acts = (("signed perms", P @ A0, B0 @ P.T),
            ("O(r) rotation", Q @ A0, B0 @ Q.T),
            ("GL(r) scaling", S @ A0, B0 @ torch.linalg.inv(S)))
    print("max |loss(transformed) - loss(original)| over "
          f"{STEPS} steps, float64, B_0 != 0\n")
    print("Signed permutations are EXACTLY invariant for every optimizer here,")
    print("so that column is each optimizer's own float-noise floor at its own")
    print("learning rate -- which is what a chaotic trajectory near the")
    print("stability edge inflates.  The O(r) column is reported as a RATIO to")
    print("that floor, so the comparison is not contaminated by how close an")
    print("optimizer runs to divergence.\n")
    print(f"{'optimizer':>26s} {'floor (perm)':>14s} {'O(r) raw':>11s} "
          f"{'O(r)/floor':>11s} {'GL(r)/floor':>12s} {'loss drop':>10s} "
          f"{'lr':>7s}")
    for kind, label in (("sgd", "SGD"), ("sgdm", "SGD + momentum"),
                        ("muon", "Muon"),
                        ("matprec", "matrix-precond Adam"),
                        ("adamw", "AdamW"), ("lion", "Lion (sign descent)"),
                        ("rmsprop", "RMSprop"), ("adagrad", "Adagrad"),
                        ("adadelta", "Adadelta")):
        lr = best_lr(A0, B0, kind, X, T)
        base = trajectory(A0, B0, kind, X, T, lr=lr)
        moved = base[0] - base[-1]      # did this optimizer actually train?
        row = []
        for _, Ai, Bi in acts:
            v = trajectory(Ai, Bi, kind, X, T, lr=lr)
            d = max(abs(a - b) for a, b in zip(base, v))
            row.append(d)
        floor = max(row[0], 1e-16)
        flag = "" if moved > 0.05 * base[0] else "  <- barely moves"
        print(f"{label:>26s} {row[0]:14.2e} {row[1]:11.2e} "
              f"{row[1]/floor:11.1e} {row[2]/floor:12.1e} {moved:10.4f} "
              f"{lr:7.0e}{flag}")
    print("\n(An `inf` in the GL(r) column means the transformed run diverged "
          "where\n the original did not -- non-invariance in its loudest form, "
          "not a bug.)")
    print("\nRead the O(r)/floor column: methods with no preconditioner, an")
    print("orthogonalised one, or a full matrix on the rank index sit at ~1")
    print("(indistinguishable from their own float noise); diagonal methods sit")
    print("many orders above it.\n")
    print("The split is not adaptivity and not the norm.  It is whether the")
    print("preconditioner is DIAGONAL in the coordinates the gauge acts on:")
    print("  no preconditioner (SGD, SGD+m), orthogonalised (Muon), or a")
    print("  full matrix on the rank index  ->  blind to O(r)")
    print("  diagonal (AdamW, Lion, RMSprop, Adagrad, Adadelta)  ->  sees it\n")
    print("Every optimizer is blind inside its group and sensitive outside it:")
    print("  signed perms  <  O(r)  <  GL(r)")
    print("     AdamW         SGD       LoRA-RITE")
    print("                   Muon      Riemannion")
    print("                   matrix-precond Adam")
    print("\nSo what an initialisation IS -- how many degrees of freedom it has")
    print("-- depends on which optimizer will consume it.  AdamW is the one")
    print("with the smallest symmetry group, and the one everybody uses.")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
