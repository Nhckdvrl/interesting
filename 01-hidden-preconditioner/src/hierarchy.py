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
           "lion": lambda: Lion([A, B], lr=lr)}[kind]()
    out = []
    for _ in range(STEPS):
        loss = ((X @ (B @ A).T - T) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        out.append(float(loss.detach()))
    return out


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
    print(f"{'optimizer':>26s} " + " ".join(f"{n:>15s}" for n, _, _ in acts))
    for kind, label in (("sgd", "SGD"), ("muon", "Muon"),
                        ("matprec", "matrix-precond Adam"), ("adamw", "AdamW"),
                        ("lion", "Lion (sign descent)")):
        base = trajectory(A0, B0, kind, X, T)
        row = []
        for _, Ai, Bi in acts:
            v = trajectory(Ai, Bi, kind, X, T)
            d = max(abs(a - b) for a, b in zip(base, v))
            row.append(d)
        print(f"{label:>26s} " + " ".join(f"{x:15.2e}" for x in row))
    print("\n(An `inf` in the GL(r) column means the transformed run diverged "
          "where\n the original did not -- non-invariance in its loudest form, "
          "not a bug.)")
    print("\nEvery optimizer is blind inside its group and sensitive outside it:")
    print("  signed perms  <  O(r)  <  GL(r)")
    print("     AdamW         SGD       LoRA-RITE")
    print("                   Muon      Riemannion")
    print("                   matrix-precond Adam")
    print("\nSo what an initialisation IS -- how many degrees of freedom it has")
    print("-- depends on which optimizer will consume it.  AdamW is the one")
    print("with the smallest symmetry group, and the one everybody uses.")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
