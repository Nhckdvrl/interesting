"""Adam with a full-matrix second moment contracted on the rank index.

This exists to sharpen the paper's rule.  "The frame is visible iff the
optimizer's norm is not orthogonally invariant" is true but not the tightest
statement; the tighter one is about the PRECONDITIONER's shape:

    none              (SGD)     covariant       measured  1e-5 nats
    orthogonalised    (Muon)    covariant       measured  2.6e-4
    full-matrix on r            covariant       this file
    DIAGONAL          (AdamW)   NOT covariant   measured  2.2e-3

Adam's second moment is the only diagonal one in that list and the only one
that sees the frame.  So the property that matters is not adaptivity, and not
even the norm -- it is whether the preconditioner is diagonal in the coordinates
the gauge acts on.

Why the matrix version is exactly covariant.  For B (d_out x r) the second
moment is contracted on the rank index, V = E[grad_B^T grad_B] (r x r), and the
step is grad_B V^{-1/2}.  Under the gauge grad_B -> grad_B Q^T:

    V -> Q V Q^T,  V^{-1/2} -> Q V^{-1/2} Q^T,
    step -> grad_B Q^T Q V^{-1/2} Q^T = (grad_B V^{-1/2}) Q^T,

which is the transformed step.  For A (r x d_in), U = E[grad_A grad_A^T] and the
step is U^{-1/2} grad_A; under grad_A -> Q grad_A the same cancellation gives
Q U^{-1/2} grad_A.  Exact, at every step, with momentum.

This is the structure LoRA-RITE (ICLR 2025) uses, so if the frame ladder comes
out flat under this optimizer, our account says what LoRA-RITE's transformation
invariance is buying: it removes a coordinate that Adam was responding to.
"""
import torch


@torch.no_grad()
def _inv_sqrt(V, eps):
    """V^{-1/2} for a symmetric PSD V, with an eigenvalue floor."""
    lam, U = torch.linalg.eigh(V.double())
    lam = lam.clamp_min(0)
    floor = eps * float(lam.max()) if float(lam.max()) > 0 else eps
    return (U * (lam + floor).rsqrt()) @ U.T


class MatPrecAdam(torch.optim.Optimizer):
    """Adam whose second moment is a matrix on the short (rank) side.

    Non-2-D parameters fall back to ordinary diagonal Adam, which is what the
    reference low-rank optimizers do too; in the experiments here only the LoRA
    factors are trainable, so every parameter takes the matrix path.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(d)
                    if d.ndim == 2:
                        r = min(d.shape)
                        st["V"] = torch.zeros(r, r, dtype=torch.float64,
                                              device=d.device)
                    else:
                        st["v"] = torch.zeros_like(d)
                st["t"] += 1
                t = st["t"]
                st["m"].mul_(b1).add_(d, alpha=1 - b1)
                mh = st["m"] / (1 - b1 ** t)
                if d.ndim == 2:
                    dd = d.double()
                    # contract on the SHORT side: that is the rank index, and
                    # the rank index is the one the gauge acts on
                    tall = d.shape[0] >= d.shape[1]      # d_out x r  (B-like)
                    M = (dd.T @ dd) if tall else (dd @ dd.T)
                    st["V"].mul_(b2).add_(M, alpha=1 - b2)
                    Vh = _inv_sqrt(st["V"] / (1 - b2 ** t), g["eps"])
                    upd = (mh.double() @ Vh) if tall else (Vh @ mh.double())
                    upd = upd.to(d.dtype)
                else:
                    st["v"].mul_(b2).addcmul_(d, d, value=1 - b2)
                    upd = mh / ((st["v"] / (1 - b2 ** t)).sqrt() + g["eps"])
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(upd, alpha=-g["lr"])
        return loss
