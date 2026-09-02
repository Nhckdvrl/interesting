"""A matrix preconditioner on the low-rank side.

The frame ladder shows AdamW responds to LoRA's gauge frame and SGD and Muon do
not.  The rule we first wrote was about norm geometry -- Frobenius and spectral
are orthogonally invariant, the elementwise max is not.  There is a sharper one.

Take grad_B (d_out x r) and precondition it on the r side with the inverse root
of R = E[grad_B^T grad_B] (r x r).  Under the gauge grad_B -> grad_B Q^T,

    R -> Q R Q^T,   R^(-1/2) -> Q R^(-1/2) Q^T,
    grad_B Q^T (Q R Q^T)^(-1/2) = grad_B R^(-1/2) Q^T,

so the update transforms exactly like the gradient and the whole trajectory is
gauge-covariant.  The same holds on the other factor with grad_A -> Q grad_A
preconditioned on the left by L = E[grad_A grad_A^T].  Verified to 2.9e-13.

So it is not the norm that decides, it is the STRUCTURE OF THE PRECONDITIONER:

    none (SGD)          covariant
    orthogonalised (Muon)   covariant
    full-matrix (Shampoo, and this)  covariant
    DIAGONAL (Adam)     not covariant

The frame is visible exactly when the preconditioner is diagonal.  That also
says what LoRA-RITE (ICLR 2025) is doing: it puts a transformation-invariant
matrix preconditioner on the low-rank side, which is precisely the structure
that removes the frame dependence.  Both preconditioners here are r x r, so
this costs almost nothing at LoRA ranks.
"""
import torch


def _inv_root(M, p=0.5, eps=1e-12):
    w, V = torch.linalg.eigh(M.double())
    w = w.clamp_min(eps * float(w.max()).__abs__() if float(w.max()) > 0 else eps)
    return (V @ torch.diag(w.pow(-p)) @ V.T).to(M.dtype)


class MatrixPrecond(torch.optim.Optimizer):
    """Adam-style moments with a full r x r preconditioner instead of a
    diagonal one, applied on whichever side is small.

    Parameters that are not 2-D, or whose smaller dimension exceeds `max_side`,
    fall back to plain momentum -- preconditioning them would need a matrix as
    large as the layer.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, wd=0.0,
                 eps=1e-12, max_side=256, update_every=1):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, wd=wd,
                                      eps=eps, max_side=max_side,
                                      update_every=update_every))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["beta1"], g["beta2"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                st = self.state[p]
                st.setdefault("t", 0)
                st["t"] += 1
                if "m" not in st:
                    st["m"] = torch.zeros_like(d)
                st["m"].mul_(b1).add_(d, alpha=1 - b1)
                mhat = st["m"] / (1 - b1 ** st["t"])
                if d.ndim == 2 and min(d.shape) <= g["max_side"]:
                    right = d.shape[1] <= d.shape[0]     # precondition small side
                    C = (d.T @ d) if right else (d @ d.T)
                    if "C" not in st:
                        st["C"] = torch.zeros_like(C)
                    st["C"].mul_(b2).add_(C, alpha=1 - b2)
                    Chat = st["C"] / (1 - b2 ** st["t"])
                    n = Chat.shape[0]
                    Chat = Chat + g["eps"] * float(torch.diagonal(Chat).mean()
                                                   + 1e-30) * torch.eye(
                        n, device=d.device, dtype=d.dtype)
                    if st["t"] % g["update_every"] == 1 or "P" not in st \
                            or g["update_every"] == 1:
                        st["P"] = _inv_root(Chat, 0.5)
                    u = (mhat @ st["P"]) if right else (st["P"] @ mhat)
                else:
                    u = mhat
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"])
        return loss
