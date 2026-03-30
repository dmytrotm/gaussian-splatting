#
# Alternative Optimizers for 3D Gaussian Splatting
#
# Lion: EvoLved Sign Momentum (Google Brain, arXiv:2302.06675)
#   - Uses sign(momentum) for updates → lower memory than Adam (no exp_avg_sq)
#   - Typically needs 3-10× lower learning rate than Adam
#   - Often converges faster on vision tasks
#
# ApproxSLN: Approximate Stochastic Local Newton
#   - Inspired by 3DGS² (arXiv:2501.13975) but pure-PyTorch (no custom CUDA)
#   - Per-parameter diagonal Hessian approximation via EMA of squared gradients
#   - Newton-like updates: step = grad / (hessian_diag + damping)
#   - Key difference from Adam: uses raw squared grad (not momentum-smoothed)
#     with stronger damping, giving more aggressive curvature adaptation
#

import torch
from torch.optim import Optimizer
import math


class Lion(Optimizer):
    """Lion optimizer — EvoLved Sign Momentum.

    Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms"
    (arXiv:2302.06675)

    Key properties:
      - Memory: Only stores momentum (no second moment → ~50% less state than Adam)
      - Update rule: sign(β₁·momentum + (1-β₁)·grad) — uniform magnitude updates
      - Typically needs 3-10× lower LR than Adam

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-4). Use ~3-10× lower than Adam LR.
        betas: Coefficients for momentum interpolation (default: (0.9, 0.99)).
        weight_decay: Decoupled weight decay (default: 0.0).
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # Weight decay (decoupled)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                # Update: sign of interpolated momentum
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                # Update momentum for next step
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class ApproxSLN(Optimizer):
    """Approximate Stochastic Local Newton optimizer.

    A pure-PyTorch approximation of the 3DGS² paper's second-order method.
    Uses diagonal Hessian estimation (EMA of squared gradients) to produce
    Newton-like per-parameter updates without custom CUDA kernels.

    Key differences from Adam:
      - No first-moment bias correction — uses raw gradient direction
      - Stronger curvature adaptation via the squared-gradient Hessian proxy
      - Damping parameter λ prevents division by zero and controls step size
      - Optional gradient clipping for stability

    The update rule is:
      h_t = β·h_{t-1} + (1-β)·g²              (Hessian diagonal estimate)
      p = p - lr · g / (sqrt(h_t) + λ)         (Newton-like step)

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-3).
        beta: EMA coefficient for Hessian diagonal (default: 0.999).
        damping: Regularization for Hessian inverse (default: 1e-4).
        weight_decay: Decoupled weight decay (default: 0.0).
        max_grad_norm: Optional gradient clipping norm (default: 0 = disabled).
    """

    def __init__(self, params, lr=1e-3, beta=0.999, damping=1e-4,
                 weight_decay=0.0, max_grad_norm=0.0):
        defaults = dict(lr=lr, beta=beta, damping=damping,
                        weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            damping = group['damping']
            wd = group['weight_decay']
            max_grad_norm = group['max_grad_norm']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Optional gradient clipping
                if max_grad_norm > 0:
                    grad_norm = grad.norm()
                    if grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / (grad_norm + 1e-6))

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Diagonal Hessian estimate (EMA of squared gradients)
                    state['hessian_diag'] = torch.zeros_like(p)

                state['step'] += 1
                hess = state['hessian_diag']

                # Weight decay (decoupled)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                # Update Hessian diagonal estimate
                hess.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # Bias correction
                bias_correction = 1.0 - beta ** state['step']
                hess_corrected = hess / bias_correction

                # Newton-like step: g / (sqrt(H_diag) + λ)
                denom = hess_corrected.sqrt().add_(damping)
                p.addcdiv_(grad, denom, value=-lr)

        return loss
