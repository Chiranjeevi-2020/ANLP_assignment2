"""
Part 2: Custom Optimizers from Scratch
Strictly subclassing torch.optim.Optimizer without predefined torch.optim modules.
Covers all required categories from 'Fantastic Pretraining Optimizers and Where to Find Them' (2025):
1. Category 1: AdamW (Decoupled weight decay)
2. Category 2: Adan (Adaptive Nesterov Momentum - Variance-Reduced AdamW)
3. Category 3: Lion (Memory-Efficient Sign-Momentum)
4. Category 4: Muon (Matrix-Based Polar Newton-Schulz Optimizer)
5. Bonus: Sophia (Hessian-Based Clipped Stochastic Optimizer)
"""

import math
from typing import Callable, Iterable, Optional, Tuple, Union, Dict, Any
import torch
from torch.optim import Optimizer


class CustomAdamW(Optimizer):
    """
    Category 1: AdamW with decoupled weight decay (Loshchilov & Hutter, 2019).
    theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CustomAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 1. Perform decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 2. Update biased 1st and 2nd moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class CustomAdan(Optimizer):
    """
    Category 2: Adan (Adaptive Nesterov Momentum Algorithm, Xie et al., 2022).
    Variance-reduced AdamW variant incorporating gradient differences to estimate
    future gradient trajectory and damp oscillations.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.02
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CustomAdan does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)           # m_k: EMA of grad
                    state["exp_avg_diff"] = torch.zeros_like(p)      # v_k: EMA of grad diff
                    state["exp_avg_sq"] = torch.zeros_like(p)        # n_k: second moment
                    state["prev_grad"] = grad.clone()

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_diff = state["exp_avg_diff"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                b1 = beta1 if beta1 > 0.5 else 1.0 - beta1
                b2 = beta2 if beta2 > 0.5 else 1.0 - beta2
                b3 = beta3 if beta3 > 0.5 else 1.0 - beta3

                grad_diff = grad - prev_grad

                # Update 1st moment of grad
                exp_avg.mul_(b1).add_(grad, alpha=1.0 - b1)

                # Update 1st moment of grad difference
                exp_avg_diff.mul_(b2).add_(grad_diff, alpha=1.0 - b2)

                # Update 2nd moment of update vector
                update_term = grad + (1.0 - b2) * grad_diff
                exp_avg_sq.mul_(b3).addcmul_(update_term, update_term, value=1.0 - b3)

                bias_correction1 = 1.0 - b1 ** step
                bias_correction2 = 1.0 - b2 ** step
                bias_correction3 = 1.0 - b3 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_diff / bias_correction2
                n_hat = exp_avg_sq / bias_correction3

                denom = n_hat.sqrt().add_(eps)
                step_direction = (m_hat + (1.0 - b2) * v_hat) / denom

                # Apply parameter update and decoupled weight decay
                p.data.add_(step_direction, alpha=-lr)
                if wd != 0.0:
                    p.data.div_(1.0 + lr * wd)

                prev_grad.copy_(grad)

        return loss


class CustomLion(Optimizer):
    """
    Category 3: Lion (EvoLved Sign Momentum, Chen et al., Google Brain 2023).
    Memory-efficient optimizer tracking only 1 momentum buffer (half the memory of AdamW).
    Uses coordinate-wise sign operation for uniform step magnitude.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CustomLion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]

                # 1. Weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 2. Lion update direction: c_t = beta1 * m_{t-1} + (1 - beta1) * grad
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1)
                p.add_(torch.sign(update), alpha=-lr)

                # 3. Update momentum buffer: m_t = beta2 * m_{t-1} + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Quintic Newton-Schulz iteration for matrix orthogonalization (Muon).
    Computes an approximate polar decomposition (orthogonal projection of G).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    X /= (X.norm() + eps)

    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


class CustomMuon(Optimizer):
    """
    Category 4: Muon (Momentum Orthogonalized by Newton-Schulz, 2024/2025).
    Matrix-based optimizer. Applies orthogonalized momentum to 2D weight matrices,
    while falling back to standard AdamW for 1D vectors and non-matrix parameters.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        fallback_lr: float = 1e-3
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            fallback_lr=fallback_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            fallback_lr = group["fallback_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                is_matrix = (p.ndim == 2 and min(p.shape) >= 2)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    if not is_matrix:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                if is_matrix:
                    # 2D Matrix parameter: Muon Newton-Schulz update
                    exp_avg = state["exp_avg"]
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    exp_avg.mul_(mu).add_(grad)
                    if nesterov:
                        buf = grad + mu * exp_avg
                    else:
                        buf = exp_avg

                    ortho_buf = newton_schulz5(buf, steps=ns_steps)
                    # RMS-matched scaling: scales update by sqrt(max(n, m)) to match unit RMS
                    scale = max(1.0, float(max(p.shape[-2:])) ** 0.5)
                    p.add_(ortho_buf, alpha=-lr * scale)
                else:
                    # 1D/Non-matrix parameter: Standard AdamW fallback
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    eps = group["eps"]

                    if wd != 0.0:
                        p.mul_(1.0 - fallback_lr * wd)

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = fallback_lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class CustomSophia(Optimizer):
    """
    Bonus: Sophia (Second-order Clipped Stochastic Optimization, Liu et al., 2023).
    Hessian-based optimizer utilizing EMA of diagonal Hessian estimates and coordinate-wise clipping.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.96, 0.99),
        rho: float = 1.0,
        gamma: float = 0.01,
        eps: float = 1e-12,
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def update_hessian(self) -> None:
        """
        Updates diagonal Hessian estimate using Hutchinson's estimator or gradient-squared approximation.
        """
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                    state["hessian_step"] = 0
                state["hessian_step"] += 1
                state["hessian_called_externally"] = True
                # Diagonal Hessian approximation via stochastic gradient outer product
                h_hat = p.grad.pow(2)
                state["hessian"].mul_(beta2).add_(h_hat, alpha=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            gamma = group["gamma"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CustomSophia does not support sparse gradients")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                    state["hessian_step"] = 0

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                # 1. Weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 2. First moment update
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                m_hat = exp_avg / bias_correction1

                if state.get("hessian_called_externally", False):
                    h_step = max(1, state.get("hessian_step", 1))
                    bias_correction2 = 1.0 - beta2 ** h_step
                    h_hat = hessian / bias_correction2
                else:
                    # Dynamically update diagonal Hessian estimate from grad^2 each step
                    hessian.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)
                    bias_correction2 = 1.0 - beta2 ** step
                    h_hat = hessian / bias_correction2

                # 4. Scaled and clipped update step (Sophia: clip(m_hat / max(h_hat, gamma), rho))
                denom = torch.clamp(h_hat, min=gamma)
                scaled_step = torch.clamp(m_hat / denom, min=-rho, max=rho)

                p.add_(scaled_step, alpha=-lr)

        return loss
