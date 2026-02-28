from typing import Iterable
from typing import Callable
from jaxtyping import Int, Float
import torch
import math


def cross_entropy(
    inputs: Float[torch.Tensor, " batch vocab_size"],
    targets: Int[torch.Tensor, " batch"],
) -> Float[torch.Tensor, ""]:
    # max for stability
    logit_max = torch.max(inputs, dim=-1)[0].unsqueeze(-1)  # (batch, 1)
    log_sum_exp = torch.sum(torch.exp(inputs - logit_max), dim=-1, keepdim=True).log() + logit_max  # (batch, 1)

    batch_size = inputs.size(0)
    target_logits = inputs[torch.arange(batch_size, device=inputs.device), targets]  # (batch,)
    loss = -target_logits + log_sum_exp.squeeze(-1)
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        super(AdamW, self).__init__(params, {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps})

    def step(self, closure: Callable | None = None) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data
                state = self.state[p]
                # initialize state on first step
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # update rule
                state["t"] += 1
                state["m"] = group["betas"][0] * state["m"] + (1 - group["betas"][0]) * g
                state["v"] = group["betas"][1] * state["v"] + (1 - group["betas"][1]) * g**2
                lr_t = (
                    group["lr"] * math.sqrt(1 - group["betas"][1] ** state["t"]) / (1 - group["betas"][0] ** state["t"])
                )
                # update params
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + group["eps"])
                # apply weight decay
                p.data -= group["lr"] * group["weight_decay"] * p.data


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / float(warmup_iters) * max_learning_rate

    if it < cosine_cycle_iters:
        cosine = math.cos(math.pi * (it - warmup_iters) / float(cosine_cycle_iters - warmup_iters))
        return min_learning_rate + 0.5 * (1 + cosine) * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = list(parameters)
    # compute combined L2 norm across all parameters
    grad_norms_sq = [torch.sum(p.grad.data**2) for p in params if p.grad is not None]
    if not grad_norms_sq:
        return
    total_norm = torch.sqrt(torch.stack(grad_norms_sq).sum())
    # scale all gradients uniformly if combined norm exceeds limit
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in params:
            if p.grad is not None:
                p.grad.data *= scale
