"""Training-only GradNorm weights, following Chen et al. (ICML 2018)."""

import math

import torch
from torch import nn

from .unified_training import TASKS


GRADNORM_STAGE = "shared_scene_gradnorm_v1"


class TaskGradNorm(nn.Module):
    def __init__(self, reference_losses, alpha=1.5):
        super().__init__()
        reference = torch.tensor([float(reference_losses[k]) for k in TASKS], dtype=torch.float32)
        if (not bool((torch.isfinite(reference) & (reference > 0)).all())
                or not math.isfinite(alpha) or alpha < 0):
            raise ValueError("GradNorm requires positive train-only reference losses and finite alpha >= 0")
        self.alpha = float(alpha)
        self.register_buffer("reference_losses", reference)
        self.weights = nn.Parameter(torch.ones(len(TASKS)))

    def weighted_losses(self, losses):
        if set(losses) != set(TASKS):
            raise ValueError("GradNorm requires all four losses")
        # Only the GradNorm objective may optimize weights, not the task-loss sum.
        return {k: losses[k] * self.weights[i].detach().clone() for i, k in enumerate(TASKS)}

    def prepare_update(self, losses, shared_parameter):
        norms = []
        for key in TASKS:
            value = torch.autograd.grad(losses[key], shared_parameter, retain_graph=True)[0]
            norm = value.detach().float().norm()
            if not bool(torch.isfinite(norm)) or not float(norm) > 0:
                raise RuntimeError("Missing/nonfinite GradNorm shared gradient: " + key)
            norms.append(norm)
        base_norms = torch.stack(norms)
        current = torch.stack([losses[k].detach().float() for k in TASKS])
        if not bool((torch.isfinite(current) & (current >= 0)).all()):
            raise RuntimeError("Invalid GradNorm task losses")
        ratios = current / self.reference_losses
        relative_rates = ratios / ratios.mean().clamp_min(1e-12)
        weighted_norms = self.weights * base_norms
        targets = (weighted_norms.mean() * relative_rates.pow(self.alpha)).detach()
        objective = (weighted_norms - targets).abs().sum()
        self.weights.grad = torch.autograd.grad(objective, self.weights)[0].detach()
        return {"weights": self.values(), "base_gradient_norms": dict(zip(TASKS, base_norms.cpu().tolist())),
                "weighted_gradient_norms": dict(zip(TASKS, weighted_norms.detach().cpu().tolist())),
                "relative_training_rates": dict(zip(TASKS, relative_rates.cpu().tolist()))}

    @torch.no_grad()
    def renormalize(self):
        if not bool(torch.isfinite(self.weights).all()):
            raise RuntimeError("Nonfinite GradNorm weight")
        self.weights.clamp_(min=1e-3)
        self.weights.mul_(len(TASKS) / self.weights.sum())

    def values(self):
        return dict(zip(TASKS, self.weights.detach().cpu().tolist()))
