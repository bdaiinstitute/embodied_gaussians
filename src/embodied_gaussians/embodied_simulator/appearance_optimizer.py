# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import warp as wp
import torch
from .adam import Adam
from .gaussians import GaussianState


class AppearanceOptimizer:
    def __init__(self, gaussians: GaussianState, min_scale: float = 0.005, max_scale: float = 0.01):
        self.gaussians = gaussians
        device = gaussians.means.device
        self.device = device
        self.inv_min_scale = torch.log(torch.tensor(min_scale)).to(device)
        self.inv_max_scale = torch.log(torch.tensor(max_scale)).to(device)

        self.gaussians.colors_logits.requires_grad = True
        self.gaussians.opacities_logits.requires_grad = True
        self.gaussians.scale_log.requires_grad = True

        self.optimizer = Adam(
            [
                wp.from_torch(
                    self.gaussians.colors_logits, dtype=wp.float32
                ).flatten(),  # flatten inside from_torch changes colors_logits to not be a leaf and torch complains
                wp.from_torch(
                    self.gaussians.opacities_logits, dtype=wp.float32
                ).flatten(),
                wp.from_torch(self.gaussians.scale_log, dtype=wp.float32).flatten(),
            ],
            lrs=[0.01, 0.01, 0.01],  # type: ignore
        )

    def set_learnings_rates(self, lrs):
        self.optimizer.lrs = lrs

    def zero_grad(self):
        if self.gaussians.colors_logits.grad is not None:
            self.gaussians.colors_logits.grad.zero_()
        if self.gaussians.opacities_logits.grad is not None:
            self.gaussians.opacities_logits.grad.zero_()
        if self.gaussians.scale_log.grad is not None:
            self.gaussians.scale_log.grad.zero_()

    def step(self):
        self.optimizer.step(
            grad=[
                wp.from_torch(
                    self.gaussians.colors_logits.grad, dtype=wp.float32
                ).flatten(),
                wp.from_torch(
                    self.gaussians.opacities_logits.grad, dtype=wp.float32
                ).flatten(),
                wp.from_torch(
                    self.gaussians.scale_log.grad, dtype=wp.float32
                ).flatten(),
            ]
        )
        if self.optimizer.lrs[2] > 0.0:
            with torch.no_grad():
                self.gaussians.scale_log.clamp_(self.inv_min_scale, self.inv_max_scale)