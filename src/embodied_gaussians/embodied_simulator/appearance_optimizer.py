import warp as wp
from .adam import Adam
from .gaussians import GaussianState


class AppearanceOptimizer:
    def __init__(self, gaussians: GaussianState):
        self.gaussians = gaussians
        device = gaussians.means.device
        self.device = device

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
