# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
import torch


@dataclass
class GaussianState:
    means: torch.Tensor  # (n_gaussians, 3)
    quats: torch.Tensor  # (n_gaussians, 4) (w, x, y, z)
    colors_logits: torch.Tensor  # (n_gaussians, 3)
    opacities_logits: torch.Tensor  # (n_gaussians,)
    scale_log: torch.Tensor  # (n_gaussians, 3)

    @property
    def colors(self):
        return self.colors_logits.sigmoid()

    @property
    def opacities(self):
        return self.opacities_logits.sigmoid()

    @property
    def scales(self):
        return self.scale_log.exp()

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    def copy(self, src: "GaussianState"):
        with torch.no_grad():
            self.means.copy_(src.means)
            self.quats.copy_(src.quats)
            self.colors_logits.copy_(src.colors_logits)
            self.opacities_logits.copy_(src.opacities_logits)
            self.scale_log.copy_(src.scale_log)

    def clone(self):
        with torch.no_grad():
            return GaussianState(
                means=self.means.clone(),
                quats=self.quats.clone(),
                colors_logits=self.colors_logits.clone(),
                opacities_logits=self.opacities_logits.clone(),
                scale_log=self.scale_log.clone(),
            )

    def slice(self, slice_obj):
        with torch.no_grad():
            return GaussianState(
                means=self.means[slice_obj],
                quats=self.quats[slice_obj],
                colors_logits=self.colors_logits[slice_obj],
                opacities_logits=self.opacities_logits[slice_obj],
                scale_log=self.scale_log[slice_obj],
            )

    def reshape(self, shape):
        with torch.no_grad():
            return GaussianState(
                means=self.means.reshape(*shape, 3),
                quats=self.quats.reshape(*shape, 4),
                colors_logits=self.colors_logits.reshape(*shape, 3),
                opacities_logits=self.opacities_logits.reshape(*shape),
                scale_log=self.scale_log.reshape(*shape, 3),
            )
    
@dataclass
class GaussianModel:
    means: torch.Tensor  # (n_gaussians, 3)
    quats: torch.Tensor  # (n_gaussians, 4) (w, x, y, z)
    scales: torch.Tensor  # (n_gaussians, 3)
    opacities: torch.Tensor  # (n_gaussians,)
    colors: torch.Tensor  # (n_gaussians, 3)
    body_ids: torch.Tensor  # (n_gaussians,)

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    @property
    def device(self):
        return self.means.device

    def state(self):
        return GaussianState(
            means=self.means.clone(),
            quats=self.quats.clone(),
            colors_logits=self.colors.logit(),
            opacities_logits=self.opacities.logit(),
            scale_log=self.scales.log(),
        )

    def copy_from_state(self, state: GaussianState):
        with torch.no_grad():
            # self.means.copy_(state.means)
            # self.quats.copy_(state.quats)
            self.scales.copy_(state.scales)
            self.colors.copy_(state.colors)
            self.opacities.copy_(state.opacities)
