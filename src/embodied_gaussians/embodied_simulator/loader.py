# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path

import torch
import warp as wp
from embodied_gaussians.physics_simulator.loader import Loader

from embodied_gaussians.embodied_simulator.builder import EmbodiedGaussiansBuilder
from embodied_gaussians.embodied_simulator.gaussians import GaussianState
from embodied_gaussians.embodied_simulator.simulator import (
    EmbodiedGaussianState,
    update_gaussian_transforms,
)


class EmbodiedGaussiansLoader(Loader):
    def __init__(self, device: str = "cuda", load_gaussian_states: bool = True):
        self.load_gaussian_states = load_gaussian_states
        self.has_gaussians_per_timestep = False
        self.has_initial_gaussians = False

        super().__init__(device)

    def load(self, path: Path):
        root = super().load(path)
        assert root is not None
        b: EmbodiedGaussiansBuilder = self.builder  # type: ignore
        self.has_gaussians_per_timestep = False
        g = b.num_gaussians()
        if g > 0 and self.load_gaussian_states:
            if "gaussian_state_means" in root:
                self.has_gaussians_per_timestep = True
                self.gaussian_state_means = root["gaussian_state_means"]
                self.gaussian_state_quats = root["gaussian_state_quats"]
                self.gaussian_state_colors_logits = root["gaussian_state_colors_logits"]
                self.gaussian_state_opacities_logits = root[
                    "gaussian_state_opacities_logits"
                ]
                self.gaussian_state_scale_log = root["gaussian_state_scale_log"]
            else:
                self.load_gaussians_from_builder()

        return root

    def get_embodied_gaussian_state_at_timestamp(
        self, timestamp: float, device: str = "cuda"
    ):
        assert self.index_look_up
        index = int(self.index_look_up.value(timestamp))
        return self.get_embodied_gaussian_state_at_index(index, device)

    def load_gaussians_from_builder(self, device: str = "cuda"):
        b: EmbodiedGaussiansBuilder = self.builder  # type: ignore
        self.gaussian_model = b.build_gaussian_model(device)
        self.gaussian_state = self.gaussian_model.state()

    def get_gaussian_state_at_index(self, index: int, device: str = "cuda"):
        b: EmbodiedGaussiansBuilder = self.builder  # type: ignore
        g = b.num_gaussians()
        assert g > 0 and self.load_gaussian_states
        gaussian_state_means = (
            torch.from_numpy(self.gaussian_state_means[index]).to(device).float()
        )
        gaussian_state_quats = (
            torch.from_numpy(self.gaussian_state_quats[index]).to(device).float()
        )
        gaussian_state_colors_logits = (
            torch.from_numpy(self.gaussian_state_colors_logits[index])
            .to(device)
            .float()
        )
        gaussian_state_opacities_logits = (
            torch.from_numpy(self.gaussian_state_opacities_logits[index])
            .to(device)
            .float()
        )
        gaussian_state_scale_log = (
            torch.from_numpy(self.gaussian_state_scale_log[index]).to(device).float()
        )
        gaussian_state = GaussianState(
            means=gaussian_state_means,
            quats=gaussian_state_quats,
            colors_logits=gaussian_state_colors_logits,
            opacities_logits=gaussian_state_opacities_logits,
            scale_log=gaussian_state_scale_log,
        )
        return gaussian_state

    def get_embodied_gaussian_state_at_index(self, index: int, device: str = "cuda"):
        state, control = super().get_state_at_index(index, device)
        assert self.builder is not None
        b: EmbodiedGaussiansBuilder = self.builder  # type: ignore
        g = b.num_gaussians()
        assert g > 0 and self.load_gaussian_states
        if self.has_gaussians_per_timestep:
            gaussian_state = self.get_gaussian_state_at_index(index, device)
        else:
            update_gaussian_transforms(
                self.gaussian_model, wp.to_torch(state.body_q), self.gaussian_state
            )
            gaussian_state = self.gaussian_state

        return EmbodiedGaussianState(
            physics_state=state,
            physics_control=control,
            gaussian_state=gaussian_state,
        )
