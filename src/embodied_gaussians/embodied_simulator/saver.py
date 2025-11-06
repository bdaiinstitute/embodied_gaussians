# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
from typing_extensions import override
from pathlib import Path

import torch

from embodied_gaussians.physics_simulator.saver import Saver

from embodied_gaussians.embodied_simulator.builder import EmbodiedGaussiansBuilder
from embodied_gaussians.embodied_simulator.simulator import EmbodiedGaussiansSimulator


class EmbodiedGaussiansSaver(Saver):
    def __init__(
        self,
        simulator: EmbodiedGaussiansSimulator,
        device: str,
        max_frames: int = 10_000,
        save_gaussian_state: bool = True,
    ):
        super().__init__(simulator, device, max_frames)
        self.save_gaussian_state = save_gaussian_state

    def save(self, path: Path):
        root = super().save(path)
        assert root
        s: EmbodiedGaussiansSimulator = self.simulator  # type: ignore
        state = s.gaussian_state
        t = self.current_index
        g = state.num_gaussians
        if g > 0 and self.save_gaussian_state:
            s = root.create_array(
                name="gaussian_state_means", shape=(t, g, 3), dtype="f4"
            )
            s[:] = self.gaussian_state_means.cpu().numpy()
            s = root.create_array(
                name="gaussian_state_quats", shape=(t, g, 4), dtype="f4"
            )
            s[:] = self.gaussian_state_quats.cpu().numpy()

            s = root.create_array(
                name="gaussian_state_colors_logits", shape=(t, g, 3), dtype="f4"
            )
            s[:] = self.gaussian_state_colors_logits.cpu().numpy()

            s = root.create_array(
                name="gaussian_state_opacities_logits", shape=(t, g), dtype="f4"
            )
            s[:] = self.gaussian_state_opacities_logits.cpu().numpy()

            s = root.create_array(
                name="gaussian_state_scale_log", shape=(t, g, 3), dtype="f4"
            )
            s[:] = self.gaussian_state_scale_log.cpu().numpy()

    @override
    def allocate_demo(self, max_frames: int, device: str):
        if self.max_frames == max_frames:
            return
        b: EmbodiedGaussiansBuilder = self.builder  # type: ignore
        g = b.num_gaussians()
        t = max_frames
        if g > 0:
            self.gaussian_state_means = torch.zeros(
                (t, g, 3), dtype=torch.float32, device=device
            )
            self.gaussian_state_quats = torch.zeros(
                (t, g, 4), dtype=torch.float32, device=device
            )
            self.gaussian_state_colors_logits = torch.zeros(
                (t, g, 3), dtype=torch.float32, device=device
            )
            self.gaussian_state_opacities_logits = torch.zeros(
                (t, g), dtype=torch.float32, device=device
            )
            self.gaussian_state_scale_log = torch.zeros(
                (t, g, 3), dtype=torch.float32, device=device
            )
        super().allocate_demo(max_frames, device)

    @override
    def clear_allocation(self):
        super().clear_allocation()
        b: EmbodiedGaussiansBuilder = self.builder  # type: ignore
        g = b.num_gaussians()
        if g > 0 and self.save_gaussian_state:
            self.gaussian_state_means.zero_()
            self.gaussian_state_quats.zero_()
            self.gaussian_state_colors_logits.zero_()
            self.gaussian_state_opacities_logits.zero_()
            self.gaussian_state_scale_log.zero_()

    @override
    def record_state_and_control(self, timestamp: float):
        i = self.current_index
        if i >= self.max_frames:
            logging.warning("Cannot save because the buffer is full.")
            return
        s: EmbodiedGaussiansSimulator = self.simulator  # type: ignore
        state = s.gaussian_state
        g = state.num_gaussians
        if g > 0 and self.save_gaussian_state:
            with torch.no_grad():
                self.gaussian_state_means[i] = state.means.clone()
                self.gaussian_state_quats[i] = state.quats.clone()
                self.gaussian_state_colors_logits[i] = state.colors_logits.clone()
                self.gaussian_state_opacities_logits[i] = state.opacities_logits.clone()
                self.gaussian_state_scale_log[i] = state.scale_log.clone()

        super().record_state_and_control(timestamp)
