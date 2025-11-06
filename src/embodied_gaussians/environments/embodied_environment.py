# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import trio
import trio.testing
import warp as wp
import warp.sim
from embodied_gaussians.environments.environment import Environment
from embodied_gaussians.physics_simulator.simulator import PhysicsSettings, copy_control
from trio_util import periodic

from embodied_gaussians.environments.virtual_cameras import VirtualCameras
from embodied_gaussians.embodied_simulator.builder import EmbodiedGaussiansBuilder
from embodied_gaussians.embodied_simulator.frames import Frames
from embodied_gaussians.embodied_simulator.visual_forces import VisualForcesSettings
from embodied_gaussians.embodied_simulator.simulator import EmbodiedGaussiansSimulator
from embodied_gaussians.environments.environment import (
    EnvironmentObservations,
)

from embodied_gaussians.embodied_simulator.gaussians import GaussianModel, GaussianState
from embodied_gaussians.environments.environment import (
    EnvironmentActions,
)


@dataclass
class EmbodiedGaussiansObservations(EnvironmentObservations):
    physics_model: warp.sim.Model
    physics_state: warp.sim.State
    gaussian_state: GaussianState
    gaussian_model: GaussianModel
    rendered_images: torch.Tensor | None = None


@dataclass
class EmbodiedGaussiansActions(EnvironmentActions):
    physics_control: warp.sim.Control


class EmbodiedGaussiansEnvironment(Environment):
    def __init__(
        self,
        builder: EmbodiedGaussiansBuilder,
        device: str = "cuda",
        requires_grad: bool = False,
    ):
        self.frames: Frames | None = None
        self.physics_settings = PhysicsSettings(substeps=20, xpbd_iterations=3)
        self.visual_forces_settings = VisualForcesSettings()
        self.sim = EmbodiedGaussiansSimulator(builder, device, requires_grad)
        self.control = self.sim.model.control()
        self.virtual_cameras: VirtualCameras | None = None
        super().__init__()
        self.streams = [torch.cuda.Stream() for _ in range(self.num_envs())]
        self.stash_state()
    
    def stash_state(self):
        self.stashed_state = self.sim.clone_embodied_gaussian_state()

    def restore_state(self):
        self.sim.copy_embodied_gaussian_state(self.stashed_state)
        self.sim.eval_ik()

    def add_virtual_cameras(self, cameras: VirtualCameras):
        self.virtual_cameras = cameras

    def builder(self) -> EmbodiedGaussiansBuilder:
        assert self.sim.builder
        return self.sim.builder  # type: ignore

    def set_ground_friction(self, val: float):
        mu = wp.to_torch(self.sim.model.shape_materials.mu)
        mu[-1] = val

    def get_ground_friction(self):
        mu = wp.to_torch(self.sim.model.shape_materials.mu)
        return float(mu[-1])

    def num_envs(self):
        return self.sim.num_envs

    def step(self):
        self.sim.physics_step(self.physics_settings)
        self.sim.update_gaussian_transforms()
        if self.virtual_cameras is not None:
            self.virtual_cameras.update_poses(
                self.time(),
                wp.to_torch(self.sim.state_0.body_q).reshape(self.num_envs(), -1, 7),
            )
        if self.frames is not None:
            self.sim.compute_visual_forces(
                self.visual_forces_settings,
                self.frames,
                self.physics_settings.dt / self.physics_settings.substeps,
            )

    def dt(self):
        return self.physics_settings.dt

    def observe(self, render_cameras: bool = True) -> EmbodiedGaussiansObservations:
        rendered_images = None
        if render_cameras and self.virtual_cameras is not None:
            self.render_virtual_cameras()
            rendered_images = self.virtual_cameras.rendered_images

        return EmbodiedGaussiansObservations(
            physics_model=self.sim.model,
            physics_state=self.sim.state_0,
            gaussian_state=self.sim.gaussian_state,
            gaussian_model=self.sim.gaussian_model,
            rendered_images=rendered_images,
        )

    def reset(self):
        self.sim.reset()
        self.restore_state()

    def act(self, actions: EmbodiedGaussiansActions):
        copy_control(self.sim.control, actions.physics_control)

    def set_robot_q(self, index: int, q: torch.Tensor):
        self.sim.set_articulation_q(index, q)

    def set_robot_desired_q(self, index: int, q: torch.Tensor):
        self.sim.set_articulation_control_q(index, q)

    def default_actions(self):
        return EmbodiedGaussiansActions(physics_control=self.sim.control)

    def time(self) -> float:
        return self.sim.get_time()

    def set_frames(self, frames: Frames):
        self.frames = frames

    def save_builder(self, path: Path):
        b: EmbodiedGaussiansBuilder = self.sim.builder
        if b.body_count > 0:
            b.body_q = self.sim.state_0.body_q.numpy().tolist()
        if b.joint_count > 0:
            b.joint_q = self.sim.state_0.joint_q.numpy().tolist()
        if b.particle_count > 0:
            b.particle_q = self.sim.state_0.particle_q.numpy().tolist()
        
        if b.num_gaussians() > 0:
            with torch.no_grad():
                # b.gaussian_means = self.sim.gaussian_state.means.cpu().numpy().tolist() # Do not add these to the builder, the builder takes in X_OG, this is X_WG (gaussian relative to the object vs world)
                # b.gaussian_quats = self.sim.gaussian_state.quats.cpu().numpy().tolist() # Do not add these to the builder
                b.gaussian_scales = self.sim.gaussian_state.scales.cpu().numpy().tolist()
                b.gaussian_opacities = self.sim.gaussian_state.opacities.cpu().numpy().tolist()
                b.gaussian_colors = self.sim.gaussian_state.colors.cpu().numpy().tolist()
                b.gaussian_body_ids = self.sim.gaussian_model.body_ids.cpu().numpy().tolist()

        self.sim.builder.save_to_file(path)

    def render_virtual_cameras(self, force: bool = False):
        if self.virtual_cameras is None:
            return
        if self.sim.get_time() == self.virtual_cameras.last_rendered_at and not force:
            return
        c = self.virtual_cameras
        return c.render(self.time(), self.sim.gaussian_state)

    async def run_with_clock(
        self, clock: trio.testing.MockClock, callbacks: list[Callable] = []
    ):
        async for _ in periodic(self.dt()):
            self.step()
            for callback in callbacks:
                callback()
            clock.jump(self.dt())
            await trio.sleep(0)

    async def run(self, callbacks: list[Callable] = []):
        async for _ in periodic(self.dt()):
            self.step()
            for callback in callbacks:
                callback()
