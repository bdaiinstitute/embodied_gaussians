# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Literal
import pysegreduce
from dataclasses import dataclass
import torch
import warp as wp
import warp.sim
from gsplat.rendering import rasterization
from embodied_gaussians.physics_simulator.simulator import Simulator, copy_control, copy_state

from embodied_gaussians.embodied_simulator import EmbodiedGaussiansBuilder
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel, GaussianState

from embodied_gaussians.embodied_simulator.appearance_optimizer import AppearanceOptimizer
from embodied_gaussians.embodied_simulator.frames import Frames
from embodied_gaussians.embodied_simulator.visual_forces import VisualForces, VisualForcesSettings
from embodied_gaussians.embodied_simulator.warp import (
    apply_forces_kernel,
    update_gaussians_transforms_kernel,
    update_visual_forces_kernel,
)


@dataclass
class EmbodiedGaussianState:
    physics_state: warp.sim.State
    physics_control: warp.sim.Control
    gaussian_state: GaussianState


class EmbodiedGaussiansSimulator(Simulator):
    def __init__(
        self,
        builder: EmbodiedGaussiansBuilder,
        device: str = "cuda",
        require_grad=False,
    ):
        super().__init__(builder, device=device, requires_grad=require_grad)
        self.gaussian_model = builder.gaussian_model
        self.gaussian_state = builder.gaussian_state
        self.bodies_affected_by_visual_forces = builder.bodies_affected_by_visual_forces
        self.visual_forces = VisualForces(
            self.gaussian_model,
            self.gaussian_state,
            bodies_affected_by_visual_forces=self.bodies_affected_by_visual_forces,
        )
        self.appearance_optimizer = AppearanceOptimizer(self.gaussian_state)

    def get_specific_environment_state(self, env_ind: int):
        with torch.no_grad():
            sim = self
            s = wp.to_torch(sim.state_0).reshape(self.num_envs(), -1, 7)[env_ind]
            c = wp.to_torch(sim.control).reshape(self.num_envs(), -1)[env_ind]
            g = sim.gaussian_state.reshape((self.num_envs(), -1, 7)).slice(env_ind).clone()
            s = wp.from_torch(s)
            c = wp.from_torch(c)
            return EmbodiedGaussianState(
                physics_state=s, physics_control=c, gaussian_state=g
            )
    
    def set_specific_environment_state(self, env_ind: int, state: EmbodiedGaussianState):
        sim = self
        with torch.no_grad():
            wp.to_torch(sim.state_0).reshape(self.num_envs(), -1, 7)[env_ind] = wp.to_torch(state.physics_state)
            wp.to_torch(sim.control).reshape(self.num_envs(), -1)[env_ind] = wp.to_torch(state.physics_control)
            g = sim.gaussian_state.reshape((self.num_envs(), -1, 7)).slice(env_ind)
            g.copy(state.gaussian_state)

    def embodied_gaussian_state(self):
        s = self.state_0
        c = self.control
        g = self.gaussian_state.clone()
        return EmbodiedGaussianState(
            physics_state=s, physics_control=c, gaussian_state=g
        )

    def clone_embodied_gaussian_state(self):
        s = self.clone_state()
        c = self.clone_control()
        g = self.gaussian_state.clone()
        return EmbodiedGaussianState(
            physics_state=s, physics_control=c, gaussian_state=g
        )

    def copy_embodied_gaussian_state(self, state: EmbodiedGaussianState):
        self.set_state(state.physics_state)
        self.set_control(state.physics_control)
        self.gaussian_state.copy(state.gaussian_state)

    def render_visual_forces(
        self,
        X_CWs: torch.Tensor,
        Ks: torch.Tensor,
        width: float,
        height: float,
        background: torch.Tensor,
    ):
        num_images = X_CWs.shape[0]
        with torch.no_grad():
            render_colors, render_alphas, info = rasterization(
                means=self.visual_forces.means,
                quats=self.visual_forces.quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=X_CWs,
                Ks=Ks,
                width=int(width),
                height=int(height),
                camera_model="pinhole",
                render_mode="RGB",
                backgrounds=background.reshape(1, 3).repeat(num_images, 1),
            )
        return render_colors, render_alphas, info

    def render_gaussians(
        self,
        gaussian_state: GaussianState,
        X_CWs: torch.Tensor,
        Ks: torch.Tensor,
        width: float,
        height: float,
        background: torch.Tensor,
        near_plane=0.01,
        far_plane=3.0,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        **kwargs,
    ):
        return render_gaussians(
            gaussian_state,
            X_CWs,
            Ks,
            width,
            height,
            background,
            near_plane,
            far_plane,
            render_mode,
            **kwargs,
        )

    def compute_visual_forces(
        self, settings: VisualForcesSettings, frames: Frames, dt: float
    ):
        self._compute_visual_forces(settings, frames, dt)

    def update_gaussian_transforms(self):
        update_gaussian_transforms(
            self.gaussian_model, self.state_0.body_q, self.gaussian_state
        )

    def _compute_visual_forces(
        self, settings: VisualForcesSettings, frames: Frames, dt: float
    ):
        with torch.no_grad():
            self.visual_forces.means.copy_(self.gaussian_state.means)
            self.visual_forces.quats.copy_(self.gaussian_state.quats)

        # self.visual_forces.optimizer.reset_internal_state()
        self.visual_forces.set_learnings_rates([settings.lr_means, settings.lr_quats])
        self.appearance_optimizer.set_learnings_rates(
            [settings.lr_color, settings.lr_opacity, settings.lr_scale]
        )

        for _ in range(settings.iterations):
            render_colors, render_alphas, info = rasterization(
                means=self.visual_forces.means,
                quats=self.visual_forces.quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=frames.X_CWs_opencv_gpu,
                Ks=frames.Ks_gpu,
                width=int(frames.width),
                height=int(frames.height),
                camera_model="pinhole",
                render_mode="RGB",
            )

            loss = torch.nn.functional.mse_loss(render_colors, frames.colors_gpu)
            # ideas: add a loss that pushes the colors back to their orignal values or to some sort of ema colors
            # ideas: allow the gaussians to jitter a bit while anchoring them to the original positions

            self.visual_forces.zero_grad()
            self.appearance_optimizer.zero_grad()
            loss.backward()
            self.visual_forces.step()
            self.appearance_optimizer.step()

        wp.launch(
            kernel=update_visual_forces_kernel,
            dim=self.gaussian_model.num_gaussians,  # type: ignore
            inputs=[
                settings.kp,
                self.gaussian_state.means.detach(),
                self.gaussian_state.quats.detach(),
                self.gaussian_state.opacities.detach(),
                self.visual_forces.means.detach(),
                self.visual_forces.quats.detach(),
                self.gaussian_model.body_ids.detach(),
                self.state_0.body_q,
                self.visual_forces.forces.detach(),
                self.visual_forces.moments.detach(),
            ],
        )

        pysegreduce.reduce_vec3f(
            self.visual_forces.forces.data_ptr(),
            self.visual_forces._start_inds.data_ptr(),
            self.visual_forces._end_inds.data_ptr(),
            len(self.visual_forces._start_inds),
            self.visual_forces._total_forces.data_ptr(),
            0,
        ) # Replace this with segmented reduce when it is implemented in warp

        pysegreduce.reduce_vec3f(
            self.visual_forces.moments.data_ptr(),
            self.visual_forces._start_inds.data_ptr(),
            self.visual_forces._end_inds.data_ptr(),
            len(self.visual_forces._start_inds),
            self.visual_forces._total_moments.data_ptr(),
            0,
        )

        wp.launch(
            kernel=apply_forces_kernel,
            dim=self.visual_forces._num_bodies,  # type: ignore
            inputs=[
                dt,
                self.visual_forces._total_forces,
                self.visual_forces._total_moments,
                self.visual_forces._body_ids,
                self.state_0.body_f,
            ],
        )


def render_gaussians(
    gaussian_state: GaussianState,
    X_CWs: torch.Tensor,
    Ks: torch.Tensor,
    width: float,
    height: float,
    background: torch.Tensor,
    near_plane=0.01,
    far_plane=3.0,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    **kwargs,
):
    num_images = X_CWs.shape[0]
    with torch.no_grad():
        render_colors, render_alphas, info = rasterization(
            means=gaussian_state.means,
            quats=gaussian_state.quats,
            scales=gaussian_state.scales,
            colors=gaussian_state.colors,
            opacities=gaussian_state.opacities,
            viewmats=X_CWs,
            Ks=Ks,
            width=int(width),
            height=int(height),
            camera_model="pinhole",
            render_mode=render_mode,
            backgrounds=background.reshape(1, 3).repeat(num_images, 1),
            near_plane=near_plane,
            far_plane=far_plane,
            **kwargs,
        )
    return render_colors, render_alphas, info


def update_gaussian_transforms(model: GaussianModel, body_q, out_state: GaussianState):
    if model.num_gaussians == 0:
        return
    wp.launch(
        kernel=update_gaussians_transforms_kernel,
        dim=model.num_gaussians,  # type: ignore
        inputs=[
            model.means,
            model.quats,
            model.body_ids,
            body_q,
        ],
        outputs=[
            out_state.means,
            out_state.quats,
        ],
    )


def copy_embodied_gaussian_state(
    dest: EmbodiedGaussianState, src: EmbodiedGaussianState
):
    copy_state(dest.physics_state, src.physics_state)
    copy_control(dest.physics_control, src.physics_control)
    dest.gaussian_state.copy(src.gaussian_state)
