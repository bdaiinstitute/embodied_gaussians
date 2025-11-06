# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import warp as wp
import warp.sim
from typing_extensions import override

from embodied_gaussians import Environment, Task, EnvironmentActions, EnvironmentObservations, PhysicsSettings, ModelBuilder, Simulator

TBLOCK_ID = wp.constant(3)

@dataclass
class PushTEnvironmentActions(EnvironmentActions):
    pusher_desired_positions: torch.Tensor

    @classmethod
    def allocate(cls, num_env: int, device: str = "cuda"):
        return PushTEnvironmentActions(
            pusher_desired_positions=torch.zeros(
                (num_env, 2), dtype=torch.float32, device=device
            )
        )

@dataclass
class PushTEnvironmentObservations(EnvironmentObservations):
    pusher_positions: torch.Tensor
    tblock_transforms: torch.Tensor

    @classmethod
    def allocate(
        cls, num_env: int, device: str = "cuda"
    ) -> "PushTEnvironmentObservations":
        tblock_transforms = torch.zeros(
            (num_env, 7), dtype=torch.float32, device=device
        )
        tblock_transforms[:, -1] = 1.0
        return PushTEnvironmentObservations(
            pusher_positions=torch.zeros(
                (num_env, 2), dtype=torch.float32, device=device
            ),
            tblock_transforms=tblock_transforms,
        )


class PushTEnvironment(Environment, Task):
    @staticmethod
    def build(num_envs: int = 1):
        builder = ModelBuilder(up_vector=(0.0, 0.0, 1.0))
        model_builder = warp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
        current_dir = Path(__file__).parent
        warp.sim.parse_urdf(
            current_dir / Path("assets/pusher.urdf"),
            model_builder,
            xform=wp.transform((0.0, 0.0, 0.17), (0.0, 0.0, 0.0, 1.0)),
            density=6000.0,
        )
        b = model_builder.add_body(
            name="tblock",
            origin=wp.transform((0.3, 0.3, 0.05), (0.0, 0.0, 0.0, 1.0)),  # type: ignore
        )
        add_tblock_shape(model_builder, b)
        add_tblock_shape(
            model_builder,
            -1,
            thickness=0.005,
            has_shape_collision=False,
            has_ground_collision=False,
        )
        for _ in range(num_envs):
            builder.add_builder(model_builder)
        return PushTEnvironment(builder)

    def __init__(
        self,
        builder: ModelBuilder,
    ):
        self._simulator = s = Simulator(builder)
        self._physics_settings = PhysicsSettings(
            dt=1.0 / 60.0, substeps=8, xpbd_iterations=10
        )
        super(Environment).__init__()
        self.builder = builder
        self._observations = PushTEnvironmentObservations.allocate(
            self.builder.num_envs
        )
        self._success = torch.zeros((s.num_envs), dtype=torch.int32, device="cuda")
        self._rewards = torch.zeros((s.num_envs), dtype=torch.float32, device="cuda")
        self._success_time = torch.zeros(
            (s.num_envs), dtype=torch.float32, device="cuda"
        )
        self._X_ET = wp.transform_identity(dtype=float)
        # warm start cuda kernels
        self.step()

    def num_envs(self):
        return self._simulator.num_envs

    def reset(self):
        self._simulator.reset()
        self._randomize_tblock()
        self._reset_pusher()

    def observe(self):
        return self._observations

    def done(self):
        return self._success

    @override
    def act(self, actions: PushTEnvironmentActions):
        self._simulator.set_joint_act(actions.pusher_desired_positions)

    def step(self):
        self._simulator.physics_step(self._physics_settings)
        self._update_state()
        self._update_task()

    def time(self):
        return self._simulator.sim_time

    def dt(self):
        return self._physics_settings.dt

    def simulator(self):
        return self._simulator

    def default_actions(self) -> PushTEnvironmentActions:
        return PushTEnvironmentActions.allocate(self.num_envs())

    def _update_state(self):
        s = self._simulator
        wp.launch(
            kernel=synchronize_state_kernel,
            dim=(s.num_envs,),
            inputs=[
                s.state_0.body_q.reshape((s.num_envs, -1)),  # type: ignore
                s.state_0.joint_q.reshape((s.num_envs, -1)),  # type: ignore
                self._observations.pusher_positions,
                self._observations.tblock_transforms,
            ],
        )

    def _update_task(self):
        distance_threshold = 0.01
        angle_threshold_degrees = 0.8
        s = self._simulator

        wp.launch(
            kernel=get_reward_and_success_kernel,
            dim=(s.num_envs,),
            inputs=[
                s.sim_time,
                self._X_ET,
                distance_threshold,
                angle_threshold_degrees * np.pi / 180,
                self._observations.tblock_transforms,
                self._rewards,
                self._success,
                self._success_time,
            ],
        )

    def _reset_pusher(self):
        s = self._simulator
        s.state_0.joint_q.zero_()
        s.state_0.joint_qd.zero_()
        s.control.joint_act.zero_()
        s.eval_fk()

    def _randomize_tblock(self):
        seed = int(np.random.randint(0, 1_000_000_000))
        s = self._simulator
        wp.launch(
            kernel=randomize_states_kernel,
            dim=(s.num_envs,),
            inputs=[
                seed,
                s.state_0.body_q.reshape((s.num_envs, -1)),  # type: ignore
            ],
        )

    def set_target(self, X_ET: wp.transformf):
        self._X_ET = X_ET


def add_tblock_shape(builder: warp.sim.ModelBuilder, body, thickness=0.04, **kwargs):
    builder.add_shape_box(
        body,
        pos=[0.0, 0.0, 0.04 / 2],
        hx=0.2 / 2,
        hy=0.05 / 2,
        hz=thickness / 2,
        **kwargs,
    )
    builder.add_shape_box(
        body,
        pos=[0.0, -0.1, 0.04 / 2],
        hy=0.15 / 2,
        hx=0.05 / 2,
        hz=thickness / 2,
        **kwargs,
    )


@wp.kernel
def synchronize_state_kernel(
    body_q: wp.array(ndim=2, dtype=wp.transformf),  # type: ignore
    joint_q: wp.array(ndim=2, dtype=wp.float32),  # type: ignore
    out_pusher_positions: wp.array(dtype=wp.vec2f),  # type: ignore
    out_tblock_transforms: wp.array(dtype=wp.transformf),  # type: ignore
):
    env_ind = wp.tid()  # per environment
    X_WC = body_q[env_ind, TBLOCK_ID]  # current pose
    t_WC = wp.transform_get_translation(X_WC)
    q_WC = wp.transform_get_rotation(X_WC)
    X_EC = wp.transformf(t_WC, q_WC)
    out_tblock_transforms[env_ind] = X_EC
    out_pusher_positions[env_ind] = wp.vec2f(joint_q[env_ind][0], joint_q[env_ind][1])


@wp.kernel
def get_reward_and_success_kernel(
    sim_time: float,
    X_ET: wp.transformf,  # transform from env to target
    distance_threshold: float,
    angle_threshold: float,
    tblock_transforms: wp.array(dtype=wp.transformf),  # type: ignore
    reward: wp.array(dtype=float),  # type: ignore
    success: wp.array(dtype=int),  # type: ignore
    time_successful: wp.array(dtype=float),  # type: ignore
):
    env_ind = wp.tid()  # per environment
    X_EC = tblock_transforms[env_ind]
    X_CE = wp.transform_inverse(X_EC)
    X_CT = X_CE * X_ET
    q_CT = wp.transform_get_rotation(X_CT)
    angle = wp.float32(0.0)  # type: ignore
    axis = wp.vec3f(0.0, 0.0, 0.0)
    wp.quat_to_axis_angle(q_CT, axis, angle)
    distance = wp.length(wp.transform_get_translation(X_CT))
    new_success = (
        abs(distance) < distance_threshold and abs(float(angle)) < angle_threshold
    )
    reward[env_ind] = -(distance + angle)
    old_success = success[env_ind]
    if not old_success and new_success:
        time_successful[env_ind] = sim_time
    success[env_ind] = int(new_success)


@wp.kernel
def randomize_states_kernel(
    random_seed: int,
    body_q_out: wp.array(ndim=2, dtype=wp.transformf),  # type: ignore
):
    env_ind = wp.tid()  # per environment
    num_envs = body_q_out.shape[0]
    rng = wp.rand_init(wp.int32(random_seed), wp.int32(1000 * env_ind * num_envs))
    angle = wp.randf(rng) * 2.0 * wp.pi
    quat = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), angle)
    extent = 0.5
    trans = wp.vec3f(wp.randf(rng), wp.randf(rng), 0.0) * extent - wp.vec3f(
        extent / 2.0, extent / 2.0, 0.0
    )
    X_WB = wp.transformf(trans, quat)
    body_q_out[env_ind, TBLOCK_ID] = X_WB
