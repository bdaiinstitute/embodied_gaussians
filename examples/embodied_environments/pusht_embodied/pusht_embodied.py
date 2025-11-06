# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import json
from pathlib import Path

import numpy as np
import warp
import warp.sim
import torch


from embodied_gaussians import (Body, Ground, VirtualCamerasBuilder, EmbodiedGaussiansBuilder, EmbodiedGaussiansEnvironment, read_extrinsics, read_ground)

Q_START = np.array(
    [
        2.6537935,
        1.5346614,
        -1.6750929,
        -2.61287848,
        1.46183864,
        1.54922744,
        0.82306163,
    ]
)
current_dir = Path(__file__).parent
GROUND_PATH = current_dir / Path("environment/ground_plane.json")
EXTRINSICS_PATH = current_dir / Path("environment/extrinsics.json")
BODY_NAME = "tblock"
BODY_ID = warp.constant(13)


def get_body(name: str) -> Body:
    path = current_dir / Path(f"objects/{name}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return Body.model_validate(data)


def build_environment(num_envs: int = 1, add_gaussians: bool = True):
    ground_data = read_ground(GROUND_PATH)
    ground = Ground(plane=ground_data)
    body = get_body(BODY_NAME)
    ground_body = get_body("ground")
    builder = EmbodiedGaussiansBuilder()
    if add_gaussians:
        builder.add_renderable_articulation_from_urdf(
            urdf_path=current_dir / Path("assets/robots/panda.urdf"),
            initial_joints=Q_START,
            stiffness=500,
            damping=100,
            ignore_inertial_definitions=True,
            ensure_nonstatic_links=True,
            collapse_fixed_joints=False,
        )
    else:
        builder.add_articulation_from_urdf(
            urdf_path=current_dir / Path("assets/robots/panda.urdf"),
            initial_joints=Q_START,
            stiffness=500,
            damping=100,
            ignore_inertial_definitions=True,
            ensure_nonstatic_links=True,
            collapse_fixed_joints=False,
        )

    bid = builder.add_rigid_body(body, add_gaussians=add_gaussians)
    if add_gaussians:
        builder.add_visual_body(ground_body)
    final_builder = EmbodiedGaussiansBuilder()
    final_builder.set_ground_plane(ground.normal(), ground.offset())
    for env in range(num_envs):
        final_builder.add_builder(builder)
    env = EmbodiedGaussiansEnvironment(final_builder)
    warp.to_torch(env.sim.model.gravity_factor).reshape(num_envs, -1)[:, :13] = 0.0
    q_start = torch.from_numpy(Q_START).float()
    env.sim.get_joint_act()[:, :7] = q_start
    env.stash_state()
    return env


def add_static_cameras(builder: VirtualCamerasBuilder):
    K = np.array(
        [
            [241.0, 0.0, 240.0],
            [0.0, 241.0, 135.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Camera data (ID, extrinsic matrix X_BV)
    camera_data = [
        # (
        #     "220422302296",
        #     np.array(
        #         [
        #             [
        #                 0.606747567653656,
        #                 -0.5086942911148071,
        #                 0.6108089089393616,
        #                 0.9495939016342163,
        #             ],
        #             [
        #                 0.779473066329956,
        #                 0.5313830971717834,
        #                 -0.33174341917037964,
        #                 0.013426600024104118,
        #             ],
        #             [
        #                 -0.15581756830215454,
        #                 0.6773936152458191,
        #                 0.7189289331436157,
        #                 0.38001641631126404,
        #             ],
        #             [0.0, 0.0, 0.0, 1.0],
        #         ]
        #     ),
        # ),
        (
            "234222302164",
            np.array(
                [
                    [
                        -0.5385846495628357,
                        -0.7674259543418884,
                        0.347827672958374,
                        0.6469128727912903,
                    ],
                    [
                        0.8420496582984924,
                        -0.504772961139679,
                        0.19014911353588104,
                        0.5172103047370911,
                    ],
                    [
                        0.02964870259165764,
                        0.39529949426651,
                        0.9180736541748047,
                        0.5189733505249023,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        # (
        #     "234222303707",
        #     np.array(
        #         [
        #             [
        #                 -0.0959780216217041,
        #                 0.971773624420166,
        #                 -0.21550923585891724,
        #                 0.17146411538124084,
        #             ],
        #             [
        #                 -0.9483034610748291,
        #                 -0.023470992222428322,
        #                 0.3164959251880646,
        #                 0.5118779540061951,
        #             ],
        #             [
        #                 0.30250418186187744,
        #                 0.23474480211734772,
        #                 0.9237889647483826,
        #                 0.7177857756614685,
        #             ],
        #             [0.0, 0.0, 0.0, 1.0],
        #         ]
        #     ),
        # ),
    ]
    for camera_id, X_BV in camera_data:
        builder.add_camera(camera_id, K, X_BV)


def add_gripper_camera(builder: VirtualCamerasBuilder):
    K = np.array(
        [
            [100.0, 0.0, 240.0],
            [0.0, 100.0, 135.0],
            [0.0, 0.0, 1.0],
        ]
    )
    X_BV = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    builder.add_camera("gripper_camera", K, X_BV, body_id=12)
