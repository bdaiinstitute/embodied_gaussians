# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import warp as wp
import warp.sim

from embodied_gaussians.utils.physics_utils import transform_from_matrix, save_builder, load_builder


class ModelBuilder(warp.sim.ModelBuilder):
    def __init__(self, up_vector=(0.0, 0.0, 1.0), gravity=-9.80665):
        super().__init__(up_vector=up_vector, gravity=gravity)
        self.rigid_contact_torsional_friction = 0.0

    def add_articulation_from_urdf(
        self,
        urdf_path: str,
        initial_joints: np.ndarray | None = None,
        X_WB: np.ndarray = np.eye(4),
        armature: float = 0.1,
        damping: float = 80.0,
        stiffness: float = 400,
        enable_self_collisions: bool = False,
        **kwargs,
    ):
        start_joint = len(self.joint_q)
        T_WB = transform_from_matrix(X_WB)
        warp.sim.parse_urdf(
            urdf_path,
            xform=wp.transformf(T_WB[:3], T_WB[3:]),
            builder=self,
            armature=armature,
            damping=damping,
            stiffness=stiffness,
            enable_self_collisions=enable_self_collisions,
            **kwargs,
        )
        end_joint = len(self.joint_q)
        num_joints = end_joint - start_joint
        if initial_joints is not None:
            assert len(initial_joints) <= num_joints, (
                f"Initial joints must have length {num_joints}"
            )
            num_joints_given = len(initial_joints)
            self.joint_q[start_joint:num_joints_given] = initial_joints

    def finalize(self, device=None, requires_grad=False):
        res = super().finalize(device, requires_grad)
        res.gravity_factor = wp.ones(
            self.body_count, dtype=wp.float32, requires_grad=requires_grad
        )
        return res
    
    def save_to_file(self, file_path: str):
        save_builder(file_path, self)
    
    @staticmethod
    def load_from_file(file_path: str):
        return load_builder(file_path)
