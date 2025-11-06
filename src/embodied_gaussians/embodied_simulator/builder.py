# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np

from embodied_gaussians.physics_simulator.builder import ModelBuilder
import torch
import warp as wp
import warp.sim
import open3d as o3d
from embodied_gaussians.scene_builders.domain import Body
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel


class EmbodiedGaussiansBuilder(ModelBuilder):
    def __init__(self, up_vector=(0.0, 0.0, 1.0), gravity=-9.80665):
        super().__init__(up_vector=up_vector, gravity=gravity)

        self.gaussian_means = []
        self.gaussian_quats = []
        self.gaussian_scales = []
        self.gaussian_opacities = []
        self.gaussian_colors = []
        self.gaussian_body_ids = []
        self.bodies_affected_by_visual_forces = []

    def add_builder(
        self,
        builder: "EmbodiedGaussiansBuilder",
        xform=None,
        update_num_env_count=True,
        separate_collision_group=True,
    ):
        body_count = self.body_count
        arrs = [
            "gaussian_means",
            "gaussian_quats",
            "gaussian_scales",
            "gaussian_opacities",
            "gaussian_colors",
        ]
        for arr in arrs:
            getattr(self, arr).extend(getattr(builder, arr))

        new_gaussian_body_ids = [
            b + body_count if b != -1 else -1 for b in builder.gaussian_body_ids
        ]
        new_bodies_affected_by_visual_forces = [
            b + body_count for b in builder.bodies_affected_by_visual_forces
        ]
        self.gaussian_body_ids.extend(new_gaussian_body_ids)
        self.bodies_affected_by_visual_forces.extend(
            new_bodies_affected_by_visual_forces
        )

        super().add_builder(
            builder, xform, update_num_env_count, separate_collision_group
        )

    def add_visual_body(self, body: Body):
        # X_WB = np.asarray(body.X_WB)
        # quat = wp.quat_from_matrix(X_WB[:3, :3])
        assert body.gaussians
        gaussians = body.gaussians
        self.gaussian_means.extend(gaussians.means)
        self.gaussian_quats.extend(gaussians.quats)
        self.gaussian_scales.extend(gaussians.scales)
        self.gaussian_opacities.extend(gaussians.opacities)
        self.gaussian_colors.extend(gaussians.colors)
        self.gaussian_body_ids.extend([-1] * len(gaussians.means))

    def add_rigid_body(self, body: Body, mu: float = 0.0, add_gaussians: bool = True):
        X_WB = np.asarray(body.X_WB)
        quat = wp.quat_from_matrix(X_WB[:3, :3])
        trans = X_WB[:3, 3]
        trans[2] = 0.1
        t = wp.transformf(*trans, *quat)
        b = self.add_body(origin=t)  # type: ignore
        self.bodies_affected_by_visual_forces.append(b)
        particles = body.particles
        assert particles
        for i in range(len(particles.means)):
            pos = particles.means[i]
            quat = particles.quats[i]
            radius = particles.radii[i]
            self.add_shape_sphere(
                body=b,
                radius=radius,
                pos=pos,
                rot=[quat[1], quat[2], quat[3], quat[0]],
                mu=mu,
                # density=10.0,
            )

        if add_gaussians:
            gaussians = body.gaussians
            assert gaussians
            self.gaussian_means.extend(gaussians.means)
            self.gaussian_quats.extend(gaussians.quats)
            self.gaussian_scales.extend(gaussians.scales)
            self.gaussian_opacities.extend(gaussians.opacities)
            self.gaussian_colors.extend(gaussians.colors)
            self.gaussian_body_ids.extend([b] * len(gaussians.means))

        return b

    def num_gaussians(self):
        return len(self.gaussian_means)

    def add_renderable_articulation_from_urdf(
        self,
        urdf_path: str,
        initial_joints: np.ndarray | None = None,
        X_WB: np.ndarray = np.eye(4),
        armature: float = 0.1,
        damping: float = 80.0,
        stiffness: float = 400,
        enable_self_collisions: bool = False,
        add_gaussians: bool = True,
        **kwargs,
    ):
        start_shape_idx = len(self.shape_body)
        self.add_articulation_from_urdf(
            urdf_path,
            initial_joints,
            X_WB,
            armature,
            damping,
            stiffness,
            enable_self_collisions,
            **kwargs,
        )
        end_shape_idx = len(self.shape_body)
        if add_gaussians:
            for i in range(start_shape_idx, end_shape_idx):
                mesh: warp.sim.model.Mesh = self.shape_geo_src[i]
                body_id = self.shape_body[i]
                mesh_open3d = o3d.geometry.TriangleMesh()
                mesh_open3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                mesh_open3d.triangles = o3d.utility.Vector3iVector(
                    mesh.indices.reshape(-1, 3)
                )
                area = mesh_open3d.get_surface_area()
                points_per_unit_area = 10000
                points: o3d.geometry.PointCloud = (
                    mesh_open3d.sample_points_poisson_disk(
                        int(area * points_per_unit_area)
                    )
                )
                means = np.asarray(points.points)
                num_points = len(points.points)
                area_per_point = 0.005

                self.gaussian_means.extend(means.tolist())
                self.gaussian_quats.extend([[1, 0, 0, 0]] * num_points)
                self.gaussian_scales.extend(
                    [[area_per_point, area_per_point, area_per_point]] * num_points
                )
                self.gaussian_opacities.extend([0.5] * num_points)
                self.gaussian_colors.extend([[0.5, 0.5, 0.5]] * num_points)
                self.gaussian_body_ids.extend([body_id] * num_points)

    def build_gaussian_model(self, device: str = "cuda"):
        gaussian_model = GaussianModel(
            means=torch.tensor(self.gaussian_means, device=device, dtype=torch.float32),
            quats=torch.tensor(self.gaussian_quats, device=device, dtype=torch.float32),
            scales=torch.tensor(
                self.gaussian_scales, device=device, dtype=torch.float32
            ),
            opacities=torch.tensor(
                self.gaussian_opacities, device=device, dtype=torch.float32
            ),
            colors=torch.tensor(
                self.gaussian_colors, device=device, dtype=torch.float32
            ),
            body_ids=torch.tensor(
                self.gaussian_body_ids, device=device, dtype=torch.int32
            ),
        )
        return gaussian_model

    def finalize(self, device=None, requires_grad=False):
        if device is None:
            device = str(wp.get_preferred_device())
        model = super().finalize(device, requires_grad)
        model.rigid_contact_torsional_friction = 0.0  # type: ignore
        model.rigid_contact_rolling_friction = 0.0  # type: ignore
        self.gaussian_model = self.build_gaussian_model(device)
        self.gaussian_state = self.gaussian_model.state()
        return model
