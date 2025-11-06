# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from embodied_gaussians.scene_builders.domain import Body, Gaussians

def visualize(body: Body):
    o3d.visualization.draw_geometries(
        [
            o3d.geometry.TriangleMesh.create_coordinate_frame(0.1),
            *sphere_meshes(body.particles.means, body.particles.radii[0], body.particles.colors),
            *ellipsoid_meshes(body.gaussians),
        ]
    )

def ellipsoid_meshes(gaussians: Gaussians):
    ellipsoid_meshes = []
    for i, mean in enumerate(gaussians.means):
        ellipsoid: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        ellipsoid.transform(np.array(
            [[gaussians.scales[i][0], 0, 0, 0],
            [0, gaussians.scales[i][1], 0, 0],
            [0, 0, gaussians.scales[i][2], 0],
            [0, 0, 0, 1]]
        )
        )
        ellipsoid.compute_vertex_normals()
        ellipsoid.paint_uniform_color(gaussians.colors[i])
        ellipsoid.translate(mean)
        quat = gaussians.quats[i]
        r = R.from_quat(quat, scalar_first=True).as_matrix()
        ellipsoid.rotate(r)
        ellipsoid_meshes.append(ellipsoid)
    return ellipsoid_meshes


def sphere_meshes(means, radius, colors: np.ndarray | None = None):
    sphere_meshes = []
    for i, mean in enumerate(means):
        sphere: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        if colors is not None:
            sphere.paint_uniform_color(colors[i])
        else:
            sphere.paint_uniform_color([0, 1, 0])
        sphere.translate(mean)
        sphere_meshes.append(sphere)
    return sphere_meshes