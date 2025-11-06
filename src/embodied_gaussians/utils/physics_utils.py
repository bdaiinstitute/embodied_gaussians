# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import dataclasses
import pickle
from pathlib import Path

import numpy as np
import warp as wp
import warp.sim

def load_builder(path: Path | str):
    with open(path, "rb") as file:
        loaded_builder = pickle.load(file)
    return loaded_builder

def save_builder(path: Path | str, builder: warp.sim.ModelBuilder):
    for s in builder.shape_geo_src:
        if isinstance(s, warp.sim.Mesh) and hasattr(s, "mesh"):
            del s.mesh  # Cant be pickled and will be rebuilt anyway
    with open(path, "wb") as file:
        pickle.dump(builder, file)

def load_mesh(url: str):
    mesh_pts, mesh_indices = warp.sim.load_mesh(url)
    mesh = warp.sim.model.Mesh(
        mesh_pts.tolist(),  # type: ignore
        mesh_indices.flatten().astype(np.int32).tolist(),  # type: ignore
    )
    return mesh


def cuda_graph_capture(func):
    def wrapper(self, arg: dataclasses.dataclass):
        recompile = False
        if not hasattr(self, f"{func.__name__}_cache"):
            recompile = True
        else:
            cache = getattr(self, f"{func.__name__}_cache")
            if cache["arg"] != arg:
                recompile = True

        if recompile:
            with wp.ScopedCapture() as capture:
                func(self, arg)
            setattr(
                self,
                f"{func.__name__}_cache",
                {"arg": dataclasses.replace(arg), "graph": capture.graph},
            )
        graph = getattr(self, f"{func.__name__}_cache")["graph"]
        wp.capture_launch(graph)

    return wrapper


def clone_state(src: warp.sim.State):
    dest = warp.sim.State()
    if src.particle_count:
        dest.particle_q = wp.clone(src.particle_q)  # type: ignore
        dest.particle_qd = wp.clone(src.particle_qd)  # type: ignore
        dest.particle_f = wp.clone(src.particle_f)  # type: ignore

    if src.body_count:
        dest.body_q = wp.clone(src.body_q)  # type: ignore
        dest.body_qd = wp.clone(src.body_qd)  # type: ignore
        dest.body_f = wp.clone(src.body_f)  # type: ignore

    if src.joint_coord_count:
        dest.joint_q = wp.clone(src.joint_q)  # type: ignore
        dest.joint_qd = wp.clone(src.joint_qd)  # type: ignore
    return dest


def copy_state(dest: warp.sim.State, src: warp.sim.State):
    if src.particle_count:
        assert dest.particle_count == src.particle_count
        wp.copy(dest.particle_q, src.particle_q)  # type: ignore
        wp.copy(dest.particle_qd, src.particle_qd)  # type: ignore
        wp.copy(dest.particle_f, src.particle_f)  # type: ignore

    if src.body_count:
        assert dest.body_count == src.body_count
        wp.copy(dest.body_q, src.body_q)  # type: ignore
        wp.copy(dest.body_qd, src.body_qd)  # type: ignore
        wp.copy(dest.body_f, src.body_f)  # type: ignore

    if src.joint_coord_count:
        assert dest.joint_coord_count == src.joint_coord_count
        wp.copy(dest.joint_q, src.joint_q)  # type: ignore
        wp.copy(dest.joint_qd, src.joint_qd)  # type: ignore


def clone_control(src: warp.sim.Control):
    dest = warp.sim.Control()
    if src.joint_act is not None:
        dest.joint_act = wp.clone(src.joint_act)  # type: ignore
    return dest


def copy_control(dest: warp.sim.Control, src: warp.sim.Control):
    if src.joint_act is not None:
        assert dest.joint_act is not None
        assert src.joint_act.shape == dest.joint_act.shape
        wp.copy(dest.joint_act, src.joint_act)  # type: ignore


def transform_from_matrix(matrix: np.ndarray):
    quat = wp.quat_from_matrix(matrix[:3, :3])
    pos = matrix[:3, 3]
    return np.array([*pos, *quat], dtype=np.float32)


def transform_to_matrix(T: np.ndarray):
    T = wp.transformf(T[:3], T[3:])
    R = np.array(wp.quat_to_matrix(wp.transform_get_rotation(T))).reshape(3, 3)

    t = wp.transform_get_translation(T)
    X = np.eye(4, dtype=np.float32)
    X[:3, :3] = R
    X[:3, 3] = t
    return X


# def matrix_to_transform(X: np.ndarray):
#     q = wp.quat_from_matrix(X[:3, :3])
#     t = X[:3, 3]
#     return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])


def synchronize_state(
    dst_state: warp.sim.State,
    src_state: warp.sim.State,
):
    wp.launch(
        kernel=synchronize_state_kernel,
        dim=dst_state.body_count,
        inputs=[
            src_state.body_count,
            src_state.body_q,
            src_state.body_qd,
            dst_state.body_q,
            dst_state.body_qd,
        ],
    )


def synchronize_control(
    dst_control: warp.sim.Control,
    src_control: warp.sim.Control,
):
    wp.launch(
        kernel=synchronize_control_kernel,
        dim=dst_control.joint_act.shape[0],
        inputs=[
            src_control.joint_act.shape[0],
            src_control.joint_act,
            dst_control.joint_act,
        ],
    )


@wp.kernel
def synchronize_state_kernel(
    bodies_per_env: int,
    src_body_q: wp.array(dtype=wp.transformf),
    src_body_qd: wp.array(dtype=wp.spatial_vectorf),
    dst_body_q: wp.array(dtype=wp.transformf),
    dst_body_qd: wp.array(dtype=wp.spatial_vectorf),
):
    tid = wp.tid()
    src_id = tid % bodies_per_env
    dst_body_q[tid] = src_body_q[src_id]
    dst_body_qd[tid] = src_body_qd[src_id]


@wp.kernel
def synchronize_control_kernel(
    joints_per_env: int,
    src_joint_act: wp.array(dtype=wp.float32),
    dst_joint_act: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    src_id = tid % joints_per_env
    dst_joint_act[tid] = src_joint_act[src_id]
