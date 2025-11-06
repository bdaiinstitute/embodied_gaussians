# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass

import torch
import numpy as np
import warp as wp

from embodied_gaussians.utils.physics_utils import transform_from_matrix
from embodied_gaussians.embodied_simulator.gaussians import GaussianState
from embodied_gaussians.embodied_simulator.simulator import render_gaussians


@dataclass
class VirtualCameras:
    width: int
    height: int
    near: float
    far: float
    parent_body: torch.Tensor
    names: list[str]
    K: torch.Tensor  # (num_envs, camera_number, 3, 3)
    T_BC: torch.Tensor  # (num_envs, camera_number, 7)
    X_CW_opencv: torch.Tensor  # (num_envs, camera_number, 4, 4)
    X_WC: torch.Tensor
    background: torch.Tensor
    rendered_images: torch.Tensor
    last_rendered_at: float = -1.0
    position_last_updated_at: float = -1.0

    def __post_init__(self):
        assert self.X_WC.shape == self.X_CW_opencv.shape
        assert self.X_WC.shape[1] == self.K.shape[0], (
            f"{self.X_WC[:1].shape} != {self.K[:1].shape}"
        )
        assert self.X_WC.shape[1] == self.T_BC.shape[0]
        assert self.T_BC.shape[-1] == 7
        self.K_cpu = self.K.cpu()
        self.streams = [torch.cuda.Stream() for _ in range(self.num_envs)]

    @property
    def num_cameras(self):
        return len(self.names)

    @property
    def num_envs(self):
        return self.X_WC.shape[0]

    def update_poses(self, timestamp: float, T_WB: torch.Tensor):
        assert T_WB.shape[0] == self.num_envs
        assert T_WB.shape[-1] == 7
        if self.position_last_updated_at == timestamp:
            return
        self.position_last_updated_at = timestamp
        wp.launch(
            kernel=update_poses_kernel,
            dim=[self.num_envs, self.num_cameras],
            inputs=[
                self.parent_body,
                T_WB,
                self.T_BC,
                self.X_CW_opencv,
                self.X_WC,
            ],
        )

    def render(
        self, timestamp: float, gaussian_state: GaussianState, force: bool = False
    ):
        if timestamp == self.last_rendered_at and not force:
            return
        c = self
        num_envs = self.num_envs
        gs = gaussian_state.reshape((num_envs, -1))
        for env in range(num_envs):
            with torch.cuda.stream(self.streams[env]):
                images, _, _ = render_gaussians(
                    gs.slice(slice(env, env + 1, None)).reshape((-1,)),
                    Ks=c.K,
                    X_CWs=c.X_CW_opencv[env],
                    width=c.width,
                    height=c.height,
                    background=c.background,
                    near_plane=c.near,
                    far_plane=c.far,
                    render_mode="RGB",
                )
                self.rendered_images[env] = images
        self.last_rendered_at = timestamp
        return c.rendered_images


class VirtualCamerasBuilder:
    def __init__(
        self,
        width: int,
        height: int,
        background: tuple[int, int, int] = (255, 255, 255),
        near: float = 0.01,
        far: float = 3.0,
    ):
        """
        background : (255, 255, 255)
        """
        self.width = width
        self.height = height
        self.Ks: list[np.ndarray] = []
        self.T_BC: list[np.ndarray] = []
        self.parent_body: list[int] = []
        self.names: list[str] = []
        self.background = background
        self.near = near
        self.far = far

    def add_camera(self, name: str, K: np.ndarray, X_BC: np.ndarray, body_id: int = -1):
        T_BC = transform_from_matrix(X_BC)
        self.Ks.append(K)
        self.T_BC.append(T_BC)
        self.names.append(name)
        self.parent_body.append(body_id)

    def finalize(self, num_envs: int, device: str = "cuda"):
        Ks = np.array(self.Ks)
        T_BCs = np.array(self.T_BC)
        num_cameras = len(self.names)
        return VirtualCameras(
            width=self.width,
            height=self.height,
            near=self.near,
            far=self.far,
            parent_body=torch.tensor(
                self.parent_body, dtype=torch.int32, device=device
            ),
            K=torch.from_numpy(Ks).float().to(device),
            T_BC=torch.from_numpy(T_BCs).float().to(device),
            X_CW_opencv=torch.zeros(
                (num_envs, num_cameras, 4, 4), dtype=torch.float32, device=device
            ),
            X_WC=torch.zeros(
                (num_envs, num_cameras, 4, 4), dtype=torch.float32, device=device
            ),
            names=self.names,
            rendered_images=torch.zeros(
                (num_envs, num_cameras, self.height, self.width, 3)
            ),
            background=torch.tensor(self.background).float().to(device) / 255.0,
        )


@wp.kernel
def update_poses_kernel(
    body_ids: wp.array(dtype=wp.int32),  # type: ignore
    T_WBs: wp.array(ndim=2, dtype=wp.transformf),  # type: ignore
    T_BCs: wp.array(dtype=wp.transformf),  # type: ignore
    X_CWs_opencv: wp.array(ndim=2, dtype=wp.mat44),  # type: ignore
    X_WCs: wp.array(ndim=2, dtype=wp.mat44),  # type: ignore
):
    env_id, tid = wp.tid()  # type: ignore
    bid = body_ids[tid]
    T_BC = T_BCs[tid]
    if bid < 0:
        T_WC = T_BC
    else:
        T_WB = T_WBs[env_id, bid]
        T_WC = T_WB * T_BC
    T_CW = wp.transform_inverse(T_WC)
    X_WC = wp.transform_to_matrix(T_WC)
    X_CW = wp.transform_to_matrix(T_CW)
    X_BLENDER_TO_OPENCV = wp.matrix_from_rows(
        wp.vec4f(1.0, 0.0, 0.0, 0.0),
        wp.vec4f(0.0, -1.0, 0.0, 0.0),
        wp.vec4f(0.0, 0.0, -1.0, 0.0),
        wp.vec4f(0.0, 0.0, 0.0, 1.0),
    ) 
    # X_BLENDER_TO_OPENCV = wp.mat44(
    #     1.0, 0.0, 0.0, 0.0,
    #     0.0, -1.0, 0.0, 0.0,
    #     0.0, 0.0, -1.0, 0.0,
    #     0.0, 0.0, 0.0, 1.0,
    # )
    X_WCs[env_id, tid] = X_WC
    X_CWs_opencv[env_id, tid] = wp.mul(X_BLENDER_TO_OPENCV, X_CW)
