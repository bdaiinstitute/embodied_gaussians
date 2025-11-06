# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Frames:
    width: int
    height: int
    names: list[str]
    timestamps: list[float]
    Ks_cpu: torch.Tensor
    Ks_gpu: torch.Tensor
    X_WCs_cpu: torch.Tensor
    X_CWs_opencv_gpu: torch.Tensor
    colors_gpu: torch.Tensor
    device: str = "cuda"

    def update_colors(self, name: str, timestamp: float, color: torch.Tensor):
        index = self.names.index(name)
        assert color.shape == (self.height, self.width, 3)
        self.timestamps[index] = timestamp
        self.colors_gpu[index].copy_(color)


class FramesBuilder:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.names = []
        self.Ks = []
        self.X_WCs = []  # blender standard
        self.X_CWs_opencv = []  # opencv standard

    def add_camera(self, name: str, K: np.ndarray, X_WC: np.ndarray):
        self.names.append(name)
        self.Ks.append(torch.from_numpy(K))
        X_WC_opencv = X_WC @ np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        X_CW_opencv = np.linalg.inv(X_WC_opencv)
        self.X_WCs.append(torch.from_numpy(X_WC))
        self.X_CWs_opencv.append(torch.from_numpy(X_CW_opencv))

    def finalize(self, device="cuda"):
        num_frames = len(self.names)
        return Frames(
            width=self.width,
            height=self.height,
            names=self.names,
            device=device,
            timestamps=[-1.0] * num_frames,
            Ks_cpu=torch.stack(self.Ks).float(),
            Ks_gpu=torch.stack(self.Ks).float().to(device),
            X_WCs_cpu=torch.stack(self.X_WCs).float(),
            X_CWs_opencv_gpu=torch.stack(self.X_CWs_opencv).float().to(device),
            colors_gpu=torch.zeros(
                (num_frames, self.height, self.width, 3),
                dtype=torch.float32,
                device=device,
            ),
        )
