# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from pydrake.trajectories import PiecewisePolynomial


class OfflineCamera:
    def __init__(
        self,
        video_path: Path,
        timestamps: np.ndarray,
        K: np.ndarray,
        X_WC: np.ndarray,
        resolution: tuple[int, int],
        device: str = "cuda",
    ):
        self.timestamps = timestamps
        self.K = K
        self._X_WC = X_WC
        self.video_path = video_path
        self.resolution = resolution
        self.device = device
        self.decoder = VideoDecoder(video_path, device=device, dimension_order="NHWC")
        self.last_index: int | None = None
        self.num_frames = self.decoder.metadata.num_frames
        self.index_look_up = PiecewisePolynomial.ZeroOrderHold(
            timestamps,
            np.arange(0, self.num_frames).astype(np.float32).reshape(-1, 1).T,
        )
        self.last_image = None

    def X_WC(self, timestamp: float) -> np.ndarray:
        return self._X_WC

    def image(self, timestamp: float) -> torch.Tensor:
        index = int(self.index_look_up.value(timestamp))
        if index == self.last_index:
            assert self.last_image is not None
            return self.last_image
        self.last_index = index
        color = self.decoder.get_frame_at(index).data / 255
        color = color[:, :, [2, 1, 0]]  # convert back to bgr
        self.last_image = color
        return self.last_image
