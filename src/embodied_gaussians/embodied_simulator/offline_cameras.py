# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import json
from pathlib import Path
import numpy as np
from .offline_camera import OfflineCamera


class OfflineCameras:
    def __init__(self, cameras: dict[str, OfflineCamera]):
        self.cameras = cameras

    def images(self, timestamp: float):
        res = {}
        for serial, cam in self.cameras.items():
            res[serial] = cam.image(timestamp)
        return res

    def resolution(self) -> tuple[int, int]:
        return next(iter(self.cameras.values())).resolution

    def keys(self):
        return self.cameras.keys()

    def values(self):
        return self.cameras.values()

    def items(self):
        return self.cameras.items()

    def __len__(self):
        return len(self.cameras)

    @staticmethod
    def from_dataset(path: Path):
        with open(path, "r") as f:
            d = json.load(f)
        cameras = {}
        for serial, camera_data in d.items():
            metadata_path = path.parent / camera_data["metadata_path"]
            with open(metadata_path, "r") as f:
                md = json.load(f)
            video_path = path.parent / camera_data["video_path"]
            X_WC = np.array(camera_data["X_WC"], dtype=np.float32)
            K = np.array(md["K"], dtype=np.float32)
            resolution = md["resolution"]
            timestamps = np.array(md["timestamps"], dtype=np.float32)
            cameras[serial] = OfflineCamera(
                video_path=video_path,
                K=K,
                X_WC=X_WC,
                resolution=resolution,
                timestamps=timestamps,
            )
        return OfflineCameras(cameras)
