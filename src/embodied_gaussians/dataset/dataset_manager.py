# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import json
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.trajectories import PiecewisePolynomial
from typing_extensions import override

from embodied_gaussians import Body, EmbodiedGaussiansEnvironment, FramesBuilder, EmbodiedGaussiansLoader, OfflineCameras


@dataclass
class RobotData:
    control: np.ndarray
    control_timestamps: np.ndarray
    states: list[dict[str, typing.Any]]
    states_timestamps: np.ndarray
    state_index_look_up: PiecewisePolynomial
    control_index_look_up: PiecewisePolynomial

@dataclass
class CameraData:
    name: str
    X_WC: np.ndarray  # Camera pose in world frame
    K: np.ndarray     # Camera intrinsics
    resolution: tuple[int, int]
    video_path: Path
    timestamps: np.ndarray


class DatasetManager:
    def __init__(
        self,
        path: Path,
        camera_file: str | None = "cameras.json",
        load_frames: bool = True,
    ):
        self.path = path
        self.camera_file = camera_file
        self.physics_loader: EmbodiedGaussiansLoader | None = None
        self.load_frames = load_frames
        self.cameras: list[CameraData] = []
        self.real_robot_data_found = False
        self.camera_data_found = False
        self.initialize()

    def first_timestamp(self):
        min_timestamp = 0.0
        for r in self.robots.values():
            min_timestamp = min(r.states_timestamps[0], min_timestamp)
        return min_timestamp

    def last_timestamp(self):
        max_timestamp = 0.0
        for r in self.robots.values():
            max_timestamp = max(r.states_timestamps[-1], max_timestamp)
        return max_timestamp

    def duration(self):
        return self.last_timestamp() - self.first_timestamp()

    def can_build_environment(self):
        return self.physics_loader is not None

    def build_environment(self):
        assert self.physics_loader
        builder = self.physics_loader.builder
        env = EmbodiedGaussiansEnvironment(builder)  # type: ignore
        s = self.physics_loader.get_embodied_gaussian_state_at_index(0)
        env.sim.copy_embodied_gaussian_state(s)
        env.stash_state()
        return env

    def initialize(self):
        robots_path = self.path / "robots.json"
        if robots_path.exists():
            with open(robots_path, "r") as f:
                rs = json.load(f)
            self.real_robot_data_found = True
        else:
            rs = {}
            self.real_robot_data_found = False

        self.offline_cameras = None
        self.cameras = []

        if self.camera_file:
            path = self.path
            camera_manifest_path = path / self.camera_file
            if camera_manifest_path.exists():
                with open(camera_manifest_path, "r") as f:
                    camera_manifest = json.load(f)
                self.camera_data_found = True
                for serial, camera_data in camera_manifest.items():
                    metadata_path = path / camera_data["metadata_path"]
                    with open(metadata_path, "r") as f:
                        md = json.load(f)
                    _WC = np.array(camera_data["X_WC"], dtype=np.float32)
                    K = np.array(md["K"], dtype=np.float32)
                    resolution = tuple(md["resolution"])
                    video_path = path / camera_data["video_path"]
                    timestamps = np.array(md["timestamps"], dtype=np.float32)
                    
                    self.cameras.append(CameraData(
                        name=serial,
                        X_WC=_WC,
                        K=K,
                        resolution=resolution,
                        video_path=video_path,
                        timestamps=timestamps
                    ))


        self.robots = {}
        for robot_name, r in rs.items():
            ct = np.array(r["control_timestamps"], dtype=np.float32)
            st = np.array(r["states_timestamps"], dtype=np.float32)
            self.robots[robot_name] = RobotData(
                control=np.array(r["control"], dtype=np.float32),
                control_timestamps=ct,
                states=r["states"],
                states_timestamps=st,
                state_index_look_up=PiecewisePolynomial.ZeroOrderHold(
                    st,
                    np.arange(0, len(st)).astype(np.float32).reshape(-1, 1).T,
                ),
                control_index_look_up=PiecewisePolynomial.ZeroOrderHold(
                    ct,
                    np.arange(0, len(ct)).astype(np.float32).reshape(-1, 1).T,
                ),
            )
        self.try_load_physics()
        if self.load_frames and self.camera_data_found:
            self.offline_cameras = OfflineCameras.from_dataset(
                self.path / self.camera_file
            )
            self.initialize_frames()

    def initialize_frames(self):
        cameras = self.offline_cameras
        w, h = cameras.resolution()
        frame_builder = FramesBuilder(width=w, height=h)
        for serial, camera in cameras.items():
            K = camera.K
            X_WC = camera.X_WC(0.0)
            frame_builder.add_camera(serial, K, X_WC)
        self.frames = frame_builder.finalize()
        self.update_frames(0.0)

    def update_frames(self, timestamp: float):
        cameras = self.offline_cameras
        # ind = self.physics_loader.get_index_at_timestamp(timestamp)
        # timestamp = ind * 1 / 60.0
        images = cameras.images(timestamp)
        for serial in cameras.keys():
            self.frames.update_colors(serial, timestamp, images[serial])

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    def get_camera(self, index: int) -> CameraData:
        if self.num_cameras == 0:
            raise ValueError("No cameras available")
        if not 0 <= index < self.num_cameras:
            raise IndexError(f"Camera index {index} out of range")
        return self.cameras[index]

    def get_camera_by_name(self, name: str) -> CameraData:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(f"Camera with name {name} not found")

    def camera_timeseries(self, index: int):
        camera = self.get_camera(index)
        num_steps = len(camera.timestamps)
        return camera.timestamps, np.arange(0, num_steps).reshape(-1, 1, 1)

    def try_load_physics(self):
        physics_path = self.path / "physics.zarr"
        if physics_path.exists():
            self.physics_loader = EmbodiedGaussiansLoader()
            self.physics_loader.load(physics_path)
        else:
            print(f"Physics file not found at {physics_path}")

    def timestamps(self) -> np.ndarray:
        assert self.physics_loader
        return self.physics_loader.timestamps[:]  # type: ignore

    def progress(self) -> np.ndarray:
        assert self.physics_loader
        num_steps = self.physics_loader.num_steps
        progress = np.linspace(
            0.0, 1.0, self.physics_loader.num_steps, dtype=np.float32
        ).reshape(num_steps, 1, 1)
        return progress

    def joints_desired(self) -> np.ndarray:
        assert self.physics_loader
        num_steps = self.physics_loader.num_steps
        joints_desired = self.physics_loader.control_joint_act[:].reshape(
            num_steps, -1, 1
        )  # type: ignore
        return joints_desired

    def body_q(self) -> np.ndarray:
        assert self.physics_loader
        num_steps = self.physics_loader.num_steps
        return self.physics_loader.state_body_q[:].reshape(num_steps, -1, 7)

    def joints(self) -> np.ndarray:
        assert self.physics_loader
        num_steps = self.physics_loader.num_steps
        return self.physics_loader.state_joint_q[:].reshape(num_steps, -1, 1)

    def panda_state(self, timestamp: float):
        res = {}
        for robot_name, r in self.robots.items():
            index = int(r.state_index_look_up.value(timestamp))
            res[robot_name] = r.states[index]
        return res

    def controller_state(self, timestamp: float):
        res = {}
        for robot_name, r in self.robots.items():
            index = int(r.control_index_look_up.value(timestamp))
            res[robot_name] = r.control[index]
        return res

    def build_default_environment(self):
        raise NotImplementedError()


def get_body(path: Path) -> Body:
    assert path.suffix == ".json"
    with open(path, "r") as f:
        data = json.load(f)
    return Body.model_validate(data)
