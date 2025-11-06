# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import pickle
import warnings
from pathlib import Path

import torch
import warp as wp
import warp.sim
import zarr

from embodied_gaussians.physics_simulator.simulator import Simulator
from embodied_gaussians.utils.physics_utils import save_builder



class Saver:
    def __init__(self, simulator: Simulator, device: str, max_frames: int = 10_000):
        self.simulator = simulator
        self.current_index = 0
        self.device = device
        self.builder = self.simulator.builder
        self.max_frames = 0
        self.allocate_demo(max_frames=max_frames, device=self.device)
        self.current_index = 0

    def at_capacity(self):
        return self.current_index == self.max_frames

    def save(self, path: Path):
        path.mkdir(exist_ok=False, parents=True)
        t = self.current_index
        p = self.builder.particle_count
        b = self.builder.body_count
        j = len(self.builder.joint_act)
        if t == 0:
            warnings.warn("No timestamps recorded but save was triggered")
            return
        if p == 0 and b == 0:
            warnings.warn("No particles or bodies found. Not saving.")
            return

        root = zarr.group(path)
        save_builder(f"{path}/builder.pckl", self.builder)
        s = root.create_array(name="timestamps", shape=(t,), dtype="f4")
        s[:] = self.timestamps.numpy()
        if p > 0:
            s = root.create_array(name="state_particle_q", shape=(t, p, 3), dtype="f4")
            s[:] = self.state_particle_q.cpu().numpy()
            s = root.create_array(name="state_particle_qd", shape=(t, p, 3), dtype="f4")
            s[:] = self.state_particle_qd.cpu().numpy()
            s = root.create_array(name="state_particle_f", shape=(t, p, 3), dtype="f4")
            s[:] = self.state_particle_f.cpu().numpy()
        if b > 0:
            s = root.create_array(name="state_body_q", shape=(t, b, 7), dtype="f4")
            s[:] = self.state_body_q.cpu().numpy()
            s = root.create_array(name="state_body_qd", shape=(t, b, 6), dtype="f4")
            s[:] = self.state_body_qd.cpu().numpy()
            s = root.create_array(name="state_body_f", shape=(t, b, 6), dtype="f4")
            s[:] = self.state_body_qd.cpu().numpy()
        if j > 0:
            s = root.create_array(name="state_joint_q", shape=(t, j), dtype="f4")
            s[:] = self.state_joint_q.cpu().numpy()
            s = root.create_array(name="state_joint_qd", shape=(t, j), dtype="f4")
            s[:] = self.state_joint_qd.cpu().numpy()
            s = root.create_array(name="control_joint_act", shape=(t, j), dtype="f4")
            s[:] = self.control_joint_act.cpu().numpy()
        return root

    def allocate_demo(self, max_frames: int, device: str):
        if self.max_frames == max_frames:
            return
        self.max_frames = max_frames

        t = max_frames
        self.timestamps = torch.zeros(max_frames, device="cpu")
        if self.builder.particle_count > 0:
            p = self.builder.particle_count
            self.state_particle_q = torch.zeros(
                (t, p, 3), dtype=torch.float32, device=device
            )
            self.state_particle_qd = torch.zeros(
                (t, p, 3), dtype=torch.float32, device=device
            )
            self.state_particle_f = torch.zeros(
                (t, p, 3), dtype=torch.float32, device=device
            )
        if self.builder.body_count > 0:
            b = self.builder.body_count
            self.state_body_q = torch.zeros(
                (t, b, 7), dtype=torch.float32, device=device
            )
            self.state_body_qd = torch.zeros(
                (t, b, 6), dtype=torch.float32, device=device
            )
            self.state_body_f = torch.zeros(
                (t, b, 6), dtype=torch.float32, device=device
            )
        j = len(self.builder.joint_act)
        if j:
            self.state_joint_q = torch.zeros((t, j), dtype=torch.float32, device=device)
            self.state_joint_qd = torch.zeros(
                (t, j), dtype=torch.float32, device=device
            )
            self.control_joint_act = torch.zeros(
                (t, j), dtype=torch.float32, device=device
            )

    def clear_allocation(self):
        self.current_index = 0
        if self.builder.particle_count > 0:
            self.state_particle_q.zero_()
            self.state_particle_qd.zero_()
            self.state_particle_f.zero_()
        if self.builder.body_count > 0:
            self.state_body_q.zero_()
            self.state_body_qd.zero_()
            self.state_body_f.zero_()
        j = len(self.builder.joint_act)
        if j > 0:
            self.state_joint_q.zero_()
            self.state_joint_qd.zero_()
            self.control_joint_act.zero_()

    def record_state_and_control(self, time: float):
        self._record_state_and_control(
            time, self.simulator.state_0, self.simulator.control
        )

    def _record_state_and_control(
        self,
        timestamp: float,
        state: warp.sim.State,
        control: warp.sim.Control,
    ):
        i = self.current_index
        if i >= self.max_frames:
            logging.warning("Cannot save because the buffer is full.")
            return
        # p = self.builder.particle_count
        # b = self.builder.body_count
        j = len(self.builder.joint_act)

        self.timestamps[i] = timestamp
        if self.builder.particle_count > 0:
            self.state_particle_q[i] = wp.to_torch(state.particle_q)
            self.state_particle_qd[i] = wp.to_torch(state.particle_qd)
            self.state_particle_f[i] = wp.to_torch(state.particle_f)
        if self.builder.body_count > 0:
            self.state_body_q[i] = wp.to_torch(state.body_q)
            self.state_body_qd[i] = wp.to_torch(state.body_qd)
            self.state_body_f[i] = wp.to_torch(state.body_f)
        if j:
            self.state_joint_q[i] = wp.to_torch(state.joint_q)
            self.state_joint_qd[i] = wp.to_torch(state.joint_qd)
            self.control_joint_act[i] = wp.to_torch(control.joint_act)
        self.current_index += 1
