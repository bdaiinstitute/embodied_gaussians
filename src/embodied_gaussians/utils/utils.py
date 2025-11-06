# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import typing

from pathlib import Path
from dataclasses import dataclass

import json

import numpy as np



@dataclass
class ExtrinsicsData:
    X_WC: np.ndarray

def read_extrinsics(path: Path) -> dict[str, ExtrinsicsData]:
    with open(path, "r") as f:
        extrinsics = json.load(f)
    res = {}
    for serial, data in extrinsics.items():
        res[serial] = ExtrinsicsData(
            X_WC=np.array(data["X_WT"])
        )
    return res

def read_ground(path: Path) -> np.array:
    with open(path, "r") as f:
        ground = json.load(f)
    return np.array(ground["plane"])

class GridBuilder:
    def __init__(self, max_cols: int = 10, spacing=1.0, z=0.0):
        self.max_cols = max_cols
        self.spacing = spacing
        self.z = z

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        col = self.num % self.max_cols
        row = self.num // self.max_cols
        x = col * self.spacing
        y = row * self.spacing
        self.num += 1
        return (x, y, self.z)