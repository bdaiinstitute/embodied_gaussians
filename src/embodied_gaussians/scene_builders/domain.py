# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path
from typing import Literal
import numpy as np
import torch
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Posed:
    X_WC: np.ndarray # (Blender standard) (4, 4)

    def get_X_WC(self, format: Literal["opencv", "blender"] = "blender") -> np.ndarray:
        """Get the camera to world transform in the specified format"""

        if format == "opencv":
            X_WC = self.X_WC @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0.0, 0.0, 0.0, 1.0]])
            return X_WC

        return self.X_WC


@dataclass
class Image:
    K: np.ndarray # (3, 3)
    image: np.ndarray # (H, W, 3) in uint8
    format: Literal['rgb', 'bgr']

@dataclass
class Depth:
    depth: np.ndarray # (H, W) in float32 [0, 1]
    depth_scale: float

@dataclass
class Masked:
    mask: np.ndarray  # Mask from camera (H, W) where 0 is background and 1 is object and 2 is occlusion

@dataclass
class PosedImage(Posed, Image):
    pass

@dataclass
class PosedImageAndDepth(Posed, Image, Depth):
    pass

@dataclass
class MaskedPosedImageAndDepth(Masked, Posed, Image, Depth):
    pass

def save_posed_images(path: Path, posed_images):
    path = Path(path)
    assert path.suffix == ".npz" 
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, posed_images, allow_pickle=True)

def load_posed_images(path: Path):
    path = Path(path)
    return np.load(path, allow_pickle=True)["arr_0"]


class GaussianLearningRates(BaseModel):
    means: float = 0.001
    opacities: float = 0.001
    colors: float = 0.01
    quats: float = 0.01 
    scales: float = 0.01

class Ground(BaseModel):
    plane: tuple[float, float, float, float] =  (0.0, 0.0, 1.0, 0.0) # (4,) ax + by + cz + d = 0

    def normal(self) -> np.ndarray:
        return self.plane[:3]
    
    def offset(self) -> float:
        return -self.plane[3]

class Gaussians(BaseModel):
    means: list[list[float]] # (n_gaussians, 3)
    quats: list[list[float]] # (n_gaussians, 4) (w, x, y, z)
    scales: list[list[float]]# (n_gaussians, 3)
    opacities: list[float]# (n_gaussians,)
    colors: list[list[float]]# (n_gaussians, 3)

    def __len__(self):
        return len(self.means)
    
    def mask(self, mask: np.ndarray):
        return Gaussians(
            means=np.asarray(self.means)[mask].tolist(),
            quats=np.asarray(self.quats)[mask].tolist(),
            scales=np.asarray(self.scales)[mask].tolist(),
            opacities=np.asarray(self.opacities)[mask].tolist(),
            colors=np.asarray(self.colors)[mask].tolist()
        )

class Particles(BaseModel):
    means: list[list[float]] # (n_gaussians, 3)
    quats: list[list[float]] # (n_gaussians, 4) (w, x, y, z)
    radii: list[float]# (n_gaussians,)
    colors: list[list[float]]# (n_gaussians, 3)

    def __len__(self):
        return len(self.means)
    
    def mask(self, mask: np.ndarray):
        return Particles(
            means=np.asarray(self.means)[mask].tolist(),
            quats=np.asarray(self.quats)[mask].tolist(),
            radii=np.asarray(self.radii)[mask].tolist(),
            colors=np.asarray(self.colors)[mask].tolist()
        )

class Body(BaseModel):
    name: str
    X_WB: list[list[float]]
    gaussians: Gaussians | None = None
    particles: Particles | None = None


class GaussianActivations:
    quat = torch.nn.functional.normalize
    scale = torch.exp
    opacity = torch.sigmoid
    color = torch.sigmoid

    inv_scale = torch.log
    inv_opacity = torch.logit
    inv_color = torch.logit