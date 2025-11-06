# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import marsoom
import numpy as np
from scipy.spatial.transform import Rotation as R


class Canvas2D(marsoom.Viewer2D):
    def __init__(self, *args, **kwargs):
        self.pixels_to_units = np.array(
            [[0.0, 0.001, 0.0], [0.001, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        super().__init__(pixels_to_units=self.pixels_to_units, *args, **kwargs)

        self.tblock_vertices = [
            (-0.100, 0.025, 0.0),
            (0.100, 0.025, 0.0),
            (0.100, -0.025, 0.0),
            (0.025, -0.025, 0.0),
            (0.025, -0.175, 0.0),
            (-0.025, -0.175, 0.0),
            (-0.025, -0.025, 0.0),
            (-0.100, -0.025, 0.0),
            (-0.100, 0.025, 0.0),
        ]

    def draw_tblock(
        self,
        p_WO,
        q_WO,
        color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        thickness: float = 4,
    ):
        q_WO = np.asarray(q_WO)
        points = (
            R.from_quat(q_WO, scalar_first=False).apply(self.tblock_vertices) + p_WO
        )
        points = points[:, :2]
        self.polyline(
            positions=points,
            color=color,
            thickness=thickness,
            unit=marsoom.eViewerUnit.UNIT,
        )
