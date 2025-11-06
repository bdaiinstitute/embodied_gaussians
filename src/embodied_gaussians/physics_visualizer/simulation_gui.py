# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import marsoom
import marsoom.overlay
from imgui_bundle import imgui
from trio_util import periodic

from embodied_gaussians.physics_simulator.simulator import Simulator
from embodied_gaussians.physics_visualizer.simulation_viewer import SimulationViewer


class SimulationGUI(marsoom.Window):
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
    ):
        super().__init__(width=width, height=height, caption="Visualizer")
        self.viewer_3d = SimulationViewer(self)
        self.callbacks_3d = []
        self.callbacks_render = []

    def set_simulation(self, simulator: Simulator):
        self.viewer_3d.set_simulator(simulator)

    async def run(self):
        async for _ in periodic(1 / 60):
            if self.should_exit():
                break
            self.step()

    def render(self):
        imgui.begin("3D Viewer")
        with self.viewer_3d.draw(in_imgui_window=True):
            self.viewer_3d.render_meshes()
        self.viewer_3d.render_manipulation()

        for callback in self.callbacks_3d:
            callback()

        self.viewer_3d.process_nav()
        imgui.end()

        for callback in self.callbacks_render:
            callback()
