# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import marsoom.overlay

from trio_util import periodic

from imgui_bundle import imgui

import marsoom

from embodied_gaussians.environments.embodied_environment import EmbodiedGaussiansEnvironment
from embodied_gaussians.embodied_visualizer.embodied_viewer import EmbodiedViewer


class EmbodiedGUI(marsoom.Window):
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        draw_controls: bool = True,
    ):
        super().__init__(width=width, height=height, caption="Visualizer")
        self.viewer_3d = EmbodiedViewer(self)
        self.draw_controls = draw_controls
        self.callbacks_3d = []
        self.callbacks_render = []

    def set_environment(self, environment: EmbodiedGaussiansEnvironment):
        self.viewer_3d.set_environment(environment)

    async def run(self):
        async for _ in periodic(1 / 60):
            if self.should_exit():
                break
            self.step()

    def render(self):
        imgui.begin("3D Viewer")
        with self.viewer_3d.draw(in_imgui_window=True):
            self.viewer_3d.render()

        self.viewer_3d.render_manipulation()
        for callback in self.callbacks_3d:
            callback()

        if self.draw_controls:
            self.viewer_3d.render_controls()
        self.viewer_3d.process_nav()
        imgui.end()

        for callback in self.callbacks_render:
            callback()
