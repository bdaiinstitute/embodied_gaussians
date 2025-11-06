# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from pathlib import Path

import trio
import trio.testing
import tyro
import warp as wp
from marsoom import guizmo, imgui
from trio_util import periodic
from typing_extensions import override

from embodied_gaussians import DatasetManager
from embodied_gaussians.vis import EmbodiedGUI


@dataclass
class Params:
    path: tyro.conf.PositionalRequiredArgs[Path]



class DatasetVisualizer(EmbodiedGUI):
    def __init__(self, path: Path):
        super().__init__()
        self.demos_path = path
        self.index_selected = -1
        self.get_demo_dirs()
        self.dataset = None
        self.current_step = 0

    def get_demo_dirs(self):
        path = self.demos_path
        dirs = sorted(list(path.iterdir()))
        res = []
        for dir in dirs:
            res.append(dir)
        self.demo_dirs = res

    def load_demo(self, demo_dir: Path):
        self.current_demo_path = demo_dir
        self.dataset = dataset = DatasetManager(
            demo_dir, camera_file="cameras.json"
        )
        if dataset.can_build_environment():
            env = dataset.build_environment()
            self.current_step = 0
        else:
            # env = dataset.build_default_environment()
            raise RuntimeError("Cannot build environment")
        self.viewer_3d.reset_cameras()
        self.environment = env
        if dataset.camera_data_found:
            self.environment.set_frames(dataset.frames)
        self.set_environment(env)

    def go_to_step(self, step: int):
        assert self.dataset and self.dataset.physics_loader
        loader = self.dataset.physics_loader
        s = loader.get_embodied_gaussian_state_at_index(step)
        t = loader.timestamp_at_index(step)
        if self.dataset.camera_data_found:
            self.dataset.update_frames(t)
        self.environment.sim.copy_embodied_gaussian_state(s)
        self.current_step = step

    def draw_demo_selector(self):
        imgui.begin("Demos")
        flags = imgui.SelectableFlags_.allow_double_click.value
        for i, demo_dir in enumerate(self.demo_dirs):
            name = f"{demo_dir.name}"
            c, s = imgui.selectable(name, self.index_selected == i, flags=flags)
            if c:
                if imgui.is_mouse_double_clicked(0):
                    self.load_demo(demo_dir)
                    self.index_selected = i
        imgui.end()

    @override
    def render(self):
        super().render()
        self.draw_demo_selector()
        imgui.begin("Controls")
        imgui.spacing()
        if self.dataset and self.dataset.physics_loader is not None:
            imgui.separator_text("Loader")
            num_steps = self.dataset.physics_loader.num_steps
            c, new_step = imgui.slider_int("Step", self.current_step, 0, num_steps - 1)
            if c:
                self.go_to_step(new_step)
        imgui.end()

    async def loop(self):
        async with trio.open_nursery() as self.nursery:
            async for _ in periodic(1 / 60):
                if self.should_exit():
                    break
                self.step()
            self.nursery.cancel_scope.cancel()


if __name__ == "__main__":
    wp.init()
    params = tyro.cli(Params)
    vis = DatasetVisualizer(params.path)
    trio.run(vis.loop)
