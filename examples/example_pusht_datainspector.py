# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from pathlib import Path

import marsoom
import marsoom.cuda
import trio
import tyro
import warp as wp
from marsoom import imgui
from trio_util import periodic

from embodied_gaussians.physics_simulator.loader import Loader
from sim_environments.pusht import PushTEnvironment
from embodied_gaussians.vis import SimulationViewer

@dataclass
class Settings:
    path: tyro.conf.PositionalRequiredArgs[Path]
    """Path to the data directory"""

class InspectGUI(marsoom.Window):
    def __init__(self, settings: Settings):
        super().__init__(caption="Inspector")
        self.settings = settings
        
        # Initialize environment and viewers
        self.pusht_env = PushTEnvironment.build()
        self.loader = Loader(device="cuda")
        self.sim_renderer = SimulationViewer(self)
        self.sim_renderer.set_simulator(self.pusht_env.simulator())
        
        # State tracking
        self.time_index = 0
        self.playing = False
        self.play_nursery = None
        self.nursery = None
        
        # Load first demo
        self.demo_paths = sorted([d for d in self.settings.path.iterdir() if d.is_dir()], 
                               key=lambda x: int(x.name))
        self.demo_number = 0
        if self.demo_paths:
            self.load_demo(0)

    def load_demo(self, index: int):
        print(f"Loading demo {index}")
        self.demo_number = index
        self.loader.reset()
        self.loader.load(self.demo_paths[index])
        self.time_index = 0
        self.go_to_index(self.time_index)

    def go_to_index(self, index: int):
        state, control = self.loader.get_state_at_index(index)
        self.pusht_env.simulator().set_state(state)
        self.pusht_env.simulator().set_control(control)

    async def start_playing(self):
        assert not self.playing
        self.playing = True
        async with trio.open_nursery() as self.play_nursery:
            async for _ in periodic(1 / 60.0):
                self.time_index += 1
                self.time_index %= self.loader.num_steps
                self.go_to_index(self.time_index)

    def stop_playing(self):
        if self.play_nursery is None:
            return
        self.playing = False
        self.play_nursery.cancel_scope.cancel()

    async def run(self):
        async def render_loop():
            render_fps = 60
            async for _ in periodic(1 / render_fps):
                if self.has_exit:
                    break
                self.step()

        async with trio.open_nursery() as self.nursery:
            await render_loop()
            self.nursery.cancel_scope.cancel()

    def keyboard(self):
        if imgui.is_key_pressed(imgui.Key.space):
            if self.playing:
                self.stop_playing()
            else:
                self.nursery.start_soon(self.start_playing)

    def render(self):
        self.keyboard()
        
        # Main control window with better styling
        imgui.set_next_window_size((400, 300), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Control Panel", flags=imgui.WindowFlags_.no_collapse)
        
        # Status information
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))
        imgui.begin_child("Status", (0, 100), True)
        imgui.text("Playback Controls")
        imgui.separator()
        
        c, self.time_index = imgui.slider_int(
            "Time", self.time_index, 0, self.loader.num_steps - 1
        )
        if c:
            self.time_index = min(max(0, self.time_index), self.loader.num_steps - 1)
            self.go_to_index(self.time_index)

        button_size = (imgui.get_content_region_avail()[0] / 2 - 5, 30)
        if not self.playing:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.6, 0.2, 1.0))
            if imgui.button("Play", button_size):
                self.nursery.start_soon(self.start_playing)
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.7, 0.2, 0.2, 1.0))
            if imgui.button("Stop", button_size):
                self.stop_playing()
            imgui.pop_style_color()
            
        imgui.end_child()
        imgui.pop_style_color()

        imgui.spacing()
        imgui.text(f"Demo number: {self.demo_number}")
        imgui.separator()

        # Demo selection list
        imgui.begin_child("Demos", (0, 0), True)
        flags = imgui.SelectableFlags_.allow_double_click.value
        for i, path in enumerate(self.demo_paths):
            selected = i == self.demo_number
            clicked, _selected = imgui.selectable(f"Demo {path.name}", selected, flags=flags)
            if clicked and not selected:
                self.load_demo(i)
        imgui.end_child()
        
        imgui.end()

        # 3D Viewer
        imgui.begin("3D Viewer")
        with self.sim_renderer.draw(True):
            self.sim_renderer.render_meshes()
        self.sim_renderer.process_nav()
        imgui.end()

async def main():
    wp.init()
    settings = tyro.cli(Settings)
    window = InspectGUI(settings)
    await window.run()

if __name__ == "__main__":
    trio.run(main)
