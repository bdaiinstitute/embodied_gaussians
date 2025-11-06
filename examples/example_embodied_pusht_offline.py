# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import trio
from pathlib import Path
import warp as wp
import torch
from marsoom import imgui
from embodied_environments.pusht_embodied.pusht_embodied import build_environment
from embodied_gaussians import DatasetManager, EmbodiedGaussiansEnvironment
from embodied_gaussians.vis import EmbodiedGUI  


class PlaybackControls:
    def __init__(self, environment: EmbodiedGaussiansEnvironment, dataset_manager: DatasetManager, fps: int):
        self.current_timestep = 0.0
        self.playing = False
        self.environment = environment
        self.dataset_manager = dataset_manager
        self.first_state = environment.sim.clone_embodied_gaussian_state()
        self.fps = fps
    
    def reset(self):
        self.current_timestep = 0.0
        self.environment.sim.copy_embodied_gaussian_state(self.first_state)
        self.environment.sim.eval_ik()
        self.go_to_timestep(0.0)
    
    def go_to_timestep(self, timestep: float):
        self.current_timestep = timestep
        q = self.dataset_manager.panda_state(timestep)["sheep"]["q"]
        q= torch.tensor(q, device=self.environment.sim.device)
        self.environment.set_robot_desired_q(0, q)
        self.dataset_manager.update_frames(timestep)

    def draw(self):
        imgui.begin("Playback")
        imgui.text(f"Current timestep: {self.current_timestep:.2f}")
        imgui.text(f"Playing: {self.playing}")
        _, self.fps = imgui.slider_int("FPS", self.fps, 1, 120)
        if imgui.button("Play"):
            self.playing = True
        imgui.same_line()
        if imgui.button("Pause"):
            self.playing = False
        imgui.same_line()
        if imgui.button("Reset"):
            self.reset()
        imgui.end()
    
    async def run_physics(self):
        dt = self.environment.dt()
        while True:
            self.environment.step()
            await trio.sleep(dt)
    
    async def run(self):
        async with trio.open_nursery() as n:
            n.start_soon(self.run_physics)
            while True:
                if self.playing:
                    self.current_timestep += 1/self.fps
                    self.go_to_timestep(self.current_timestep)
                await trio.sleep(1/self.fps)

async def main():
    environment = build_environment()
    environment.visual_forces_settings.iterations = 7

    current_dir = Path(__file__).parent
    dataset_manager = DatasetManager(current_dir / Path("embodied_environments/pusht_embodied/sample_demos/0"))
    environment.frames = dataset_manager.frames
    fps = 60
    playback_controls = PlaybackControls(environment, dataset_manager, fps)
    playback_controls.reset()


    visualizer = EmbodiedGUI()
    visualizer.set_environment(environment)
    visualizer.callbacks_render.append(playback_controls.draw)


    async with trio.open_nursery() as n:
        n.start_soon(playback_controls.run)
        await visualizer.run()
        n.cancel_scope.cancel()


if __name__ == "__main__":
    wp.config.quiet = True
    wp.init()
    trio.run(main)