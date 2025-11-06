# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass, field

import marsoom
import marsoom.cuda
import pyglet
import torch
import trio
import tyro
import warp as wp
from marsoom import guizmo, imgui
from trio_util import periodic

from embodied_gaussians import PhysicsSettings
from embodied_gaussians.vis import SimulationViewer
from helpers.canvas_2d import Canvas2D
from sim_environments.pusht import PushTEnvironment


@dataclass
class Settings:
    """Configuration settings for the PushT simulation environment."""
    physics: PhysicsSettings = field(default_factory=lambda: PhysicsSettings())


class PushTGUI(marsoom.Window):
    """
    GUI application for the PushT environment, demonstrating object manipulation.
    
    This class provides a 2D and 3D visualization of a robotic pushing task,
    where a pusher can interact with a T-shaped block.
    """
    def __init__(self, settings: Settings):
        super().__init__(caption="PushT")
        self.settings = settings

        # Initialize environment and renderers
        self.environment = PushTEnvironment.build()
        self.sim_renderer = SimulationViewer(self)
        self.sim_renderer.set_simulator(self.environment.simulator())
        self.viewer_2d = Canvas2D(self)

        # Graphics and simulation state
        self.batch = pyglet.graphics.Batch()
        self.target_transform = wp.transformf((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        
        # Control parameters
        self.angle_threshold_degrees = 0.8  # Maximum allowed angular deviation
        self.distance_threshold = 0.01      # Maximum allowed positional error (meters)
        
        # State variables
        self.press_latch = False            # Track if mouse button is being held
        self.actions = self.environment.default_actions()
        self.observations = self.environment.observe()
        self.X_WO = self.environment.simulator().get_body_q(3)

    async def run(self):
        """Run the simulation and rendering loops concurrently."""
        async def physics_loop():
            """Updates physics simulation at a fixed timestep."""
            dt = self.settings.physics.dt
            async for _ in periodic(dt):
                self.environment.step()

        async def render_loop():
            """Updates visualization at 60 FPS."""
            render_fps = 60
            async for _ in periodic(1 / render_fps):
                if self.has_exit:
                    break
                self.step()

        async with trio.open_nursery() as n:
            n.start_soon(physics_loop)
            await render_loop()
            n.cancel_scope.cancel()

    def set_joints(self, x: float, y: float):
        """
        Set the desired position for the pusher.
        
        Args:
            x: Target x-coordinate in world space
            y: Target y-coordinate in world space
        """
        self.actions.pusher_desired_positions[:, :2] = torch.tensor([x, y]).cuda()
        self.environment.act(self.actions)

    def draw_2d_viewer(self):
        """Render the 2D interactive view of the environment."""
        imgui.begin("Control Plane")
        self.viewer_2d.draw()
        
        # Draw target transform
        self.viewer_2d.draw_tblock(
            wp.transform_get_translation(self.target_transform),
            wp.transform_get_rotation(self.target_transform),
            (0.0, 1.0, 0.0, 0.7),  # Semi-transparent green
        )
        
        # Get state for the first environment (env=0)
        env = 0
        joints = self.observations.pusher_positions.cpu().numpy()[env]
        tblock_transform = self.observations.tblock_transforms.cpu().numpy()[env]
        desired_joints = self.actions.pusher_desired_positions.cpu().numpy()[env]

        # Draw current state
        self.viewer_2d.draw_tblock(tblock_transform[:3], tblock_transform[3:])
        
        # Draw desired and current pusher positions
        hovered = self.viewer_2d.circle(
            position=desired_joints,
            color=(0, 1, 0, 1),    # Green for desired position
            radius=0.015,
            thickness=2,
            unit=marsoom.eViewerUnit.UNIT,
        )
        self.viewer_2d.circle(
            position=joints,
            color=(0, 0, 1, 1),    # Blue for current position
            radius=0.02,
            thickness=2,
            unit=marsoom.eViewerUnit.UNIT,
        )

        # Handle user interaction
        if hovered or self.press_latch:
            if imgui.is_mouse_down(0) or imgui.is_key_down(imgui.Key.space):
                self.press_latch = True
                x, y = self.viewer_2d.get_mouse_position(unit=marsoom.eViewerUnit.UNIT)
                self.set_joints(x, y)
            else:
                self.press_latch = False

        imgui.end()

    def reset(self):
        """Reset the environment to its initial state."""
        self.environment.reset()
        self.press_latch = False

    def render(self):
        """Render the complete GUI including controls and both 2D/3D views."""
        # Main control window with better styling and organization
        imgui.set_next_window_size((400, 300), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Control Panel", flags=imgui.WindowFlags_.no_collapse)
        
        # Status information in a colored frame
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.2, 0.2, 0.2, 1.0))
        imgui.begin_child("Status", (0, 100), True)
        imgui.text("Status Information")
        imgui.separator()
        t = self.environment.time()
        imgui.text(f"Simulation Time: {t:.2f}s")
        
        # Get current state info
        obs = self.observations
        tblock_transform = obs.tblock_transforms[0].cpu().numpy()
        imgui.text(f"T-Block Position: ({tblock_transform[0]:.2f}, {tblock_transform[1]:.2f})")
        imgui.end_child()
        imgui.pop_style_color()

        imgui.spacing()
        
        # Reset button with color
        button_size = (imgui.get_content_region_avail()[0], 30)
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.7, 0.2, 0.2, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.8, 0.3, 0.3, 1.0))
        if imgui.button("Reset##reset", button_size):
            self.reset()
        imgui.pop_style_color(2)

        # Help text
        imgui.spacing()
        imgui.text_wrapped("Click and drag in the 2D view to control the pusher position.")
        
        imgui.end()

        # Viewers with consistent sizing
        imgui.set_next_window_size((600, 400), cond=imgui.Cond_.first_use_ever)
        self.draw_2d_viewer()
        
        imgui.set_next_window_size((600, 400), cond=imgui.Cond_.first_use_ever)
        imgui.begin("3D Viewer", flags=imgui.WindowFlags_.no_collapse)
        with self.sim_renderer.draw(True):
            self.batch.draw()
            self.sim_renderer.render_meshes()
        self.sim_renderer.render_manipulation()
        self.sim_renderer.process_nav()
        imgui.end()


async def main():
    wp.init()
    settings = tyro.cli(Settings)
    window = PushTGUI(settings)
    await window.run()


if __name__ == "__main__":
    trio.run(main)
