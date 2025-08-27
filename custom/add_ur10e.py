# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""
Spawn a local USD (UR10e + convex surface) into an Isaac Lab InteractiveScene.

Usage:
  ./isaaclab.sh -p scripts/custom/add_ur10e.py \
      --usd_path /home/eunseop/isaac/isaac_save/ur10e_convex_surface.usd \
      --num_envs 1

Tested with: Isaac Sim 4.5.0, Isaac Lab 2.0
"""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------
# 1) Parse CLI
# ---------------------------
parser = argparse.ArgumentParser(description="Add a local USD asset into Isaac Lab.")
parser.add_argument("--usd_path", type=str, required=False,
                    default="/home/eunseop/isaac/isaac_save/ur10e_convex_surface.usd",
                    help="Absolute path to the USD file to spawn.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to clone.")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Spacing between env origins.")
# Append AppLauncher args (e.g., --headless)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------
# 2) Launch the app
# ---------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # noqa: F401 (needed to keep app alive)

# ---------------------------
# 3) Imports that need app up
# ---------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

# USD stage 접근용
import omni.usd
from pxr import UsdPhysics

# ---------------------------
# 4) Scene Config
# ---------------------------
@configclass
class UR10eConvexSurfaceSceneCfg(InteractiveSceneCfg):
    """Scene with ground, light, and a user-provided USD prim."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # Simple dome light
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    )

    # User USD asset
    ur10e_convex = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10eConvex",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd_path,
        )
    )

# ---------------------------
# 5) Main
# ---------------------------
def main():
    # (a) Simulation context
    sim_cfg = SimulationCfg(dt=0.01)  # 100 Hz physics
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 3.5, 2.5], target=[0.0, 0.0, 0.5])

    # (a-1) Physics scene prim 보장
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/physicsScene"):
        UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        print("[INFO] Added missing /World/physicsScene prim.")

    # (b) Build scene
    scene_cfg = UR10eConvexSurfaceSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        replicate_physics=True
    )
    scene = InteractiveScene(cfg=scene_cfg)

    # (c) Play / reset
    sim.reset()
    print(f"[INFO] Spawned: {args_cli.usd_path}")
    print(f"[INFO] Envs: {scene.num_envs}, Origins: {scene.env_origins}")

    # (d) Sim loop
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
