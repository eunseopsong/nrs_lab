# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""
Spawn a local UR10e USD as an Articulation in Isaac Lab, with fixed base and custom initial state.

Usage:
  ./isaaclab.sh -p scripts/custom/add_ur10e_articulation_fixed.py \
      --usd_path /home/USER/isaac/isaac_save/ur10e_convex_surface.usd \
      --num_envs 1 --headless

Tested with: Isaac Sim 4.5.0, Isaac Lab 2.0
"""

import argparse
import math

from isaaclab.app import AppLauncher

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(description="UR10e articulation with fixed base and custom initial joint state.")
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the UR10e USD.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of cloned environments.")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Spacing between env origins.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---------------------------
# Launch Omniverse app
# ---------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # keep alive

# ---------------------------
# Imports (after app up)
# ---------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# ensure physics scene prim exists
import omni.usd
from pxr import UsdPhysics

# ---------------------------
# UR10e desired initial state
#   joint pos (deg):  [0, -90, -90, -90, 90, 0]
#   joint vel:        all zeros
# ---------------------------
UR10E_INIT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -math.pi / 2.0,
    "elbow_joint": -math.pi / 2.0,
    "wrist_1_joint": -math.pi / 2.0,
    "wrist_2_joint":  math.pi / 2.0,
    "wrist_3_joint": 0.0,
}
UR10E_INIT_JOINT_VEL = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": 0.0,
    "elbow_joint": 0.0,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
}

# ---------------------------
# Articulation config (fixed base)
# ---------------------------
UR10E_CONFIG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/UR10e",   # /World/envs/env_*/UR10e
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fixed_base=True,                    # ✅ 베이스 고정
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 베이스 루트 포즈(원하면 높이/자세 수정)
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # (w,x,y,z)
        joint_pos=UR10E_INIT_JOINT_POS,
        joint_vel=UR10E_INIT_JOINT_VEL,
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=5.0,
            stiffness=8000.0,
            damping=100.0,
        )
    },
)

# ---------------------------
# Scene config
# ---------------------------
@configclass
class UR10eSceneCfg(InteractiveSceneCfg):
    """Ground + Light + UR10e articulation (fixed base)."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    )

    ur10e = UR10E_CONFIG


def run(sim: SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        # 주기적 리셋 예시 (원치 않으면 제거)
        if count % 500 == 0:
            count = 0
            root_state = scene["ur10e"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["ur10e"].write_root_pose_to_sim(root_state[:, :7])
            scene["ur10e"].write_root_velocity_to_sim(root_state[:, 7:])
            jpos = scene["ur10e"].data.default_joint_pos.clone()
            jvel = scene["ur10e"].data.default_joint_vel.clone()
            scene["ur10e"].write_joint_state_to_sim(jpos, jvel)
            scene.reset()
            print("[INFO] Reset UR10e to fixed-base initial state.")

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    # Simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 2.0, 2.5], target=[0.0, 0.0, 0.6])

    # ensure /World/physicsScene exists
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/physicsScene"):
        UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        print("[INFO] Added missing /World/physicsScene prim.")

    # build scene
    scene_cfg = UR10eSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        replicate_physics=True
    )
    scene = InteractiveScene(scene_cfg)

    # start sim
    sim.reset()
    print(f"[INFO] Spawned UR10e from: {args_cli.usd_path}")
    print(f"[INFO] Envs: {scene.num_envs}, Origins: {scene.env_origins}")

    # loop
    run(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
