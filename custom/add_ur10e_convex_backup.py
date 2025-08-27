# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""
Spawn a local UR10e USD as an Articulation in Isaac Lab and set initial joint positions/velocities.

Usage:
  ./isaaclab.sh -p scripts/custom/add_ur10e_convex.py \
      --usd_path /home/eunseop/isaac/isaac_save/ur10e_convex_surface.usd \
      --num_envs 1 --headless

Tested with: Isaac Sim 4.5.0, Isaac Lab 2.0
"""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(description="Add local UR10e USD as Articulation with initial joint state.")
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the UR10e USD.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of cloned environments.")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Spacing between env origins.")
# Isaac Lab app args (e.g., --headless, --renderer)
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
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Ensure physicsScene exists (avoid "scene not found" logs)
import omni.usd
from pxr import UsdPhysics

# ---------------------------
# UR10e joint initial state (from user-provided list)
# ---------------------------
UR10E_INIT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": 0.0011,
    "elbow_joint": 0.0001,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0004,
    "wrist_3_joint": 0.0,
}
UR10E_INIT_JOINT_VEL = {
    "shoulder_pan_joint": 0.0001,
    "shoulder_lift_joint": -0.0159,
    "elbow_joint": 0.0319,
    "wrist_1_joint": -0.0217,
    "wrist_2_joint": 0.0371,
    "wrist_3_joint": 0.1543,
}

# ---------------------------
# Articulation config for UR10e (local USD)
# ---------------------------
UR10E_CONFIG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/UR10e",   # ex) /World/envs/env_0/UR10e
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.usd_path,
        # 필요 시 rigid/articulation properties 추가
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    # 초기 상태: root pose(원하면 pos/rot 수정) + joint pos/vel
    init_state=ArticulationCfg.InitialStateCfg(
        # Stage 월드 좌표에서 로봇 베이스(루트) 위치/자세
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # quaternion (w,x,y,z) : no rotation
        joint_pos=UR10E_INIT_JOINT_POS,
        joint_vel=UR10E_INIT_JOINT_VEL,
    ),
    # 간단히 모든 조인트에 동일한 implicit actuator 설정
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],   # 모든 조인트
            effort_limit_sim=200.0,    # 필요 시 조정
            velocity_limit_sim=5.0,    # 필요 시 조정
            stiffness=8000.0,          # 필요 시 조정
            damping=100.0,             # 필요 시 조정
        )
    },
)

# ---------------------------
# Scene config
# ---------------------------
@configclass
class UR10eSceneCfg(InteractiveSceneCfg):
    """Ground + Light + UR10e articulation"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    )

    ur10e = UR10E_CONFIG


# ---------------------------
# Run loop
# ---------------------------
def run(sim: SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # 주기적 리셋(예: 500 스텝마다)
        if count % 500 == 0:
            count = 0
            # 각 env 원점으로 루트 상태 offset
            root_state = scene["ur10e"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins

            # 루트 포즈/속도 초기화
            scene["ur10e"].write_root_pose_to_sim(root_state[:, :7])
            scene["ur10e"].write_root_velocity_to_sim(root_state[:, 7:])

            # 조인트 초기 상태 복사 (init_state에서 정의된 값들이 default_*에 들어있음)
            jpos = scene["ur10e"].data.default_joint_pos.clone()
            jvel = scene["ur10e"].data.default_joint_vel.clone()
            scene["ur10e"].write_joint_state_to_sim(jpos, jvel)

            # 내부 버퍼 초기화
            scene.reset()
            print("[INFO] Reset UR10e to initial joint state.")

        # 여기서 제어 입력을 주고 싶다면 set_joint_position_target / set_joint_velocity_target 사용
        # 예: 유지(무동작)
        # scene["ur10e"].set_joint_position_target(scene["ur10e"].data.default_joint_pos)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    # Simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 2.0, 2.5], target=[0.0, 0.0, 0.6])

    # physicsScene 보장 (없으면 생성)
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World/physicsScene"):
        UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        print("[INFO] Added missing /World/physicsScene prim.")

    # Scene build
    scene_cfg = UR10eSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        replicate_physics=True
    )
    scene = InteractiveScene(scene_cfg)

    # Sim start
    sim.reset()
    print(f"[INFO] Spawned UR10e from: {args_cli.usd_path}")
    print(f"[INFO] Envs: {scene.num_envs}, Origins: {scene.env_origins}")

    # Run
    run(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
