# Copyright (c) 2025
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Spawn a single UR10e from a local USD and apply random joint efforts (arms.py style).

./isaaclab.sh -p nrs_lab/custom/run_ur10e_random_effort.py \
  --usd_path /home/eunseop/isaac/isaac_save/ur10e_convex_surface.usd \
  --effort_scale 3.0 --headless



"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments (arms.py 스타일)
parser = argparse.ArgumentParser(description="UR10e random-effort demo (tutorial-style).")
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the UR10e .usd file.")
parser.add_argument("--effort_scale", type=float, default=3.0, help="Scale for random torques (Nm).")
# AppLauncher 공통 인자 추가 (예: --headless, --device)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext

# physicsScene 보장 (없으면 생성)
import omni.usd
from pxr import UsdPhysics
_stage = omni.usd.get_context().get_stage()
if not _stage.GetPrimAtPath("/World/physicsScene"):
    UsdPhysics.Scene.Define(_stage, "/World/physicsScene")


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene (arms.py 스타일)."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 하나의 Origin 그룹
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # UR10e articulation (로컬 USD 사용)
    ur10e_cfg = ArticulationCfg(
        prim_path="/World/Origin.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            # 주의: 네 버전은 fixed_base 인자 미지원 → 전달하지 않음
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        # 토크 제어 위주 → PD 간섭 최소화
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=5.0,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    ur10e = Articulation(cfg=ur10e_cfg)

    scene_entities = {"ur10e": ur10e}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop (arms.py 스타일)."""
    robot = entities["ur10e"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0
            # root state (origin 보정)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # 초기 joint pos에 작은 잡음 (±0.1rad)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # 내부 버퍼 초기화
            robot.reset()
            print("[INFO]: Resetting UR10e state...")

        # ---- 무작위 토크 적용 (완만한 제한을 위해 tanh 사용) ----
        rand = torch.randn_like(robot.data.joint_pos)
        efforts = torch.tanh(rand) * args_cli.effort_scale   # [-effort_scale, +effort_scale] 근사
        robot.set_joint_effort_target(efforts)

        # write & step
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
