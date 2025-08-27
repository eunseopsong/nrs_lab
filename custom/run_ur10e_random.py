# Copyright (c) 2025
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
UR10e from a local USD, random interaction (arms.py style).
- control=position : 데모처럼 joint position targets(±noise) + PD
- control=effort   : 무작위 토크 (tanh로 완만 제한)

./isaaclab.sh -p nrs_lab/custom/run_ur10e_random.py \
  --usd_path /home/eunseop/isaac/isaac_save/ur10e_convex_surface.usd \
  --control position --noise 0.10 --headless

"""

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# ---- CLI (arms.py 스타일) ----
parser = argparse.ArgumentParser(description="UR10e random interaction (tutorial-style).")
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the UR10e .usd file.")
parser.add_argument("--control", type=str, default="position", choices=["position", "effort"],
                    help="Interaction mode (default: position).")
parser.add_argument("--noise", type=float, default=0.10, help="Position noise amplitude in rad (position mode).")
parser.add_argument("--effort_scale", type=float, default=3.0, help="Torque scale (effort mode).")
# App launcher 공통 인자 (예: --headless, --device)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch Omniverse app
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

# physicsScene 보장
import omni.usd
from pxr import UsdPhysics
_stage = omni.usd.get_context().get_stage()
if not _stage.GetPrimAtPath("/World/physicsScene"):
    UsdPhysics.Scene.Define(_stage, "/World/physicsScene")

# ---- 액추에이터 설정: 모드별 PD 게인 ----
if args_cli.control == "position":
    # 데모 느낌: 적당한 PD로 목표자세를 따라가서 안정적
    ACTUATOR = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=200.0,
        velocity_limit_sim=5.0,
        stiffness=3000.0,   # 필요시 1500~5000 사이에서 조절
        damping=60.0,       # 필요시 20~100 사이에서 조절
    )
else:
    # 토크 모드: PD 끄고 토크만 적용
    ACTUATOR = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=200.0,
        velocity_limit_sim=5.0,
        stiffness=0.0,
        damping=0.0,
    )

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene (arms.py 스타일)."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 한 개 Origin 그룹
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
                linear_damping=0.05,   # 약간의 감쇠(베이스 흔들림 완화)
                angular_damping=0.05,
            ),
            # 주의: 네 환경에선 fixed_base 인자 미지원 → 전달하지 않음
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        actuators={"all": ACTUATOR},
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
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # 초기 자세: 기본값 + 작은 잡음(포지션 모드 기준 데모와 동일)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * args_cli.noise
            # 안전하게 소프트 제한 안으로 클램프
            lo, hi = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
            joint_pos = joint_pos.clamp_(lo, hi)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print(f"[INFO] Reset UR10e | control={args_cli.control}")

        if args_cli.control == "position":
            # 데모 스타일: 매 스텝 랜덤 근처 타깃 (작게 움직임)
            target = robot.data.default_joint_pos + (torch.randn_like(robot.data.joint_pos) * (args_cli.noise * 0.5))
            target = target.clamp_(robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1])
            robot.set_joint_position_target(target)
        else:
            # 토크 모드: 완만 제한(tanh)한 무작위 토크
            rand = torch.randn_like(robot.data.joint_pos)
            efforts = torch.tanh(rand) * args_cli.effort_scale
            robot.set_joint_effort_target(efforts)

        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
