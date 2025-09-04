# Copyright (c) 2025
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
UR10e from a local USD, random interaction with periodic reset.
- control=position : joint position targets(±noise) + PD
- control=effort   : random torque (tanh limited)
- fixed_base       : keep robot base fixed to the world (recommended)
- reset_every      : periodically reset to a UR10e "home" pose

UR10e home pose used here (rad):
  shoulder_pan=0, shoulder_lift=-1.5708, elbow=-1.5708,
  wrist_1=-1.5708, wrist_2=1.5708, wrist_3=0

.. code-block:: bash

    ./isaaclab.sh -p nrs_lab/run_ur10e_random_reset.py \
    --usd_path /home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd \
    --control effort --effort_scale 2.0 \
    --fixed_base --reset_every 400 --num_envs 6 --env_spacing 2.0 --headless

    ./isaaclab.sh -p nrs_lab/run_ur10e_random_reset.py \
    --usd_path /home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd \
    --control position \
    --fixed_base --reset_every 400 --num_envs 6 --env_spacing 2.0 --headless

"""

"""Launch Isaac Sim Simulator first."""
import argparse
import math
import torch

from isaaclab.app import AppLauncher

# ---- CLI ----
parser = argparse.ArgumentParser(description="UR10e random interaction with periodic reset (home pose).")
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the UR10e .usd file.")
parser.add_argument("--control", type=str, default="position", choices=["position", "effort"],
                    help="Interaction mode (default: position).")
parser.add_argument("--noise", type=float, default=0.10, help="Position noise amplitude in rad (position mode).")
parser.add_argument("--effort_scale", type=float, default=3.0, help="Torque scale (effort mode).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Spacing (meters) between environment origins.")
parser.add_argument("--fixed_base", action="store_true",
                    help="Fix the robot's root link to the world (recommended for UR10 on a table).")
parser.add_argument("--reset_every", type=int, default=500,
                    help="Reset period in sim steps (return to home pose periodically).")
# App launcher 공통 인자 (예: --headless, --device)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
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

# ---- 액추에이터 설정 ----
if args_cli.control == "position":
    ACTUATOR = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=200.0,
        velocity_limit_sim=5.0,
        stiffness=3000.0,
        damping=60.0,
    )
else:
    ACTUATOR = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=200.0,
        velocity_limit_sim=5.0,
        stiffness=0.0,
        damping=0.0,
    )

# ---- UR10e 홈 포즈 (rad) ----
UR10E_HOME_DICT = {
    # 표준 UR10e 조인트 네이밍 (USD의 조인트 이름이 다르면 아래 키만 바꿔주면 됨)
    "shoulder_pan_joint": 3.14159265359,
    "shoulder_lift_joint": -1.57079632679,
    "elbow_joint": -1.57079632679,
    "wrist_1_joint": -1.57079632679,
    "wrist_2_joint": 1.57079632679,
    "wrist_3_joint": 0.0,
}

def _make_env_origins(num_envs: int, spacing: float) -> list[list[float]]:
    """num_envs 개의 env origin을 격자(grid)로 생성."""
    if num_envs <= 1:
        return [[0.0, 0.0, 0.0]]
    cols = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / cols)
    origins = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= num_envs:
                break
            origins.append([c * spacing, r * spacing, 0.0])
    # 중앙 정렬
    if len(origins) > 1:
        xs = [o[0] for o in origins]
        ys = [o[1] for o in origins]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        for o in origins:
            o[0] -= cx
            o[1] -= cy
    return origins

def design_scene(num_envs: int, env_spacing: float) -> tuple[dict, list[list[float]]]:
    """Designs the scene with multiple envs."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 여러 Origin 생성
    origins = _make_env_origins(num_envs, env_spacing)
    for i, org in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform", translation=org)

    # UR10e articulation: init_state에 홈 포즈를 명시
    ur10e_cfg = ArticulationCfg(
        prim_path="/World/Origin_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.05,
                angular_damping=0.05,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=args_cli.fixed_base,
            ),
        ),
        actuators={"all": ACTUATOR},
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),                   # 필요시 (x,y,z) 올려도 됨
            rot=(1.0, 0.0, 0.0, 0.0),              # 단위 쿼터니언
            joint_pos=UR10E_HOME_DICT,             # ★ 홈 포즈를 이름으로 지정
        ),
    )
    ur10e = Articulation(cfg=ur10e_cfg)

    scene_entities = {"ur10e": ur10e}
    return scene_entities, origins

def _build_home_batch(robot: Articulation, device, num_envs: int) -> torch.Tensor:
    """
    로봇의 joint_names에 맞춰 UR10E_HOME_DICT를 벡터(q_home)로 변환하고,
    (num_envs, dof) 배치 텐서로 만들어 반환.
    joint 이름이 다르면 기본값(default_joint_pos)을 유지하고, 일치하는 이름만 덮어씀.
    """
    # 기본값에서 시작
    q_home = robot.data.default_joint_pos.clone().to(device)

    # joint_names가 노출되면 이름 기반 매핑 적용
    names = getattr(robot.data, "joint_names", None)
    if names is not None:
        name_to_idx = {n: i for i, n in enumerate(names)}
        for jn, val in UR10E_HOME_DICT.items():
            if jn in name_to_idx:
                q_home[:, name_to_idx[jn]] = float(val)

    return q_home

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    robot = entities["ur10e"]
    sim_dt = sim.get_physics_dt()
    count = 0

    # 초기 루트/조인트 상태(기본값)와 홈 포즈 벡터 준비
    init_root = robot.data.default_root_state.clone()
    q_home = _build_home_batch(robot, sim.device, origins.shape[0])
    dq_zero = torch.zeros_like(q_home)

    while simulation_app.is_running():
        if count % args_cli.reset_every == 0:
            count = 0
            # 루트 포즈: env origins 반영
            root_state = init_root.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # 홈 포즈로 리셋 (position 모드면 약간의 랜덤 노이즈 허용)
            q = q_home.clone()
            if args_cli.control == "position" and args_cli.noise > 0.0:
                q += torch.rand_like(q) * args_cli.noise
                lo = robot.data.soft_joint_pos_limits[..., 0]
                hi = robot.data.soft_joint_pos_limits[..., 1]
                q.clamp_(lo, hi)

            robot.write_joint_state_to_sim(q, dq_zero)
            robot.reset()
            print(f"[INFO] Reset → UR10e home pose | fixed_base={args_cli.fixed_base}")

        if args_cli.control == "position":
            # 홈 포즈 근처를 작은 랜덤으로 왔다갔다
            target = q_home + (torch.randn_like(robot.data.joint_pos) * (args_cli.noise * 0.5))
            target = target.clamp_(robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1])
            robot.set_joint_position_target(target)
        else:
            rand = torch.randn_like(robot.data.joint_pos)
            efforts = torch.tanh(rand) * args_cli.effort_scale
            robot.set_joint_effort_target(efforts)

        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([4.5, 0.0, 6.0], [0.0, 0.0, 1.5])

    scene_entities, scene_origins = design_scene(args_cli.num_envs, args_cli.env_spacing)
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()
