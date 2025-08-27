# SPDX-License-Identifier: BSD-3-Clause

"""
UR10e from a local USD, random interaction (arms.py style) with base anchoring:
- Spawn UR10e
- Auto-detect base link and weld it to /World/Origin1 via a FixedJoint
- High-friction ground via GroundPlaneCfg
- Position-control (PD) with small random targets (demo-like)

./isaaclab.sh -p scripts/custom/run_ur10e_random_fixed.py \
    --usd_path /home/USER/isaac/isaac_save/ur10e_convex_surface.usd \
    --noise 0.10 --stiffness 3000 --damping 60

"""

# Launch Isaac Sim first -------------------------------------------------------
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR10e random (position) with base fixed joint (tutorial-style).")
parser.add_argument("--usd_path", type=str, required=True, help="Absolute path to the UR10e .usd file.")
parser.add_argument("--noise", type=float, default=0.10, help="Position noise amplitude in rad.")
parser.add_argument("--stiffness", type=float, default=3000.0, help="PD stiffness for joints.")
parser.add_argument("--damping", type=float, default=60.0, help="PD damping for joints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest follows ----------------------------------------------------------------
import torch
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext

# Ensure physics scene exists (avoid 'scene not found')
import omni.usd
from pxr import UsdPhysics
_stage = omni.usd.get_context().get_stage()
if not _stage.GetPrimAtPath("/World/physicsScene"):
    UsdPhysics.Scene.Define(_stage, "/World/physicsScene")


def _weld_base_to_origin(robot: Articulation, origin_prim: str):
    """Create a FixedJoint between robot base link and origin xform."""
    base_prim = None
    # Prefer articulation-reported first body name (usually base)
    if len(robot.data.body_names) > 0:
        base_prim = robot.data.body_names[0]
    # Fallback search
    if base_prim is None:
        for name in ("base_link", "ur_base_link", "base"):
            cand = f"{robot.prim_path}/{name}"
            if _stage.GetPrimAtPath(cand):
                base_prim = cand
                break
    if base_prim is None:
        print("[WARN] Could not auto-detect base link. Skipping weld joint.")
        return

    # Create FixedJoint under the robot prim
    fj = sim_utils.FixedJointCfg()
    fj.func(
        prim_path=f"{robot.prim_path}/BaseWeld",
        cfg=fj,
        body0=origin_prim,   # parent
        body1=base_prim,     # child
    )
    print(f"[INFO] Welded base to origin via FixedJoint: {base_prim} -> {origin_prim}")


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene (arms.py style)."""
    # Ground-plane with higher friction (use GroundPlaneCfg fields that exist in your build)
    gcfg = sim_utils.GroundPlaneCfg(
        static_friction=1.5,   # if your build doesn't accept these, try removing them
        dynamic_friction=1.2,
        restitution=0.0,
    )
    gcfg.func("/World/defaultGroundPlane", gcfg)

    # Lights
    lcfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    lcfg.func("/World/Light", lcfg)

    # One origin group
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # UR10e articulation (local USD)
    ur10e_cfg = ArticulationCfg(
        prim_path="/World/Origin.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.05,
                angular_damping=0.05,
            ),
            # no fixed_base field in this build
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        actuators={
            "pd_all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=5.0,
                stiffness=args_cli.stiffness,
                damping=args_cli.damping,
            )
        },
    )
    ur10e = Articulation(cfg=ur10e_cfg)

    # Weld base to origin (after prim exists)
    _weld_base_to_origin(ur10e, "/World/Origin1")

    scene_entities = {"ur10e": ur10e}
    return scene_entities, origins


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop (arms.py style)."""
    robot = entities["ur10e"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            # 약간 위로 띄워 초기 관통/튐 방지 (원하면 주석 해제)
            # root_state[:, 2] += 0.01
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # default + 작은 잡음
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * args_cli.noise
            lo, hi = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
            joint_pos = joint_pos.clamp_(lo, hi)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("[INFO]: Resetting UR10e state... (base welded)")

        # Small random position target around default (demo-like)
        target = robot.data.default_joint_pos + (torch.randn_like(robot.data.joint_pos) * (args_cli.noise * 0.5))
        target = target.clamp_(robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1])
        robot.set_joint_position_target(target)

        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
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
