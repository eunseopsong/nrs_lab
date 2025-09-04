# SPDX-License-Identifier: BSD-3-Clause
# Minimal, runnable Manager-based config for UR10e on a concave surface USD.

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

# Managers (groups/terms)
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    TerminationTermCfg,
    SceneEntityCfg,
)


# MDP terms: actions / observations
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp import observations as mdp_obs

# ---- Observation group (joint pos/vel) ----
class UR10eObsCfg(ObservationGroupCfg):
    # asset_name must match the attribute name used below ("robot")
    joint_pos: ObservationTermCfg = ObservationTermCfg(
        func=mdp_obs.joint_pos,
        params=SceneEntityCfg(name="robot"),
    )
    joint_vel: ObservationTermCfg = ObservationTermCfg(
        func=mdp_obs.joint_vel,
        params=SceneEntityCfg(name="robot"),
    )

class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    def __init__(self, usd_path: str = "/home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd"):
        super().__init__()

        # --- Simulation ---
        self.sim = SimulationCfg(device="cuda:0")  # 필요 시 "cpu" 가능

        # --- Scene: robot + ground ---
        self.robot = ArticulationCfg(
            prim_path="/World/UR10e",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
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
                    fix_root_link=True,  # 탁상 고정 권장
                ),
            ),
        )
        self.ground = sim_utils.GroundPlaneCfg()

        # --- Actions: joint position targets (PD inside simulator) ---
        self.actions = {
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"],      # 모든 조인트(정규식)
                use_default_offset=True, # USD 초기자세를 오프로 사용
                scale=1.0,
            )
        }

        # --- Observations: expose joint state to the policy group ---
        self.observations = {
            "policy": UR10eObsCfg(),
        }

        # --- Terminations: minimal timeout ---
        self.terminations = {
            "time_out": TerminationTermCfg(time_out=True),
        }

        # Rewards / Commands / Curriculum / Randomization : intentionally omitted for minimal runnable setup
