# SPDX-License-Identifier: BSD-3-Clause
# Minimal, runnable Manager-based config for UR10e on a concave surface USD.

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

# managers / mdp
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.envs.mdp import rewards as mdp_rew
from isaaclab.envs.mdp import terminations as mdp_terms

# scene / actuators
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass


# ---- Observation group (joint pos/vel) ----
class UR10eObsCfg(ObservationGroupCfg):
    joint_pos: ObservationTermCfg = ObservationTermCfg(
        func=mdp_obs.joint_pos, params=SceneEntityCfg(name="robot")
    )
    joint_vel: ObservationTermCfg = ObservationTermCfg(
        func=mdp_obs.joint_vel, params=SceneEntityCfg(name="robot")
    )


@configclass
class ObservationsCfg:
    policy: UR10eObsCfg = UR10eObsCfg()


@configclass
class ActionsCfg:
    # PD position targets to all joints (regex ".*")
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True, scale=1.0
    )


@configclass
class RewardsCfg:
    # 최소 보상: 살아있기 보상과 종료 패널티
    alive: RewTerm = RewTerm(func=mdp_rew.is_alive, weight=1.0)
    terminating: RewTerm = RewTerm(func=mdp_rew.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    # 에피소드 타임아웃
    time_out: DoneTerm = DoneTerm(func=mdp_terms.time_out, time_out=True)


class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL Env config for UR10e on a concave surface."""

    # --- 필수: scene/주기/에피소드 길이 ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    decimation: int = 2                 # 60 Hz 물리 * 2 = 30 Hz 제어주기
    episode_length_s: float = 20.0      # 20초 에피소드(원하면 조절)

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(device="cuda:0")

    # --- Scene entities: robot + ground ---
    robot: ArticulationCfg = ArticulationCfg(
    prim_path="/World/ur10e_w_spindle_robot",  # ← Stage에서 실제 경로로 교체
    spawn=None,                                 # 새 스폰 X, 기존 프림 사용
    actuators={},                               # 기본 PD 사용
    )
    ground: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg()

    # --- Managers ---
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
