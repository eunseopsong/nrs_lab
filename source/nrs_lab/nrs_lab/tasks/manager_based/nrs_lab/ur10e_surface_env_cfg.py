# SPDX-License-Identifier: BSD-3-Clause
# UR10e Surface Environment Config

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# MDP 함수들
from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.envs.mdp import rewards as mdp_rew
from isaaclab.envs.mdp import terminations as mdp_terms
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg


# ----------------- Observation group -----------------
@configclass
class UR10eObsCfg(ObsGroup):
    """Joint state observations for policy."""

    joint_pos: ObsTerm = ObsTerm(func=mdp_obs.joint_pos, params=SceneEntityCfg("robot"))
    joint_vel: ObsTerm = ObsTerm(func=mdp_obs.joint_vel, params=SceneEntityCfg("robot"))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: UR10eObsCfg = UR10eObsCfg()


@configclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        scale=1.0,
    )


@configclass
class RewardsCfg:
    alive: RewTerm = RewTerm(func=mdp_rew.is_alive, weight=1.0)
    terminating: RewTerm = RewTerm(func=mdp_rew.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    time_out: DoneTerm = DoneTerm(func=mdp_terms.time_out, time_out=True)


# ----------------- Scene -----------------
@configclass
class UR10eSceneCfg(InteractiveSceneCfg):
    """Scene containing UR10e and ground plane."""

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # UR10e robot → 반드시 "robot" 키로 등록!
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/ur10e_w_spindle_robot",  # ← Stage에서 Copy Path 해서 정확히 수정
        spawn=None,       # 프리빌트 프림 사용
        actuators={},     # 기본 PD 드라이브
    )


# ----------------- Env Config -----------------
@configclass
class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL Env for UR10e on concave surface."""

    # Scene settings
    scene: UR10eSceneCfg = UR10eSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # general
        self.decimation = 2
        self.episode_length_s = 20

        # simulation
        self.sim.device = "cuda:0"
        self.sim.create_stage_in_memory = False
