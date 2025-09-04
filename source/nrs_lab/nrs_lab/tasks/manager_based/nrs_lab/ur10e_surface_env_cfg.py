# SPDX-License-Identifier: BSD-3-Clause
# Manager-based RL Env config for UR10e on concave surface (prebuilt USD stage).

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
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

# mdp 모듈 (관절 관측/보상/종료 함수)
from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.envs.mdp import rewards as mdp_rew
from isaaclab.envs.mdp import terminations as mdp_terms
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg


# ----------------- Observation group -----------------
@configclass
class UR10eObsCfg(ObsGroup):
    """Joint position/velocity observations for policy group."""

    joint_pos: ObsTerm = ObsTerm(func=mdp_obs.joint_pos, params=SceneEntityCfg(name="robot"))
    joint_vel: ObsTerm = ObsTerm(func=mdp_obs.joint_vel, params=SceneEntityCfg(name="robot"))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: UR10eObsCfg = UR10eObsCfg()


@configclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True, scale=1.0
    )


@configclass
class RewardsCfg:
    alive: RewTerm = RewTerm(func=mdp_rew.is_alive, weight=1.0)
    terminating: RewTerm = RewTerm(func=mdp_rew.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    time_out: DoneTerm = DoneTerm(func=mdp_terms.time_out, time_out=True)


# ----------------- Env Config -----------------
@configclass
class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL Env for UR10e with concave surface (using existing USD stage)."""

    # 기본 필드들
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization hook."""
        # --- general settings ---
        self.decimation = 2
        self.episode_length_s = 20

        # --- simulation settings ---
        self.sim.device = "cuda:0"
        self.sim.create_stage_in_memory = False  # 프리빌트 스테이지 그대로 사용

        # --- robot binding ---
        # Stage 트리에서 Copy Path로 정확한 prim_path 확인
        self.robot = ArticulationCfg(
            prim_path="/World/ur10e_w_spindle_robot",
            spawn=None,   # 새로 스폰하지 않고 기존 프림 사용
            actuators={}, # 기본 PD 드라이브
        )
