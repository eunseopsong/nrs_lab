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

from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.envs.mdp import rewards as mdp_rew
from isaaclab.envs.mdp import terminations as mdp_terms
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg


# ----------------- Observation group -----------------
@configclass
class UR10eObsCfg(ObsGroup):
    """Joint state observations for policy."""
    joint_pos: ObsTerm = ObsTerm(
        func=mdp_obs.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},   # ✅ dict 로 전달
    )
    joint_vel: ObsTerm = ObsTerm(
        func=mdp_obs.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},   # ✅ dict 로 전달
    )

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
    """Scene containing UR10e and concave surface (spawn from a single USD)."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 로봇 + 표면이 함께 들어있는 USD에서 스폰
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/ur10e_w_spindle_robot",                 # 스테이지에서 Copy Path
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd",
            # 필요시 prim_path="/*" 생략 가능 (루트에 World가 포함된 파일)
        ),
        actuators={},   # 기본 PD
    )
    # 표면은 위 USD에 함께 포함되어 로드됨 (따로 정의 불필요)


# ----------------- Env Config -----------------
@configclass
class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: UR10eSceneCfg = UR10eSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20
        self.sim.device = "cuda:0"
        self.sim.create_stage_in_memory = True       # 스폰 방식이면 in-memory 권장
        self.sim.render_interval = self.decimation   # 경고 제거
