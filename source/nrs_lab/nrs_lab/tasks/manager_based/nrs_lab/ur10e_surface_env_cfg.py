# SPDX-License-Identifier: BSD-3-Clause
# UR10e Surface Environment Config (load stage USD; bind robot prim; viewer+light set)

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

# MDP building blocks
from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.envs.mdp import rewards as mdp_rew
from isaaclab.envs.mdp import terminations as mdp_terms
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg


# ----------------- Observations -----------------
@configclass
class UR10eObsCfg(ObsGroup):
    """Expose joint position & velocity of the robot."""
    joint_pos: ObsTerm = ObsTerm(
        func=mdp_obs.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},  # dict로 전달 필수
    )
    joint_vel: ObsTerm = ObsTerm(
        func=mdp_obs.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: UR10eObsCfg = UR10eObsCfg()


# ----------------- Actions -----------------
@configclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],          # 모든 조인트
        use_default_offset=True,
        scale=1.0,
    )


# ----------------- Rewards / Terminations -----------------
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
    """
    Load the full USD (robot + concave surface) as the stage,
    then bind the existing robot prim as an articulation.
    """

    # 1) Load the USD stage from disk (most robust for materials/meshes)
    stage_usd = AssetBaseCfg(
        prim_path="/World",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd",
        ),
    )

    # 2) Bind the robot that already exists in the loaded stage
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/ur10e_w_spindle_robot",  # Stage에서 Copy Path한 이름
        spawn=None,                                # 이미 stage에 있으므로 스폰 X
        actuators={},                              # 기본 PD
    )

    # 가시성 보장을 위한 돔라이트 (USD에 조명이 없다면 유용)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=5000.0, color=(1.0, 1.0, 1.0)),
    )

    # NOTE:
    # - ground가 USD에 이미 있으면 여기서 따로 만들지 않습니다(중복 방지).
    # - num_envs=1이므로 복제 이슈 없음.


# ----------------- Environment -----------------
@configclass
class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL Env for UR10e on concave surface."""

    # Scene & managers
    scene: UR10eSceneCfg = UR10eSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # timing
        self.decimation = 2
        self.episode_length_s = 20

        # viewer: look at the scene in world coords
        self.viewer.cam_prim_path = "/OmniverseKit_Persp"
        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.6)

        # simulation
        self.sim.device = "cuda:0"
        self.sim.create_stage_in_memory = False   # 디스크 스테이지 사용(머티리얼/메시 안정)
        self.sim.render_interval = self.decimation
