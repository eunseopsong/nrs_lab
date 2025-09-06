# SPDX-License-Identifier: BSD-3-Clause
# UR10e Surface Environment Config (prebuilt USD stage loader + robot binding)

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


# ---------- Observations ----------
@configclass
class UR10eObsCfg(ObsGroup):
    joint_pos: ObsTerm = ObsTerm(func=mdp_obs.joint_pos, params=SceneEntityCfg("robot"))
    joint_vel: ObsTerm = ObsTerm(func=mdp_obs.joint_vel, params=SceneEntityCfg("robot"))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: UR10eObsCfg = UR10eObsCfg()


# ---------- Actions ----------
@configclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        scale=1.0,
    )


# ---------- Rewards / Terminations ----------
@configclass
class RewardsCfg:
    alive: RewTerm = RewTerm(func=mdp_rew.is_alive, weight=1.0)
    terminating: RewTerm = RewTerm(func=mdp_rew.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    time_out: DoneTerm = DoneTerm(func=mdp_terms.time_out, time_out=True)


# ---------- Scene ----------
@configclass
class UR10eSceneCfg(InteractiveSceneCfg):
    """
    Scene: load the prebuilt stage USD, then bind 'robot' entity to the existing prim.
    """

    # 1) ✅ Load the prebuilt USD stage (this makes /World/... prims exist)
    stage_usd = AssetBaseCfg(
        prim_path="/",  # load at root
        spawn=sim_utils.UsdFileCfg(
            # usd_path="/home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd"
            usd_path="/home/eunseop/isaac/isaac_save/ur10e_only.usd"
        ),
    )

    # 2) ✅ Bind robot to existing prim in that stage
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/ur10e_w_spindle_robot",  # 네가 스샷에서 보여준 정확한 경로
        spawn=None,       # do not spawn new; use the existing prim
        actuators={},     # use default PD drives
    )

    # ⚠️ Ground/PhysicsScene는 USD에 이미 있으니 추가 생성하지 않음


# ---------- Env Config ----------
@configclass
class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: UR10eSceneCfg = UR10eSceneCfg(num_envs=1, env_spacing=2.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # timings
        self.decimation = 2
        self.episode_length_s = 20

        # sim settings
        self.sim.device = "cuda:0"
        # 외부 USD를 우리가 직접 로드하므로(in scene.stage_usd), in-memory stage 사용해도 문제 없음
        # 필요시 다음 라인을 주석 해제/변경 가능:
        # self.sim.create_stage_in_memory = True
