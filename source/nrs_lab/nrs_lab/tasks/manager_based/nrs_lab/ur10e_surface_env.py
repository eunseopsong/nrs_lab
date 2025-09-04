# SPDX-License-Identifier: BSD-3-Clause
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .ur10e_surface_env_cfg import UR10eSurfaceEnvCfg

class UR10eSurfaceEnv(ManagerBasedRLEnv):
    cfg: UR10eSurfaceEnvCfg

    def __init__(self, cfg: UR10eSurfaceEnvCfg = UR10eSurfaceEnvCfg(), **kwargs):
        super().__init__(cfg, **kwargs)

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        # 홈 포즈(0)로 간단 초기화
        if hasattr(self, "robot"):
            q = torch.zeros_like(self.robot.data.joint_pos)
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        return obs
