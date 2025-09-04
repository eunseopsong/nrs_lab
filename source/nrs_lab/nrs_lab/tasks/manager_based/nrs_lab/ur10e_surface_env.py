import torch
from isaaclab.envs.manager_based.manager_env import ManagerBasedRLEnv
from .ur10e_surface_env_cfg import UR10eSurfaceEnvCfg

class UR10eSurfaceEnv(ManagerBasedRLEnv):
    cfg: UR10eSurfaceEnvCfg

    def __init__(self, cfg: UR10eSurfaceEnvCfg = UR10eSurfaceEnvCfg(), **kwargs):
        super().__init__(cfg, **kwargs)

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        # 간단히 UR10e 홈포즈로 리셋
        if hasattr(self, "robot"):
            q_home = torch.zeros_like(self.robot.data.joint_pos)
            self.robot.write_joint_state_to_sim(q_home, torch.zeros_like(q_home))
        return obs

    def pre_physics_step(self, actions: torch.Tensor):
        # 간단히 랜덤 노이즈 기반 제어
        if hasattr(self, "robot"):
            self.robot.set_joint_position_target(actions)
        super().pre_physics_step(actions)
