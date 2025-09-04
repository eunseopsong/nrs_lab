import gymnasium as gym
from . import agents

# 기존 템플릿 태스크
gym.register(
    id="Template-Nrs-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nrs_lab_env_cfg:NrsLabEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# 새 UR10e 태스크 등록
gym.register(
    id="UR10e-Surface-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_surface_env_cfg:UR10eSurfaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
