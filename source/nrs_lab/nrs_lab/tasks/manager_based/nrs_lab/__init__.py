@configclass
class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20
        # simulation settings
        self.sim.device = "cuda:0"
        self.sim.create_stage_in_memory = False

        # robot 바인딩 (spawn=None)
        self.robot = ArticulationCfg(
            prim_path="/World/ur10e_w_spindle_robot",  # Stage에서 Copy Path
            spawn=None,
            actuators={},  # 기본 PD
        )
