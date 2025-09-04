from isaaclab.envs.manager_based.manager_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils


class UR10eSurfaceEnvCfg(ManagerBasedRLEnvCfg):
    def __init__(self, usd_path: str = "/home/eunseop/isaac/isaac_save/ur10e_concave_surface.usd"):
        super().__init__()

        # 시뮬레이션 설정
        self.sim = SimulationCfg(device="cuda:0")

        # UR10e 로봇 설정
        self.robot = ArticulationCfg(
            prim_path="/World/UR10e",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        )

        # ground plane 추가
        self.ground = sim_utils.GroundPlaneCfg()
