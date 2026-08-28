from dataclasses import dataclass, field
from typing import Tuple
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

@RobotConfig.register_subclass("uf::robot")
@dataclass
class UFRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {}
    )
    cameras_args: dict = None
    robot_ip: str = "192.168.1.127"
    robot_dof: int | None = None  # Set it correctly if controlling in joint space!
    control_space: str = "joint"
    gripper_type: int = 1       # 1: xArm Gripper, 2: xArm Gripper G2, 10: Pika Gripper, 11: Robotiq 2F-85
    gripper_port: str = None    # only used by pika gripper (gripper_type=10)
    gripper_speed: int = -1     # auto
    gripper_force: int = -1     # auto
    observe_joint_vel: bool = False # only effective in joint control mode
    # cartesian(endpoint) 제어 모드에서도 RT report의 실제 관절값 J1~J7을 관측에
    # 포함한다. pi0 학습 데이터는 관절 스키마를 유지해야 하므로 endpoint 녹화에서
    # 반드시 true로 쓴다. 관측 순서는 [J1..J7, pose 6, gripper]다.
    cartesian_obs_include_joints: bool = False
    start_joints: Tuple[float, ...] = (0, 0, 0, 90, 0, 90, 0) # °
    start_tcp_pose: Tuple[float, ...] = None # [x, y, z, roll(°), pitch(°), yaw(°)]
    # 텔레옵 시작 직후의 안전 확인 구간입니다.
    # 이 시간 동안에는 initial_sync_joint_velocity로만 관절 목표를 보냅니다.
    # 목표 관절각을 자르거나 제한하지는 않습니다.
    initial_sync_duration_s: float = 3.0
    initial_sync_joint_velocity: float = 3.0 # °/s
    max_joint_velocity: int = 90   # °/s, only effective in joint control mode
    max_linear_velocity: int = 200 # mm/s, only effective in cartesian control mode
    # cartesian 모드 텔레옵 시작 직후 initial_sync_duration_s 동안 쓰는 저속
    # linear 속도다. joint 모드의 initial_sync_joint_velocity에 대응한다.
    initial_sync_linear_velocity: float = 20.0 # mm/s
    no_action: bool = False # only for debug

    def __post_init__(self):
        super().__post_init__()
        self.id = 'uf_robot' if self.id is None else self.id
