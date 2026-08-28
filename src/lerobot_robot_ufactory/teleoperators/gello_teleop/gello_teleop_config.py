#!/usr/bin/env python

from dataclasses import dataclass
from typing import Tuple
from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("uf::gello_teleop")
@dataclass
class GelloTeleopConfig(TeleoperatorConfig):
    # Port to connect to the gello dummy arm
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJZYC7-if00-port0"

    # Others: Calibration angles, joint directions etc
    joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    joint_signs: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1) # if follow the original open-sourced gello xarm7 setup
    start_joints: Tuple[float, ...] = (0, 0, 0, 90, 0, 90, 0)  # °
    gripper_id: int = 8  # -1: no gripper
    torque_joint_ids: Tuple[int, ...] = None  # the joints will activate torque mode.
    # 에피소드 시작마다 리더의 현재 자세를 팔로워의 현재 자세에 상대 정렬합니다.
    # 이렇게 하면 GELLO를 기계적으로 정확한 초기 관절각에 맞추지 않아도 첫 명령에서
    # 팔로워가 갑자기 움직이지 않습니다.
    relative_alignment_enabled: bool = True
    leader_stability_duration_s: float = 0.5
    leader_stability_max_delta_deg: float = 1.0
    first_action_max_delta_deg: float = 1.0

    # "joint": GELLO 관절값을 그대로 xArm 관절 목표로 보낸다(기존 방식).
    # "endpoint": GELLO 관절값을 xArm7 FK에 넣어 endpoint(TCP)만 추적하고,
    #   실제 관절 궤적은 xArm 컨트롤러의 온라인 planning(mode 7)이 결정한다.
    #   endpoint 모드는 robot.control_space가 "cartesian"이어야 한다.
    tracking_mode: str = "joint"
    # endpoint 모드에서 리더 FK에 적용할 TCP offset [x, y, z(mm), roll, pitch, yaw(°)].
    # 반드시 xArm 컨트롤러에 설정된 값과 같아야 방향 변화 시 오차가 없다.
    # 기본값은 표준 xArm Gripper의 z=172mm다.
    tcp_offset: Tuple[float, ...] = (0.0, 0.0, 172.0, 0.0, 0.0, 0.0)
    # endpoint 모드 전용 첫 action 안전 한계. 상대 정렬 직후 첫 목표는 팔로워
    # 현재 TCP와 수학적으로 같아야 하므로 아주 작은 값만 허용한다.
    first_action_max_delta_mm: float = 5.0
    first_action_max_delta_rot_deg: float = 2.0
    # endpoint 모드에서 한 프레임 사이 목표 이동이 이 값을 넘으면 엔코더 이상으로
    # 판단하고 텔레옵을 중단한다(목표를 잘라내지 않고 멈춘다는 프로젝트 원칙 유지).
    max_frame_jump_mm: float = 150.0
    # endpoint 모드에서 FK를 계산할 xArm 컨트롤러 주소. 지정하면 컨트롤러의
    # get_forward_kinematics(공장 캘리브레이션 포함, LAN 왕복 ~0.2ms)를 사용해
    # 로컬 공칭 FK의 수 mm 모델 오차를 제거한다. 빈 문자열이면 로컬 FK만 쓴다.
    robot_ip: str = ""
    # endpoint 목표 pose의 EMA 스무딩 계수(0 < α ≤ 1, 1 = 스무딩 없음).
    # GELLO 엔코더 갱신(57600 baud에서 약 18Hz)이 30Hz 루프보다 느려 생기는
    # 계단형 목표를 보간한다. α가 작을수록 부드럽지만 지연이 커진다.
    endpoint_smoothing_alpha: float = 1.0

    def __post_init__(self):
        self.id = 'gello_teleop' if self.id is None else self.id
        if self.leader_stability_duration_s <= 0:
            raise ValueError("leader_stability_duration_s는 0보다 커야 합니다.")
        if not 0 < self.leader_stability_max_delta_deg <= 10:
            raise ValueError("leader_stability_max_delta_deg는 0보다 크고 10° 이하여야 합니다.")
        if not 0 < self.first_action_max_delta_deg <= 10:
            raise ValueError("first_action_max_delta_deg는 0보다 크고 10° 이하여야 합니다.")
        if self.tracking_mode not in ("joint", "endpoint"):
            raise ValueError(f"tracking_mode는 joint 또는 endpoint여야 합니다: {self.tracking_mode!r}")
        if len(tuple(self.tcp_offset)) != 6:
            raise ValueError("tcp_offset은 [x, y, z(mm), roll, pitch, yaw(°)] 6개 값이어야 합니다.")
        if not 0 < self.first_action_max_delta_mm <= 50:
            raise ValueError("first_action_max_delta_mm는 0보다 크고 50mm 이하여야 합니다.")
        if not 0 < self.first_action_max_delta_rot_deg <= 10:
            raise ValueError("first_action_max_delta_rot_deg는 0보다 크고 10° 이하여야 합니다.")
        if not 10 <= self.max_frame_jump_mm <= 500:
            raise ValueError("max_frame_jump_mm는 10~500mm 범위여야 합니다.")
        if not 0 < self.endpoint_smoothing_alpha <= 1.0:
            raise ValueError("endpoint_smoothing_alpha는 0보다 크고 1 이하여야 합니다.")
