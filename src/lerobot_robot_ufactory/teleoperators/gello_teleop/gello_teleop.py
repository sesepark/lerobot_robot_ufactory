#!/usr/bin/env python
import logging
import time
import math
import numpy as np
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations
from ..base_teleop import UFBaseTeleop
from .gello_teleop_config import GelloTeleopConfig
from .xarm7_kinematics import fk_tcp_matrix_mm, tcp_offset_matrix_mm

CARTESIAN_ACTION_KEYS = ("pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz")


logger = logging.getLogger(__name__)

class GelloTeleop(UFBaseTeleop):
    """
    GELLO for xArm tele-op, ref: https://wuphilipp.github.io/gello_site/
    """

    config_class = GelloTeleopConfig
    name = "Gello Teleop For xArm"

    def __init__(self, config: GelloTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True # CHECK!!
        self._teleop_enabled = False
        self._follower_reference = None
        self._last_leader_arm = None
        self._leader_accumulated_delta = None
        self._first_action_pending = False

        # endpoint 추적 모드: GELLO 관절 delta를 팔로워 기준 관절에 더한 "가상
        # 관절"을 FK에 넣어 TCP delta만 추적한다. GELLO 절대값을 직접 FK에 넣으면
        # 시작 시 수 °의 엔코더 offset 오차가 FK 비선형성 때문에 null-space
        # 움직임에서 endpoint 오차로 새므로, 관절 모드와 같은 delta 상쇄를 쓴다.
        self._tracking_endpoint = self.config.tracking_mode == "endpoint"
        tcp_offset = list(self.config.tcp_offset)
        self._tcp_offset_matrix = tcp_offset_matrix_mm(
            tcp_offset[:3] + [math.radians(v) for v in tcp_offset[3:6]]
        )
        self._fk_arm = None              # 컨트롤러 FK용 읽기 전용 XArmAPI 연결
        self._fk_arm_healthy = False
        self._follower_ref_joints = None # 팔로워 기준 관절(rad, 가상 관절의 기준점)
        self._follower_ref_pos = None    # 팔로워 기준 TCP 위치 (mm)
        self._follower_ref_rot = None    # 팔로워 기준 TCP 회전행렬 (3x3)
        self._leader_ref_pos = None      # 기준 관절의 FK TCP 위치 (mm)
        self._leader_ref_rot = None
        self._last_target_pos = None     # 프레임 간 급점프 감시용 (스무딩 전 원목표)
        self._last_leader_tcp = None     # 컨트롤러 FK 일시 실패 시 재사용
        self._smooth_pos = None          # EMA 스무딩 상태
        self._smooth_rot = None

        from gello.dynamixel.driver import DynamixelDriver
        from gello.agents.gello_agent import DynamixelRobotConfig

        # auto get joint offset from gello
        joint_ids = []
        joint_ids.extend(self.config.joint_ids)
        if self.config.gripper_id >= 0:
            joint_ids.append(self.config.gripper_id)
        driver = DynamixelDriver(joint_ids, port=self.config.port, baudrate=57600)
        for _ in range(10):
            driver.get_joints()  # warmup
        curr_joints = driver.get_joints()
        driver.close()
        joint_offsets = []
        start_joints = list(map(math.radians, self.config.start_joints))
        for i in range(len(start_joints)):
            offset = curr_joints[i] - start_joints[i] / self.config.joint_signs[i]
            joint_offsets.append(offset)
        if self.config.gripper_id >= 0:
            gripper_config = [self.config.gripper_id, np.rad2deg(curr_joints[-1]) - 0.2, np.rad2deg(curr_joints[-1]) - 42]
        else:
            gripper_config = None

        param_dict = {
                "joint_ids": self.config.joint_ids,
                "joint_signs": self.config.joint_signs,
                "joint_offsets": joint_offsets,
                "gripper_config": gripper_config
        }
        self._dynamixel_robo_config = DynamixelRobotConfig(**param_dict)
        print(self._dynamixel_robo_config)
        self.dof = len(start_joints)

        if self.config.torque_joint_ids:
            driver = DynamixelDriver(self.config.torque_joint_ids, port=self.config.port, baudrate=57600)
            driver.set_torque_mode(True)
            driver.close()

    @property
    def action_features(self) -> dict:
        if self._tracking_endpoint:
            # UFRobot의 cartesian action과 같은 키를 낸다.
            return {key: float for key in CARTESIAN_ACTION_KEYS} | {"gripper.pos": float}
        act_ft = { f"J{i+1}.pos": float for i in range(self.dof) } | {"gripper.pos": float}
        return act_ft

    @property
    def feedback_features(self) -> dict:
        # fbk_ft = {
        #     "joint_position": {
        #     "dtype": "float",
        #     "shape": (self.dof+1,)
        #     }
        # }
        fbk_ft = { f"J{i+1}.pos": float for i in range(self.dof) } | {"gripper.pos": float}
        return fbk_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        from gello.agents.gello_agent import GelloAgent

        self.gello_agent = GelloAgent(port=self.config.port, dynamixel_config=self._dynamixel_robo_config)

        if self._tracking_endpoint and self.config.robot_ip:
            # 컨트롤러 FK 전용 읽기 연결. 모션 명령은 절대 보내지 않는다.
            # 컨트롤러 FK는 공장 캘리브레이션(공칭 DH 대비 수 mm)과 현재 TCP
            # offset을 포함하므로 로컬 FK의 모델 오차가 사라진다.
            try:
                from xarm.wrapper import XArmAPI

                self._fk_arm = XArmAPI(self.config.robot_ip, do_not_open=True)
                self._fk_arm.connect()
                code, _ = self._fk_arm.get_forward_kinematics(
                    [0.0] * 7, input_is_radian=True, return_is_radian=True
                )
                self._fk_arm_healthy = code == 0
            except Exception as exc:
                self._fk_arm = None
                self._fk_arm_healthy = False
                print(f"⚠️ [endpoint FK] 컨트롤러 FK 연결 실패({exc}). 로컬 FK로 동작합니다.", flush=True)
            if self._fk_arm_healthy:
                print("✅ [endpoint FK] 컨트롤러 get_forward_kinematics 사용 (캘리브레이션 포함)", flush=True)

        if not self._is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        self._is_connected = True
        super().connect(calibrate)
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        # TODO: Go to sync position slowly? Can not 
        pass

    @staticmethod
    def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
        """관절각 차이를 [-pi, pi]로 바꿔 2pi 경계의 순간 점프를 막습니다."""
        return np.arctan2(np.sin(values), np.cos(values))

    def _read_raw_action_array(self) -> np.ndarray:
        """부호와 기존 GELLO 오프셋이 적용된 현재 리더 값을 한 번 읽습니다."""
        start = time.perf_counter()
        fake_obs = dict({"joint_state": np.array([0.0]*(self.dof+1))}) # for agent.act() argument, actually no use
        action_array = self.gello_agent.act(fake_obs) # current gello joint pos as np.ndarray
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return np.asarray(action_array, dtype=float)

    def _leader_fk_matrix(self, joints_rad: np.ndarray) -> np.ndarray | None:
        """가상 관절 → base 기준 TCP 4x4(mm).

        컨트롤러 FK(캘리브레이션 포함)를 우선 사용하고, 일시 실패 시 None을
        돌려줘 호출부가 직전 값을 재사용하게 한다. 컨트롤러 연결 자체가 없으면
        로컬 공칭 FK를 사용한다(모델 오차 수 mm 감수).
        """
        if self._fk_arm is not None and self._fk_arm_healthy:
            try:
                code, pose = self._fk_arm.get_forward_kinematics(
                    list(joints_rad[:7]), input_is_radian=True, return_is_radian=True
                )
            except Exception:
                code = -1
            if code == 0:
                return Transformations.xyzrpy_to_rotation_matrix(*pose)
            return None
        return fk_tcp_matrix_mm(joints_rad, self._tcp_offset_matrix)

    def set_teleop_enabled(self, enabled: bool, obs=None):
        """에피소드 시작 시 리더 변화량의 영점을 현재 팔로워 관절값에 맞춥니다.

        팔 관절에만 상대 영점을 적용합니다. 그리퍼는 pi0 데이터에서 항상
        0=완전 열림, 1=완전 닫힘이라는 절대 의미를 유지해야 하므로 보정하지 않습니다.
        """
        if not enabled:
            self._teleop_enabled = False
            self._follower_reference = None
            self._last_leader_arm = None
            self._leader_accumulated_delta = None
            self._first_action_pending = False
            self._follower_ref_joints = None
            self._follower_ref_pos = None
            self._follower_ref_rot = None
            self._leader_ref_pos = None
            self._leader_ref_rot = None
            self._last_target_pos = None
            self._last_leader_tcp = None
            self._smooth_pos = None
            self._smooth_rot = None
            return

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "GELLO가 연결되지 않아 상대 영점을 설정할 수 없습니다."
            )
        if obs is None:
            raise RuntimeError("상대 영점 설정에는 현재 팔로워 관절 관측값이 필요합니다.")
        if not self.config.relative_alignment_enabled:
            raise RuntimeError(
                "relative_alignment_enabled가 꺼져 있습니다. "
                "초기 급이동을 막기 위해 텔레옵을 시작하지 않습니다."
            )

        follower_reference = None
        follower_tcp_matrix = None
        if self._tracking_endpoint:
            try:
                follower_pose_aa = [float(obs[key]) for key in CARTESIAN_ACTION_KEYS]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "팔로워 TCP 관측값 pose.x~pose.rz를 읽지 못했습니다. "
                    "endpoint 모드는 robot.control_space가 cartesian이어야 합니다."
                ) from exc
            follower_tcp_matrix = Transformations.xyzrxryrz_to_rotation_matrix(*follower_pose_aa)
        # endpoint 모드에서도 가상 관절의 기준점으로 팔로워 관절 관측이 필요하다
        # (cartesian_obs_include_joints: true가 전제).
        try:
            follower_reference = np.asarray(
                [float(obs[f"J{i + 1}.pos"]) for i in range(self.dof)],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "팔로워 관절 관측값 J1.pos~J7.pos를 읽지 못했습니다."
                + (" endpoint 모드는 robot.cartesian_obs_include_joints가 true여야 합니다."
                   if self._tracking_endpoint else "")
            ) from exc

        print(
            f"[리더 안정성 검사] {self.config.leader_stability_duration_s:.1f}초 동안 "
            f"GELLO를 움직이지 말고 잡고 계세요. 허용 변화는 "
            f"{self.config.leader_stability_max_delta_deg:.1f}°입니다.",
            flush=True,
        )
        samples = []
        previous_raw = None
        unwrapped = None
        start_time = time.monotonic()
        while time.monotonic() - start_time < self.config.leader_stability_duration_s:
            raw_arm = self._read_raw_action_array()[:self.dof]
            if previous_raw is None:
                unwrapped = raw_arm.copy()
            else:
                unwrapped = unwrapped + self._wrap_to_pi(raw_arm - previous_raw)
            samples.append(unwrapped.copy())
            previous_raw = raw_arm
            time.sleep(0.01)

        sample_array = np.stack(samples)
        movement_range_deg = np.rad2deg(np.ptp(sample_array, axis=0))
        largest_movement_deg = float(np.max(movement_range_deg))
        if largest_movement_deg > self.config.leader_stability_max_delta_deg:
            self.set_teleop_enabled(False)
            raise RuntimeError(
                f"GELLO가 안정성 검사 중 최대 {largest_movement_deg:.3f}° 움직였습니다. "
                f"허용값은 {self.config.leader_stability_max_delta_deg:.3f}°입니다."
            )

        # 안정성 검사가 끝난 바로 그 순간의 리더 자세를 변화량 0의 기준으로 삼습니다.
        # 따라서 첫 팔로워 목표는 수학적으로 follower_reference와 같아집니다.
        leader_reference = self._read_raw_action_array()[:self.dof]
        if self._tracking_endpoint:
            # 가상 관절의 기준점 = 팔로워의 실제 관절값. FK는 이 기준 관절에서
            # 계산하므로 GELLO 엔코더 offset 오차가 완전히 상쇄된다.
            self._follower_ref_joints = follower_reference.copy()
            ref_tcp = self._leader_fk_matrix(self._follower_ref_joints)
            if ref_tcp is None:
                self.set_teleop_enabled(False)
                raise RuntimeError("기준 관절의 컨트롤러 FK 계산에 실패해 endpoint 텔레옵을 시작하지 않습니다.")
            self._follower_ref_pos = follower_tcp_matrix[:3, 3].copy()
            self._follower_ref_rot = follower_tcp_matrix[:3, :3].copy()
            self._leader_ref_pos = ref_tcp[:3, 3].copy()
            self._leader_ref_rot = ref_tcp[:3, :3].copy()
            self._last_target_pos = self._follower_ref_pos.copy()
            self._last_leader_tcp = ref_tcp.copy()
            self._smooth_pos = self._follower_ref_pos.copy()
            self._smooth_rot = self._follower_ref_rot.copy()
            # 기준 관절 FK와 관측 TCP는 같은 컨트롤러 모델이므로 거의 일치해야
            # 한다. 차이가 크면 관측/FK 설정 문제(예: tcp_offset)다.
            abs_gap_mm = float(np.linalg.norm(self._leader_ref_pos - self._follower_ref_pos))
            if abs_gap_mm > 10.0:
                print(
                    f"⚠️ [endpoint 확인] 기준 관절 FK와 관측 TCP의 차이가 "
                    f"{abs_gap_mm:.1f}mm입니다. tcp_offset/관측 설정을 확인하세요.",
                    flush=True,
                )
        else:
            self._follower_reference = follower_reference
        self._last_leader_arm = leader_reference
        self._leader_accumulated_delta = np.zeros(self.dof, dtype=float)
        self._first_action_pending = True
        self._teleop_enabled = True
        mode_label = "endpoint(TCP)" if self._tracking_endpoint else "관절"
        print(
            f"✅ [상대 영점 설정 완료 · {mode_label} 추적] 첫 목표는 현재 팔로워 "
            f"자세와 같습니다. 그리퍼는 절대 [0,1] 매핑을 유지합니다.",
            flush=True,
        )

    def _get_endpoint_action(self, raw_action: np.ndarray) -> dict[str, np.ndarray]:
        """GELLO 관절 delta → 가상 관절 → FK → TCP delta → 절대 pose 목표.

        가상 관절 = 팔로워 기준 관절 + GELLO 누적 delta(2π 경계 처리 포함).
        목표 TCP = 팔로워 기준 TCP + (가상 관절 FK − 기준 관절 FK).
        실제 관절 궤적은 xArm 컨트롤러의 온라인 planning(mode 7)이 결정한다.
        """
        raw_arm = raw_action[:self.dof]
        leader_step = self._wrap_to_pi(raw_arm - self._last_leader_arm)
        self._leader_accumulated_delta = self._leader_accumulated_delta + leader_step
        self._last_leader_arm = raw_arm
        virtual_joints = self._follower_ref_joints + self._leader_accumulated_delta

        leader_tcp = self._leader_fk_matrix(virtual_joints)
        if leader_tcp is None:
            # 컨트롤러 FK 일시 실패: 직전 리더 TCP를 재사용해 프레임을 건너뛴다.
            leader_tcp = self._last_leader_tcp
        else:
            self._last_leader_tcp = leader_tcp
        delta_pos = leader_tcp[:3, 3] - self._leader_ref_pos
        delta_rot = leader_tcp[:3, :3] @ self._leader_ref_rot.T

        target_pos = self._follower_ref_pos + delta_pos
        target_rot = delta_rot @ self._follower_ref_rot

        if self._first_action_pending:
            first_delta_mm = float(np.linalg.norm(delta_pos))
            cos_angle = np.clip((np.trace(delta_rot) - 1.0) / 2.0, -1.0, 1.0)
            first_rot_deg = math.degrees(float(np.arccos(cos_angle)))
            if (
                first_delta_mm > self.config.first_action_max_delta_mm
                or first_rot_deg > self.config.first_action_max_delta_rot_deg
            ):
                self.set_teleop_enabled(False)
                raise RuntimeError(
                    f"첫 endpoint 목표가 팔로워 현재 TCP와 {first_delta_mm:.2f}mm / "
                    f"{first_rot_deg:.2f}° 다릅니다. 허용값은 "
                    f"{self.config.first_action_max_delta_mm:.1f}mm / "
                    f"{self.config.first_action_max_delta_rot_deg:.1f}°입니다."
                )
            print(
                f"✅ [첫 endpoint action 검사 통과] 위치 {first_delta_mm:.2f}mm, "
                f"회전 {first_rot_deg:.2f}°",
                flush=True,
            )
            self._first_action_pending = False

        frame_jump_mm = float(np.linalg.norm(target_pos - self._last_target_pos))
        if frame_jump_mm > self.config.max_frame_jump_mm:
            self.set_teleop_enabled(False)
            raise RuntimeError(
                f"endpoint 목표가 한 프레임에 {frame_jump_mm:.1f}mm 점프했습니다. "
                f"허용값은 {self.config.max_frame_jump_mm:.1f}mm입니다. "
                "GELLO 엔코더/케이블 상태를 확인하세요."
            )
        self._last_target_pos = target_pos.copy()

        # EMA 스무딩: 엔코더 갱신(약 18Hz)이 루프(30Hz)보다 느려 생기는 계단형
        # 목표를 보간한다. 급점프 가드는 스무딩 전 원목표에 이미 적용했다.
        alpha = self.config.endpoint_smoothing_alpha
        if alpha < 1.0:
            self._smooth_pos = self._smooth_pos + alpha * (target_pos - self._smooth_pos)
            rel_rotvec = np.asarray(
                Transformations.rotation_matrix_to_rxryrz(target_rot @ self._smooth_rot.T)
            )
            self._smooth_rot = (
                Transformations.rxryrz_to_rotation_matrix(*(alpha * rel_rotvec)) @ self._smooth_rot
            )
            target_pos = self._smooth_pos
            target_rot = self._smooth_rot
        else:
            self._smooth_pos = target_pos.copy()
            self._smooth_rot = target_rot.copy()

        target_matrix = np.eye(4)
        target_matrix[:3, :3] = target_rot
        target_matrix[:3, 3] = target_pos
        pose_aa = Transformations.rotation_matrix_to_xyzrxryrz(target_matrix)
        action = {key: float(value) for key, value in zip(CARTESIAN_ACTION_KEYS, pose_aa)}
        # 그리퍼는 관절 모드와 동일하게 절대 [0,1] 값을 유지한다.
        action["gripper.pos"] = raw_action[self.dof]
        return action

    def get_action(self) -> dict[str, np.ndarray]:
        if not self._teleop_enabled:
            raise RuntimeError("상대 영점 설정 전에는 GELLO action을 사용할 수 없습니다.")

        raw_action = self._read_raw_action_array()
        if self._tracking_endpoint:
            return self._get_endpoint_action(raw_action)
        raw_arm = raw_action[:self.dof]
        leader_step = self._wrap_to_pi(raw_arm - self._last_leader_arm)
        self._leader_accumulated_delta = self._leader_accumulated_delta + leader_step
        self._last_leader_arm = raw_arm
        aligned_arm = self._follower_reference + self._leader_accumulated_delta

        if self._first_action_pending:
            first_error_deg = np.rad2deg(
                np.abs(aligned_arm - self._follower_reference)
            )
            largest_error_deg = float(np.max(first_error_deg))
            if largest_error_deg > self.config.first_action_max_delta_deg:
                self.set_teleop_enabled(False)
                raise RuntimeError(
                    f"첫 GELLO 목표가 팔로워 현재 자세와 최대 {largest_error_deg:.3f}° "
                    f"다릅니다. 허용값은 {self.config.first_action_max_delta_deg:.3f}°입니다."
                )
            print(
                f"✅ [첫 action 검사 통과] 팔로워 현재값과 최대 차이 "
                f"{largest_error_deg:.3f}°",
                flush=True,
            )
            self._first_action_pending = False

        action = {}
        for i in range(self.dof):
            action.update({f"J{i+1}.pos": aligned_arm[i]})
        # 그리퍼는 상대 영점을 더하지 않습니다. 기존 GELLO 절대 [0,1] 값을 그대로
        # 저장하고 xArm에 보내야 pi0 학습/추론에서도 의미가 유지됩니다.
        action.update({"gripper.pos": raw_action[self.dof]})
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self._is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        if self._fk_arm is not None:
            try:
                self._fk_arm.disconnect()
            except Exception:
                pass
            self._fk_arm = None
            self._fk_arm_healthy = False
        self._is_connected = False
        logger.info(f"{self} disconnected.")
