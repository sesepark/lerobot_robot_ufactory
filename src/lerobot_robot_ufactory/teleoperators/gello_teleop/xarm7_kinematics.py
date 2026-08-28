#!/usr/bin/env python
"""xArm7 로컬 순기구학(FK).

GELLO endpoint 추적 모드에서 리더(GELLO)의 관절값을 xArm7 관절값으로 해석해
"그 관절값일 때 xArm TCP가 있을 위치"를 30Hz 루프 안에서 네트워크 왕복 없이
계산한다. 관절 원점 파라미터는 UFACTORY 공식 xarm_description의
xarm7_default_kinematics.yaml 값이다.

검증: config의 start_joints(2026-07-30 확인 자세)에 기본 그리퍼 TCP offset
z=172mm를 적용하면 문서에 기록된 TCP (330, 50, 350, 180, 0, 0)와 수 mm
이내로 일치함을 확인했다. 런타임에서는 컨트롤러의 실제 tcp_offset과
get_position 결과를 비교하는 검사를 추가로 수행한다.
"""

from __future__ import annotations

import math

import numpy as np

from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations

# xarm_description/config/kinematics/default/xarm7_default_kinematics.yaml
# 각 관절 원점: (x, y, z[m], roll, pitch, yaw[rad]), 회전축은 모두 로컬 z축.
XARM7_JOINT_ORIGINS = (
    (0.0, 0.0, 0.267, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -1.5708, 0.0, 0.0),
    (0.0, -0.293, 0.0, 1.5708, 0.0, 0.0),
    (0.0525, 0.0, 0.0, 1.5708, 0.0, 0.0),
    (0.0775, -0.3425, 0.0, 1.5708, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.5708, 0.0, 0.0),
    (0.076, 0.097, 0.0, -1.5708, 0.0, 0.0),
)


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fk_flange_matrix_m(joints_rad) -> np.ndarray:
    """관절각 7개(rad) → base 기준 flange 4x4 변환(단위 m)."""
    joints = np.asarray(joints_rad, dtype=float)
    if joints.shape != (7,):
        raise ValueError(f"xArm7 FK에는 관절각 7개가 필요합니다. 입력: {joints.shape}")
    T = np.eye(4)
    for (x, y, z, roll, pitch, yaw), q in zip(XARM7_JOINT_ORIGINS, joints):
        A = np.eye(4)
        A[:3, :3] = Transformations.rpy_to_rotation_matrix(roll, pitch, yaw) @ _rotz(q)
        A[:3, 3] = (x, y, z)
        T = T @ A
    return T


def tcp_offset_matrix_mm(tcp_offset_mm_rad) -> np.ndarray:
    """컨트롤러 tcp_offset [x,y,z(mm), roll,pitch,yaw(rad)] → 4x4 변환(단위 mm)."""
    offset = np.asarray(tcp_offset_mm_rad, dtype=float)
    if offset.shape != (6,):
        raise ValueError(f"tcp_offset은 6개 값이어야 합니다. 입력: {offset.shape}")
    return Transformations.xyzrpy_to_rotation_matrix(*offset)


def fk_tcp_matrix_mm(joints_rad, tcp_offset_matrix: np.ndarray) -> np.ndarray:
    """관절각 7개(rad) → base 기준 TCP 4x4 변환(단위 mm)."""
    T = fk_flange_matrix_m(joints_rad)
    T_mm = T.copy()
    T_mm[:3, 3] *= 1000.0
    return T_mm @ tcp_offset_matrix


def fk_tcp_pose_aa_mm(joints_rad, tcp_offset_matrix: np.ndarray) -> list[float]:
    """관절각 7개(rad) → [x,y,z(mm), rx,ry,rz(axis-angle rad)].

    반환 형식은 UFRobot cartesian 관측(pose.*)과 set_position_aa 명령과 같다.
    """
    return Transformations.rotation_matrix_to_xyzrxryrz(
        fk_tcp_matrix_mm(joints_rad, tcp_offset_matrix)
    )
