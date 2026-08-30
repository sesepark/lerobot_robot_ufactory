# 이 fork에서 무엇을 추가했는가

[xArm-Developer/lerobot_robot_ufactory](https://github.com/xArm-Developer/lerobot_robot_ufactory)의
`e492233`에서 갈라져 나와, UFACTORY xArm7 텔레오퍼레이션에 필요한 기능을 추가한 브랜치입니다.

**상류 대비 14개 파일 / 1,811줄 추가, 51줄 수정.**

> *Work added on top of upstream `e492233`: a GELLO endpoint-tracking mode, a local xArm7
> forward-kinematics implementation, and a WebXR phone teleoperation module — 14 files,
> +1,811 / −51. Written in Korean below.*

---

## 추가한 것

| 영역 | 파일 | 줄 |
|---|---|---|
| GELLO endpoint 추적 모드 | `teleoperators/gello_teleop/gello_teleop.py`, `..._config.py` | +397 |
| xArm7 로컬 순기구학 | `teleoperators/gello_teleop/xarm7_kinematics.py` | +78 |
| WebXR 휴대폰 텔레옵 | `teleoperators/webxr_teleop/` (4개 파일) | +830 |
| 로봇·기록 스크립트 연동 | `robots/uf_robot/`, `scripts/` | +418 |
| 테스트 | `tests/test_webxr_teleop.py` | +133 |

## 설계에서 신경 쓴 부분

### 엔코더 offset이 endpoint 오차로 새지 않게 한다

GELLO endpoint 추적은 리더의 **절대 관절값을 FK에 넣지 않습니다.** 팔로워의 현재 관절을 기준
삼아 리더의 delta만 더한 "가상 관절"을 FK에 넣습니다.

절대값을 쓰면 시작 시점에 존재하는 수 °의 리더 엔코더 offset이 FK 비선형성을 타고, 특히
null-space 움직임에서 endpoint 오차로 새어 나옵니다. 리더암은 매번 정확히 같은 자세에서
시작하지 않기 때문에 이 offset은 없앨 수 있는 종류가 아니고, 상쇄해야 하는 종류입니다.

### 30Hz 제어 루프 안에서 네트워크 왕복을 없앤다

`xarm7_kinematics.py`에 xArm7 순기구학을 직접 구현했습니다. 관절 원점 파라미터는 UFACTORY 공식
`xarm_description`의 `xarm7_default_kinematics.yaml` 값을 씁니다. 컨트롤러에 FK를 물어보면
30Hz 루프마다 왕복이 생기기 때문입니다.

직접 구현한 FK는 믿고 쓰는 게 아니라 검증해서 씁니다. 문서에 기록된 확인 자세에 기본 그리퍼
TCP offset(z=172mm)을 적용해 기록된 TCP 값과 수 mm 이내로 일치함을 확인했고, 런타임에도
컨트롤러의 실제 `tcp_offset`과 `get_position` 결과를 비교하는 검사를 둡니다. 컨트롤러 FK를 쓸 수
있을 때는 공장 캘리브레이션과 실제 TCP offset이 반영된 그쪽을 우선합니다.

### 안전 한계를 "첫 프레임"과 "매 프레임"으로 나눈다

상대 정렬 직후의 첫 목표와, 이후 프레임 간 이동에 각각 다른 한계를 둡니다
(`first_action_max_delta_mm`, `first_action_max_delta_rot_deg`와 프레임 간 한계).
첫 목표는 정렬이 잘못됐을 때 로봇이 크게 튀는 유일한 순간이고, 이후 프레임의 급격한 점프는
엔코더 이상 신호이기 때문에 성격이 다릅니다.

### 휴대폰 텔레옵의 페어링을 실행마다 새로 발급한다

WebXR 모듈은 실행할 때마다 새 페어링 토큰을 발급하고, HTTP 진입점과 WebSocket 양쪽에서
`secrets.compare_digest`로 검사합니다. WebXR은 HTTPS를 요구하므로 TLS 인증서가 필요한데,
인증서는 로컬 CA로 그때그때 발급하고 **저장소에 포함하지 않습니다.** 없으면 조용히 평문으로
떨어지지 않고 셋업 스크립트를 안내하며 실패합니다.

## 상류에 올리지 않은 이유

xArm7과 특정 GELLO 구성, 그리고 실험실 장비에 맞춰 만든 기능이라 상류의 일반적인 지원 범위와
결이 다릅니다. 기여 범위가 그대로 보이도록 fork 브랜치로 분리해 두었습니다.

## 관련 저장소

- [sesepark/xarm7-pi0-teleop](https://github.com/sesepark/xarm7-pi0-teleop) — 이 텔레옵을 실제로
  운용하는 데이터 수집·학습 워크스페이스
- upstream: [xArm-Developer/lerobot_robot_ufactory](https://github.com/xArm-Developer/lerobot_robot_ufactory)
