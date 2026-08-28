from __future__ import annotations

import time
from typing import get_type_hints

import numpy as np
import pytest

from lerobot_robot_ufactory.teleoperators.webxr_teleop import (
    WebXRTeleop,
    WebXRTeleopConfig,
)


def message(
    sequence: int,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    move: bool = False,
    quaternion=(0.0, 0.0, 0.0, 1.0),
):
    return {
        "type": "pose",
        "data": {
            "sequence": sequence,
            "position": {"x": x, "y": y, "z": z},
            "orientation": dict(zip(("x", "y", "z", "w"), quaternion, strict=True)),
            "move": move,
            "gripper": "open",
            "fps": 30.0,
        },
    }


def observation(x=330.0, y=50.0, z=350.0):
    return {
        "pose.x": x,
        "pose.y": y,
        "pose.z": z,
        "pose.rx": 0.0,
        "pose.ry": 0.0,
        "pose.rz": 0.0,
    }


def armed_teleop(**overrides) -> WebXRTeleop:
    defaults = {
        "max_input_frame_translation_mm": 200.0,
        "smoothing_time_constant_s": 0.0,
    }
    defaults.update(overrides)
    teleop = WebXRTeleop(WebXRTeleopConfig(**defaults))
    teleop._is_connected = True
    teleop._client_connected = True
    teleop.ingest_message(message(1))
    teleop.set_teleop_enabled(True, observation())
    return teleop


def test_webxr_basis_and_translation_scale():
    teleop = armed_teleop(translation_scale=0.5)
    teleop.ingest_message(message(2, move=True))
    teleop.update_observation(observation())
    teleop.get_action()  # deadman rising edge establishes the clutch reference

    # WebXR -Z is forward. 10cm phone motion at scale 0.5 becomes +50mm xArm X.
    teleop.ingest_message(message(3, z=-0.1, move=True))
    action = teleop.get_action()
    assert action["pose.x"] == pytest.approx(380.0)
    assert action["pose.y"] == pytest.approx(50.0)
    assert action["pose.z"] == pytest.approx(350.0)


def test_deadman_release_holds_latest_actual_robot_pose():
    teleop = armed_teleop()
    teleop.ingest_message(message(2, move=True))
    teleop.get_action()
    teleop.ingest_message(message(3, z=-0.05, move=True))
    teleop.get_action()

    latest = observation(342.0, 49.0, 351.0)
    teleop.update_observation(latest)
    teleop.ingest_message(message(4, z=-0.05, move=False))
    action = teleop.get_action()
    assert np.allclose(
        [action["pose.x"], action["pose.y"], action["pose.z"]],
        [342.0, 49.0, 351.0],
    )


def test_tracking_jump_latches_fault_and_holds():
    teleop = armed_teleop(max_input_frame_translation_mm=20.0)
    teleop.ingest_message(message(2, move=True))
    teleop.get_action()
    latest = observation(331.0, 52.0, 349.0)
    teleop.update_observation(latest)
    teleop.ingest_message(message(3, z=-0.10, move=True))
    action = teleop.get_action()
    assert "jumped" in teleop.fault_reason
    assert [action["pose.x"], action["pose.y"], action["pose.z"]] == pytest.approx(
        [331.0, 52.0, 349.0]
    )


def test_stale_pose_during_motion_latches_fault():
    teleop = armed_teleop(stale_timeout_s=0.05)
    teleop.ingest_message(message(2, move=True))
    teleop.get_action()
    time.sleep(0.06)
    teleop.get_action()
    assert "timed out" in teleop.fault_reason


def test_invalid_quaternion_is_rejected():
    teleop = WebXRTeleop(WebXRTeleopConfig())
    with pytest.raises(ValueError, match="quaternion norm"):
        teleop.ingest_message(message(1, quaternion=(0.0, 0.0, 0.0, 0.0)))


def test_out_of_order_sequence_is_ignored():
    teleop = WebXRTeleop(WebXRTeleopConfig())
    teleop.ingest_message(message(5, x=0.1))
    teleop.ingest_message(message(4, x=0.2))
    assert teleop.get_phone_status()["sequence"] == 5


def test_fastapi_resolves_websocket_parameter_type():
    teleop = WebXRTeleop(WebXRTeleopConfig(pairing_token="unit-test-token"))
    app = teleop._make_app()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/ws")
    assert get_type_hints(endpoint)["websocket"].__name__ == "WebSocket"
