#!/usr/bin/env python

from __future__ import annotations

import json
import math
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_ufactory.devices.umi.vive_tracker.transformations import Transformations
from lerobot_robot_ufactory.teleoperators.base_teleop.base_teleop import UFBaseTeleop

from .webxr_teleop_config import WebXRTeleopConfig

CARTESIAN_ACTION_KEYS = ("pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz")
_FastAPIWebSocket = Any

# WebXR uses right/up/back (x/y/z). xArm's base convention is treated as
# forward/left/up. This is the same basis conversion used by SpesRobotics'
# Apache-2.0 WebXR teleop reference implementation.
WEBXR_TO_XARM = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class WebXRSample:
    sequence: int
    received_at: float
    position_mm: np.ndarray
    rotation: np.ndarray
    move: bool
    gripper_closed: bool
    fps: float


def _rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _scaled_rotation(rotation: np.ndarray, scale: float) -> np.ndarray:
    rotvec = np.asarray(Transformations.rotation_matrix_to_rxryrz(rotation), dtype=float)
    return Transformations.rxryrz_to_rotation_matrix(*(rotvec * scale))


def _pose_from_observation(obs: dict[str, Any]) -> np.ndarray:
    try:
        values = [float(obs[key]) for key in CARTESIAN_ACTION_KEYS]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "WebXR requires Cartesian robot observations pose.x~pose.rz. "
            "Set robot.control_space to cartesian."
        ) from exc
    if not np.all(np.isfinite(values)):
        raise RuntimeError("The robot Cartesian observation contains NaN or infinity.")
    return Transformations.xyzrxryrz_to_rotation_matrix(*values)


class WebXRTeleop(UFBaseTeleop):
    """Phone/VR WebXR input adapter for the existing xArm Cartesian loop.

    The networking thread never calls the robot. It only publishes the newest
    validated WebXR sample. ``get_action`` consumes that sample from the normal
    LeRobot loop, rebases it against the latest follower observation, and emits
    the same absolute Cartesian action schema as GELLO endpoint mode.
    """

    config_class = WebXRTeleopConfig
    name = "WebXR Teleop for xArm"

    def __init__(self, config: WebXRTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True
        self._teleop_enabled = False
        self._lock = threading.RLock()
        self._latest_sample: WebXRSample | None = None
        self._latest_observation_pose: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None
        self._smooth_pose: np.ndarray | None = None
        self._phone_reference: np.ndarray | None = None
        self._follower_reference: np.ndarray | None = None
        self._previous_input_pose: np.ndarray | None = None
        self._motion_active = False
        self._fault_reason: str | None = None
        self._fault_reported = False
        self._last_smoothing_time: float | None = None
        self._client_connected = False
        self._active_client: str | None = None
        self._pairing_token = config.pairing_token or secrets.token_urlsafe(6)
        self._server = None
        self._server_thread: threading.Thread | None = None

    @property
    def action_features(self) -> dict:
        return {key: float for key in CARTESIAN_ACTION_KEYS} | {"gripper.pos": float}

    @property
    def feedback_features(self) -> dict:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def pairing_token(self) -> str:
        return self._pairing_token

    @property
    def fault_reason(self) -> str | None:
        with self._lock:
            return self._fault_reason

    def calibrate(self) -> None:
        self._is_calibrated = True

    def configure(self) -> None:
        pass

    @staticmethod
    def _local_ip() -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
        except OSError:
            return "127.0.0.1"
        finally:
            sock.close()

    @property
    def phone_url(self) -> str:
        return f"https://{self._local_ip()}:{self.config.port}/?token={self._pairing_token}"

    def _validate_tls(self) -> tuple[Path, Path]:
        certfile = Path(self.config.tls_certfile).expanduser()
        keyfile = Path(self.config.tls_keyfile).expanduser()
        if not certfile.is_file() or not keyfile.is_file():
            raise RuntimeError(
                "WebXR HTTPS certificate is missing. Run ./scripts/setup_webxr_tls.sh first. "
                f"Expected cert={certfile}, key={keyfile}."
            )
        return certfile, keyfile

    def _make_app(self):
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import HTMLResponse, JSONResponse
        except ImportError as exc:
            raise RuntimeError(
                "WebXR server dependencies are missing. Install the project with the webxr extra."
            ) from exc

        # This module uses postponed annotations while FastAPI is an optional,
        # lazy import. FastAPI resolves endpoint annotations from module globals;
        # expose this alias so it recognizes the parameter as a WebSocket rather
        # than treating it as a missing query parameter (HTTP 403 handshake).
        globals()["_FastAPIWebSocket"] = WebSocket

        frontend = Path(__file__).with_name("index.html")
        app = FastAPI(title="xArm7 WebXR Teleop")

        @app.get("/")
        async def index(token: str = ""):
            if not secrets.compare_digest(token, self._pairing_token):
                return HTMLResponse("Invalid or missing pairing token.", status_code=403)
            return HTMLResponse(frontend.read_text(encoding="utf-8"))

        @app.get("/health")
        async def health():
            with self._lock:
                sample_age_ms = None
                if self._latest_sample is not None:
                    sample_age_ms = round(
                        (time.monotonic() - self._latest_sample.received_at) * 1000, 1
                    )
                return JSONResponse(
                    {
                        "server": "ok",
                        "phone_connected": self._client_connected,
                        "teleop_enabled": self._teleop_enabled,
                        "motion_active": self._motion_active,
                        "sample_age_ms": sample_age_ms,
                        "fault": self._fault_reason,
                    }
                )

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: _FastAPIWebSocket):
            token = websocket.query_params.get("token", "")
            if not secrets.compare_digest(token, self._pairing_token):
                await websocket.close(code=1008, reason="invalid pairing token")
                return

            client = f"{websocket.client.host}:{websocket.client.port}"
            with self._lock:
                if self._client_connected:
                    reject = True
                else:
                    reject = False
                    self._client_connected = True
                    self._active_client = client
                    # A newly opened page restarts its sequence counter.
                    self._latest_sample = None
            if reject:
                await websocket.close(code=1008, reason="another controller is already connected")
                return

            await websocket.accept()
            print(f"✅ [WebXR] phone connected: {client}", flush=True)
            try:
                while True:
                    payload = json.loads(await websocket.receive_text())
                    self.ingest_message(payload)
            except WebSocketDisconnect:
                pass
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"⚠️ [WebXR] invalid phone message: {exc}", flush=True)
            finally:
                with self._lock:
                    was_moving = bool(self._latest_sample and self._latest_sample.move)
                    self._client_connected = False
                    self._active_client = None
                    if was_moving:
                        self._latch_fault_locked("Phone/WebSocket disconnected while moving.")
                print(f"⚠️ [WebXR] phone disconnected: {client}", flush=True)

        return app

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        certfile, keyfile = self._validate_tls()
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "uvicorn is missing. Install the project with the webxr extra."
            ) from exc

        app = self._make_app()
        server_config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            ssl_certfile=str(certfile),
            ssl_keyfile=str(keyfile),
            log_level="warning",
            ws_ping_interval=5.0,
            ws_ping_timeout=5.0,
        )
        self._server = uvicorn.Server(server_config)
        self._server_thread = threading.Thread(
            target=self._server.run,
            name="webxr-https-server",
            daemon=True,
        )
        self._server_thread.start()
        deadline = time.monotonic() + 5.0
        while not self._server.started and self._server_thread.is_alive():
            if time.monotonic() >= deadline:
                self._server.should_exit = True
                raise RuntimeError("Timed out while starting the WebXR HTTPS server.")
            time.sleep(0.02)
        if not self._server_thread.is_alive():
            raise RuntimeError(
                f"WebXR HTTPS server failed to start on {self.config.host}:{self.config.port}."
            )

        self._is_connected = True
        super().connect(calibrate)
        print("============================================================", flush=True)
        print("WebXR phone controller is ready.", flush=True)
        print(f"Open this exact URL on the phone:\n{self.phone_url}", flush=True)
        print(
            "Keep the phone Move button released, then press Space here to arm teleop.", flush=True
        )
        print("============================================================", flush=True)

    def disconnect(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=3.0)
        self._server = None
        self._server_thread = None
        self._is_connected = False
        self._teleop_enabled = False
        super().disconnect()

    def set_teleop_enabled(self, enabled: bool, obs=None):
        with self._lock:
            if not enabled:
                self._teleop_enabled = False
                self._motion_active = False
                self._phone_reference = None
                self._follower_reference = None
                self._previous_input_pose = None
                self._fault_reason = None
                self._fault_reported = False
                return
            if not self._is_connected:
                raise DeviceNotConnectedError("The WebXR server is not connected.")
            if not self._client_connected or self._latest_sample is None:
                raise RuntimeError("Connect the phone WebXR page before enabling teleop.")
            age = time.monotonic() - self._latest_sample.received_at
            if age > self.config.stale_timeout_s:
                raise RuntimeError(f"The latest phone pose is stale ({age * 1000:.0f}ms).")
            if self.config.require_deadman and self._latest_sample.move:
                raise RuntimeError("Release the phone Move button before enabling teleop.")
            follower_pose = _pose_from_observation(obs)
            self._latest_observation_pose = follower_pose.copy()
            self._target_pose = follower_pose.copy()
            self._smooth_pose = follower_pose.copy()
            self._last_smoothing_time = time.monotonic()
            self._motion_active = False
            self._phone_reference = None
            self._follower_reference = None
            self._previous_input_pose = None
            self._fault_reason = None
            self._fault_reported = False
            self._teleop_enabled = True
        print("✅ [WebXR] armed at the current xArm TCP pose.", flush=True)

    def update_observation(self, obs: dict[str, Any]) -> None:
        pose = _pose_from_observation(obs)
        with self._lock:
            self._latest_observation_pose = pose

    def _latch_fault_locked(self, reason: str) -> None:
        if self._fault_reason is None:
            self._fault_reason = reason
        self._motion_active = False
        self._phone_reference = None
        self._follower_reference = None
        self._previous_input_pose = None

    def _sample_from_payload(self, payload: dict[str, Any]) -> WebXRSample:
        if payload.get("type") != "pose":
            raise ValueError("message type must be 'pose'")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("message data must be an object")
        position = data.get("position")
        orientation = data.get("orientation")
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            raise ValueError("position and orientation are required")
        xyz_m = np.asarray([position.get(axis) for axis in "xyz"], dtype=float)
        quaternion = np.asarray(
            [orientation.get(axis) for axis in ("x", "y", "z", "w")], dtype=float
        )
        if not np.all(np.isfinite(xyz_m)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("pose contains NaN or infinity")
        norm = float(np.linalg.norm(quaternion))
        if not 0.9 <= norm <= 1.1:
            raise ValueError(f"quaternion norm is invalid: {norm:.3f}")
        quaternion /= norm
        webxr_rotation = Transformations.quaternion_to_rotation_matrix(quaternion)
        position_mm = WEBXR_TO_XARM @ (xyz_m * 1000.0)
        rotation = WEBXR_TO_XARM @ webxr_rotation @ WEBXR_TO_XARM.T
        sequence = int(data.get("sequence", 0))
        fps = float(data.get("fps", 0.0))
        if not math.isfinite(fps) or fps < 0:
            fps = 0.0
        return WebXRSample(
            sequence=sequence,
            received_at=time.monotonic(),
            position_mm=position_mm,
            rotation=rotation,
            move=bool(data.get("move", False)),
            gripper_closed=data.get("gripper") == "close",
            fps=fps,
        )

    def ingest_message(self, payload: dict[str, Any]) -> None:
        """Validate and publish a phone message; public for deterministic tests."""
        sample = self._sample_from_payload(payload)
        with self._lock:
            if self._latest_sample is not None and sample.sequence <= self._latest_sample.sequence:
                return
            self._latest_sample = sample

    def get_phone_status(self) -> dict[str, Any]:
        """Return a read-only status snapshot for the phone-only test command."""
        with self._lock:
            age_ms = None
            if self._latest_sample is not None:
                age_ms = (time.monotonic() - self._latest_sample.received_at) * 1000.0
            return {
                "connected": self._client_connected,
                "sample_age_ms": age_ms,
                "sequence": self._latest_sample.sequence if self._latest_sample else None,
                "move": self._latest_sample.move if self._latest_sample else False,
                "gripper_closed": (
                    self._latest_sample.gripper_closed if self._latest_sample else False
                ),
                "fps": self._latest_sample.fps if self._latest_sample else 0.0,
                "fault": self._fault_reason,
            }

    @staticmethod
    def _sample_pose(sample: WebXRSample) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = sample.rotation
        pose[:3, 3] = sample.position_mm
        return pose

    def _action_from_pose_locked(self, pose: np.ndarray, gripper_closed: bool) -> dict[str, float]:
        values = Transformations.rotation_matrix_to_xyzrxryrz(pose)
        return {
            key: float(value) for key, value in zip(CARTESIAN_ACTION_KEYS, values, strict=True)
        } | {"gripper.pos": 1.0 if gripper_closed else 0.0}

    def _hold_action_locked(self, sample: WebXRSample) -> dict[str, float]:
        if self._latest_observation_pose is not None:
            self._target_pose = self._latest_observation_pose.copy()
            self._smooth_pose = self._latest_observation_pose.copy()
        if self._target_pose is None:
            raise RuntimeError("No robot pose is available for WebXR hold.")
        return self._action_from_pose_locked(self._target_pose, sample.gripper_closed)

    def get_action(self) -> dict[str, float]:
        with self._lock:
            if not self._teleop_enabled:
                raise RuntimeError("WebXR teleop is not armed. Press Space to enable it.")
            sample = self._latest_sample
            if sample is None:
                raise RuntimeError("No WebXR pose has been received.")

            now = time.monotonic()
            age = now - sample.received_at
            if self._motion_active and age > self.config.stale_timeout_s:
                self._latch_fault_locked(f"WebXR pose timed out while moving ({age * 1000:.0f}ms).")
            if self._fault_reason is not None:
                if not self._fault_reported:
                    print(
                        f"❌ [WebXR safety hold] {self._fault_reason} "
                        "Release Move, pause with Space, then re-arm.",
                        flush=True,
                    )
                    self._fault_reported = True
                return self._hold_action_locked(sample)

            move = sample.move or not self.config.require_deadman
            if not move:
                self._motion_active = False
                self._phone_reference = None
                self._follower_reference = None
                self._previous_input_pose = None
                return self._hold_action_locked(sample)

            input_pose = self._sample_pose(sample)
            if not self._motion_active:
                if self._latest_observation_pose is None:
                    raise RuntimeError("No current robot pose is available for WebXR clutching.")
                self._motion_active = True
                self._phone_reference = input_pose.copy()
                self._follower_reference = self._latest_observation_pose.copy()
                self._previous_input_pose = input_pose.copy()
                self._target_pose = self._latest_observation_pose.copy()
                self._smooth_pose = self._latest_observation_pose.copy()
                self._last_smoothing_time = now
                return self._action_from_pose_locked(self._target_pose, sample.gripper_closed)

            frame_translation = float(
                np.linalg.norm(input_pose[:3, 3] - self._previous_input_pose[:3, 3])
            )
            frame_rotation = _rotation_angle_rad(
                input_pose[:3, :3] @ self._previous_input_pose[:3, :3].T
            )
            self._previous_input_pose = input_pose.copy()
            if frame_translation > self.config.max_input_frame_translation_mm:
                self._latch_fault_locked(
                    f"Phone tracking jumped {frame_translation:.1f}mm in one frame."
                )
                return self._hold_action_locked(sample)
            frame_rotation_deg = math.degrees(frame_rotation)
            if frame_rotation_deg > self.config.max_input_frame_rotation_deg:
                self._latch_fault_locked(
                    f"Phone tracking jumped {frame_rotation_deg:.1f} degrees in one frame."
                )
                return self._hold_action_locked(sample)

            relative_translation = input_pose[:3, 3] - self._phone_reference[:3, 3]
            relative_translation *= self.config.translation_scale
            relative_rotation = input_pose[:3, :3] @ self._phone_reference[:3, :3].T
            relative_rotation = _scaled_rotation(relative_rotation, self.config.rotation_scale)
            translation_norm = float(np.linalg.norm(relative_translation))
            rotation_deg = math.degrees(_rotation_angle_rad(relative_rotation))
            if translation_norm > self.config.max_relative_translation_mm:
                self._latch_fault_locked(
                    "Relative phone movement exceeded "
                    f"{self.config.max_relative_translation_mm:.0f}mm."
                )
                return self._hold_action_locked(sample)
            if rotation_deg > self.config.max_relative_rotation_deg:
                self._latch_fault_locked(
                    "Relative phone rotation exceeded "
                    f"{self.config.max_relative_rotation_deg:.0f} degrees."
                )
                return self._hold_action_locked(sample)

            target = np.eye(4)
            target[:3, 3] = self._follower_reference[:3, 3] + relative_translation
            target[:3, :3] = relative_rotation @ self._follower_reference[:3, :3]
            self._target_pose = target

            if self._smooth_pose is None or self.config.smoothing_time_constant_s == 0:
                self._smooth_pose = target.copy()
            else:
                dt = max(0.0, now - (self._last_smoothing_time or now))
                alpha = 1.0 - math.exp(-dt / self.config.smoothing_time_constant_s)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                self._smooth_pose[:3, 3] += alpha * (target[:3, 3] - self._smooth_pose[:3, 3])
                smooth_delta = target[:3, :3] @ self._smooth_pose[:3, :3].T
                self._smooth_pose[:3, :3] = (
                    _scaled_rotation(smooth_delta, alpha) @ self._smooth_pose[:3, :3]
                )
            self._last_smoothing_time = now
            return self._action_from_pose_locked(self._smooth_pose, sample.gripper_closed)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError
