#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("uf::webxr_teleop")
@dataclass
class WebXRTeleopConfig(TeleoperatorConfig):
    """Configuration for phone/VR pose teleoperation over WebXR."""

    host: str = "0.0.0.0"
    port: int = 8443
    tls_certfile: str = ""
    tls_keyfile: str = ""
    pairing_token: str = ""

    # WebXR reports metres and quaternions. The robot action uses mm and
    # axis-angle radians. Translation and rotation gains are deliberately
    # conservative for the first physical trials.
    translation_scale: float = 0.5
    rotation_scale: float = 0.5
    stale_timeout_s: float = 0.25
    max_input_frame_translation_mm: float = 50.0
    max_input_frame_rotation_deg: float = 20.0
    max_relative_translation_mm: float = 250.0
    max_relative_rotation_deg: float = 90.0
    smoothing_time_constant_s: float = 0.08

    # The phone frontend must send a held deadman button while motion is
    # allowed. Releasing it acts as a clutch and rebases the next movement.
    require_deadman: bool = True

    def __post_init__(self):
        self.id = "webxr_teleop" if self.id is None else self.id
        if not 1 <= self.port <= 65535:
            raise ValueError("WebXR port must be between 1 and 65535.")
        if not 0 < self.translation_scale <= 2.0:
            raise ValueError("translation_scale must be greater than 0 and at most 2.")
        if not 0 < self.rotation_scale <= 1.0:
            raise ValueError("rotation_scale must be greater than 0 and at most 1.")
        if not 0.05 <= self.stale_timeout_s <= 2.0:
            raise ValueError("stale_timeout_s must be between 0.05 and 2 seconds.")
        if not 1 <= self.max_input_frame_translation_mm <= 200:
            raise ValueError("max_input_frame_translation_mm must be between 1 and 200mm.")
        if not 1 <= self.max_input_frame_rotation_deg <= 90:
            raise ValueError("max_input_frame_rotation_deg must be between 1 and 90 degrees.")
        if not 10 <= self.max_relative_translation_mm <= 1000:
            raise ValueError("max_relative_translation_mm must be between 10 and 1000mm.")
        if not 5 <= self.max_relative_rotation_deg <= 180:
            raise ValueError("max_relative_rotation_deg must be between 5 and 180 degrees.")
        if not 0 <= self.smoothing_time_constant_s <= 1.0:
            raise ValueError("smoothing_time_constant_s must be between 0 and 1 second.")
