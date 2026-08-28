import sys
import argparse
import logging
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from pprint import pformat
import lerobot_robot_ufactory # patch
from lerobot.scripts.lerobot_record import register_third_party_plugins
from lerobot.processor import (
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
)
from lerobot_robot_ufactory.configs import parser
from lerobot_robot_ufactory.utils.utils import is_headless, init_keyboard_listener
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop


@dataclass
class TeleopConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    fps: int = 30
    # True이면 UFACTORY 원본처럼 Space 입력 전까지 일시정지합니다.
    # run.sh처럼 별도의 시작 확인 절차가 있는 경우에만 False로 설정합니다.
    start_paused: bool = True

    def __post_init__(self):
        if hasattr(self.robot, 'robots'):
            for _, robot in self.robot.robots.items():
                robot.cameras = {}
        else:
            self.robot.cameras = {}


def teleop_loop(cfg: TeleopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    if hasattr(cfg.robot, "teleop"):
        cfg.robot.teleop = teleop
    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot.connect()
    teleop.connect()

    sleep_time_s = 1 / cfg.fps

    is_evt = not is_headless()
    is_uf_teleop = isinstance(teleop, UFBaseTeleop)

    is_reset = False
    is_paused = cfg.start_paused
    events = {"exit": False}
    listener = None
    key_dict = {}

    if is_evt:
        from pynput import keyboard

        key_dict = {
            keyboard.Key.esc: 0,    # exit
            keyboard.Key.left: 0,   # reset and pause
            keyboard.Key.space: 0,  # start/pause
            keyboard.Key.enter: 0,  # help
        }

        def on_press(key):
            if key_dict.get(key, 1) == 0:
                try:
                    if key == keyboard.Key.esc:
                        events["exit"] = True
                        print("\nEscape key pressed. Stopping ...")
                except Exception as e:
                    print(f"Error handling key press: {e}")
            if key in key_dict:
                key_dict[key] = True

        def on_release(key):
            try:
                if key == keyboard.Key.enter:
                    if is_paused:
                        if is_reset:
                            print('⌨   [ESC] Exit  [Space] Reset / Start  [←] Reset')
                        else:
                            print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
                    else:
                        print('⌨   [ESC] Exit  [Space] Pause  [←] Pause / Reset')
            except Exception as e:
                print(f"Error handling key release: {e}")
            if key in key_dict:
                key_dict[key] = False

        listener, events = init_keyboard_listener(events=events, on_press=on_press, on_release=on_release)
        if not is_paused and is_uf_teleop:
        # run.sh의 start 확인과 3초 안전 대기를 이미 통과했습니다.
            # 현재 로봇 자세를 기준으로 텔레옵을 활성화합니다.
            obs = robot.get_observation()
            teleop.set_teleop_enabled(True, obs)
        print("\n********** Teleop Control Loop Start **********")
        if is_paused:
            print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
        else:
            print('✅ 텔레옵 자동 시작 완료  [Space] Pause  [ESC] Exit  [←] Reset')
    else:
        if is_paused:
            input('⌨   Press Enter to start teleop >>> ')
        if is_uf_teleop:
            obs = robot.get_observation()
            teleop.set_teleop_enabled(True, obs)
        is_paused = False
        is_reset = False
        print("\n********** Teleop Control Loop Start **********")

    key_space_pressed = False
    key_left_pressed = False

    while not events["exit"]:
        start_loop_t = time.perf_counter()

        if is_evt:
            if key_dict[keyboard.Key.left] and not key_left_pressed:
                key_left_pressed = True
                is_reset = True
                if not is_paused:
                    is_paused = True
                    if is_uf_teleop:
                        teleop.set_teleop_enabled(False)
                print('⌨   [ESC] Exit  [Space] Reset / Start  [←] Reset')
            elif not key_dict[keyboard.Key.left] and key_left_pressed:
                key_left_pressed = False

            if key_dict[keyboard.Key.space] and not key_space_pressed:
                key_space_pressed = True
                is_paused = not is_paused
                if is_paused:
                    if is_uf_teleop:
                        teleop.set_teleop_enabled(False)
                    # print('========== Teleop is paused ==========')
                    print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
                else:
                    if is_reset:
                        is_reset = False
                        robot.configure()
                    # print('========== Teleop is start ==========')
                    if is_uf_teleop:
                        obs = robot.get_observation()
                        teleop.set_teleop_enabled(True, obs)
                    print('⌨   [ESC] Exit  [Space] Pause  [←] Reset')
                continue
            elif not key_dict[keyboard.Key.space] and key_space_pressed:
                key_space_pressed = False

            if is_reset or is_paused:
                continue

        # Get robot observation
        obs = robot.get_observation()
        # WebXR uses the current follower pose to hold immediately when its
        # deadman is released or tracking becomes stale.
        if hasattr(teleop, "update_observation"):
            teleop.update_observation(obs)

        act = teleop.get_action()
        act_processed_teleop = teleop_action_processor((act, obs))

        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(sleep_time_s - dt_s)
    
    print("\n********** Teleop Control Loop Exit **********")
    robot.disconnect()
    teleop.disconnect()
    if is_evt and listener is not None:
        listener.stop()

@parser.wrap()
def get_cfg(cfg: TeleopConfig) -> TeleopConfig:
    return cfg

def main():
    parser = argparse.ArgumentParser(description='configuration args')
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    teleop_loop(cfg)


if __name__ == "__main__":
    main()
