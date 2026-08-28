import sys
import copy
import time
import math
import queue
import argparse
import logging
import shutil
import threading
from pathlib import Path
from dataclasses import dataclass
import lerobot_robot_ufactory # patch
from lerobot.scripts.lerobot_record import *
from lerobot_robot_ufactory.teleoperators.uf_mock_teleop import UFMockTeleop
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop


@dataclass
class UFRecordConfig(RecordConfig):
    """UFACTORY 기록에만 필요한 에피소드 시작 안전 대기 설정."""

    # Space를 누른 직후 GELLO 입력을 xArm에 연결하기 전 마지막 확인 시간이다.
    start_countdown_s: float = 3.0
    # LeRobotDataset.create()에 실제로 전달할 영상 인코더입니다.
    video_codec: str = "h264"
    # 성공 저장 또는 실패 폐기 판정 직후의 자동 초기 자세 복귀 속도입니다.
    return_to_start_velocity: float = 8.0
from lerobot_robot_ufactory.utils.utils import init_keyboard_listener


def _get_dataset_writer(dataset):
    return getattr(dataset, "writer", None)


def _get_episode_buffer(dataset):
    try:
        return dataset.episode_buffer
    except AttributeError:
        pass

    writer = _get_dataset_writer(dataset)
    if writer is not None and hasattr(writer, "episode_buffer"):
        return writer.episode_buffer
    raise RuntimeError("Unable to access dataset episode buffer for async save.")


def _set_episode_buffer(dataset, episode_buffer):
    updated = False
    writer = _get_dataset_writer(dataset)
    if writer is not None and hasattr(writer, "episode_buffer"):
        writer.episode_buffer = episode_buffer
        updated = True

    try:
        getattr(dataset, "episode_buffer")
    except AttributeError:
        pass
    else:
        try:
            dataset.episode_buffer = episode_buffer
            updated = True
        except AttributeError:
            pass

    if not updated:
        raise RuntimeError("Unable to replace dataset episode buffer for async save.")


def _to_int(value):
    if isinstance(value, (list, tuple)):
        return int(value[0])
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(value[0])


def _episode_buffer_size(episode_buffer):
    return _to_int(episode_buffer.get("size", 0))


def _episode_buffer_index(episode_buffer):
    return _to_int(episode_buffer["episode_index"])


def _current_episode_index(dataset):
    try:
        return _episode_buffer_index(_get_episode_buffer(dataset))
    except Exception:
        return dataset.num_episodes


def _create_empty_episode_buffer(dataset, episode_index, template_episode_buffer):
    writer = _get_dataset_writer(dataset)

    if writer is not None and hasattr(writer, "_create_episode_buffer"):
        episode_buffer = writer._create_episode_buffer()
    elif hasattr(dataset, "create_episode_buffer"):
        episode_buffer = dataset.create_episode_buffer(episode_index=episode_index)
    elif hasattr(dataset, "_create_episode_buffer"):
        episode_buffer = dataset._create_episode_buffer()
    else:
        episode_buffer = copy.deepcopy(template_episode_buffer)
        for key, value in list(episode_buffer.items()):
            if key == "size":
                episode_buffer[key] = 0
            elif key == "episode_index":
                continue
            elif isinstance(value, list):
                episode_buffer[key] = []
            else:
                episode_buffer[key] = []

    if _episode_buffer_index(episode_buffer) != episode_index:
        episode_buffer["episode_index"] = episode_index
    return episode_buffer


def _create_next_episode_buffer(dataset, current_episode_buffer):
    current_episode_index = _episode_buffer_index(current_episode_buffer)
    return _create_empty_episode_buffer(dataset, current_episode_index + 1, current_episode_buffer)


class AsyncEpisodeSaver:
    _STOP = object()

    def __init__(self, dataset):
        self.dataset = dataset
        self._queue = queue.Queue()
        self._total_cnts = 0
        self._finish_cnts = 0
        self._exception = None
        self._thread = threading.Thread(target=self._run, name="uf-async-episode-saver", daemon=True)
        self._thread.start()

    def submit_current_episode(self):
        self._raise_if_failed()
        episode_buffer = _get_episode_buffer(self.dataset)
        if _episode_buffer_size(episode_buffer) == 0:
            raise RuntimeError("Cannot async save an empty episode buffer.")

        episode_index = _episode_buffer_index(episode_buffer)
        next_episode_buffer = _create_next_episode_buffer(self.dataset, episode_buffer)
        _set_episode_buffer(self.dataset, next_episode_buffer)
        self._queue.put((episode_index, episode_buffer))
        return episode_index

    def wait_idle(self):
        self._queue.join()
        self._raise_if_failed()

    def close(self):
        self._queue.join()
        self._queue.put(self._STOP)
        self._queue.join()
        self._thread.join()
        self._raise_if_failed()

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                episode_index, episode_buffer = item
                print(f'[Async] saving episode {episode_index}')
                try:
                    self.dataset.save_episode(episode_data=episode_buffer)
                except TypeError as exc:
                    if "episode_data" in str(exc):
                        raise RuntimeError(
                            "--async-save requires LeRobotDataset.save_episode(episode_data=...)."
                        ) from exc
                    raise
                self._delete_saved_image_dirs(episode_index)
                print(f'[Async] save episode {episode_index} finish')
            except BaseException as exc:
                self._exception = exc
                print(f'[Async] episode {episode_index} save failed, {exc}')
            finally:
                self._queue.task_done()

    def _delete_saved_image_dirs(self, episode_index):
        writer = _get_dataset_writer(self.dataset)
        meta = getattr(writer, "_meta", getattr(self.dataset, "meta", None))
        image_keys = getattr(meta, "image_keys", [])
        image_dir_owner = writer if writer is not None and hasattr(writer, "_get_image_file_dir") else self.dataset
        if not image_keys or not hasattr(image_dir_owner, "_get_image_file_dir"):
            return

        for cam_key in image_keys:
            img_dir = image_dir_owner._get_image_file_dir(episode_index, cam_key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir)

    def _raise_if_failed(self):
        if self._exception is not None:
            raise RuntimeError("Async episode save failed.") from self._exception
    

@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    frame_callback: callable = None,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    last_robot_cmd = robot.get_observation()
    # only positional cmd for now: Remove velo from observation for cmd if needed!
    last_robot_cmd = { k: v for k,v in last_robot_cmd.items() if not "vel" in k }

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)

        elif policy is None and isinstance(teleop, Teleoperator):
            # WebXR uses the current follower pose to hold immediately when
            # its deadman is released or tracking becomes stale.
            if hasattr(teleop, "update_observation"):
                teleop.update_observation(obs)
            act = teleop.get_action()

            # (space mouse) from delta Cartesian cmd to absolute command
            if "pose.dx" in act:
                last_robot_cmd.update({"pose.x": last_robot_cmd["pose.x"] + act["pose.dx"], "pose.y": last_robot_cmd["pose.y"] + act["pose.dy"], "pose.z": last_robot_cmd["pose.z"] + act["pose.dz"]})
                act = last_robot_cmd.copy() # watch out this is shallow copy, not for nested dict

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Applies a pipeline to the action, default is IdentityProcessor
        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            if frame_callback is not None:
                frame = frame_callback(frame)
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


def episode_start_countdown(seconds: float) -> None:
    """각 에피소드에서 사람이 작업영역을 확인할 시간을 준다.

    이 함수가 끝날 때까지 teleop.set_teleop_enabled(True)를 호출하지 않으므로,
    GELLO를 움직여도 xArm에 새 추종 명령이 전달되지 않는다.
    """
    seconds = max(0.0, seconds)
    if seconds == 0:
        return
    print(f"[에피소드 안전 대기] {seconds:.0f}초 뒤 녹화를 시작합니다. 사람·물체·케이블을 확인하세요.")
    end_time = time.monotonic() + seconds
    last_remaining = None
    while True:
        remaining = max(0, int(math.ceil(end_time - time.monotonic())))
        if remaining != last_remaining and remaining > 0:
            print(f"[에피소드 안전 대기] {remaining}초...")
            last_remaining = remaining
        if remaining == 0:
            break
        time.sleep(0.05)
    print("[에피소드 시작] 이제 GELLO 추종과 데이터 기록을 시작합니다.")


def print_record_controls(recording: bool = False) -> None:
    """초보자도 성공 저장과 실패 폐기를 혼동하지 않도록 항상 같은 안내를 보인다."""
    if recording:
        print("⌨   [ESC] 전체 종료  [→] 성공 저장  [←] 실패 폐기")
    else:
        print("⌨   [ESC] 전체 종료  [Space] 녹화 시작  [→] 성공 저장  [←] 실패 폐기")


def wait_for_episode_decision(events: dict, is_evt: bool) -> None:
    """시간 만료 뒤에도 자동 저장하지 않고 성공/실패 판정을 기다린다."""
    print("\n[판정 대기] 이번 시도 녹화가 끝났습니다. 아직 저장하지 않았습니다.")
    print("           → : 성공으로 저장 / ← : 실패로 폐기 / ESC : 전체 수집 종료")
    if is_evt:
        while not (
            events["save_episode"]
            or events["rerecord_episode"]
            or events["stop_recording"]
        ):
            time.sleep(0.05)
        return

    # 화면/키보드 이벤트가 없는 환경에서도 자동 저장은 금지한다.
    while True:
        answer = input("성공이면 'save', 실패면 'discard', 종료면 'exit'를 입력하세요: ").strip().lower()
        if answer == "save":
            events["save_episode"] = True
            return
        if answer == "discard":
            events["rerecord_episode"] = True
            return
        if answer == "exit":
            events["stop_recording"] = True
            return
        print("[안내] save / discard / exit 중 하나를 입력하세요.")


def return_to_start_after_decision(robot, cfg: UFRecordConfig, decision: str) -> None:
    """성공 저장과 실패 폐기 모두 판정 직후 같은 시작 자세로 복귀한다."""
    print(f"[자동 복귀] {decision} 판정이 끝났습니다. 기본 그리퍼를 열고 초기 자세로 돌아갑니다.")
    robot.configure(alignment_velocity_deg_s=cfg.return_to_start_velocity)
    print(
        f"[자동 복귀 완료] {cfg.return_to_start_velocity:.1f}°/s로 초기 자세 복귀를 완료했습니다. "
        "다음 물체를 배치한 뒤 Space를 누르세요."
    )


def record(cfg: UFRecordConfig, async_save: bool = False) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.video_codec,
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    is_evt = not is_headless()
    is_uf_teleop = isinstance(teleop, UFBaseTeleop)
    is_recorded = False
    key_dict = {}
    # save_episode와 rerecord_episode는 각각 → 성공 저장, ← 실패 폐기를 뜻한다.
    # 시간 제한에 도달했다고 자동 저장하지 않도록 두 판정 상태를 명시적으로 분리한다.
    events = {
        "exit_early": False,
        "save_episode": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    if is_evt:
        from pynput import keyboard

        key_dict = {
            keyboard.Key.space: 0,  # start
            keyboard.Key.enter: 0,  # help
        }

        def on_press(key):
            try:
                if key == keyboard.Key.right:
                    print("[입력] → 성공 저장을 선택했습니다.")
                    events["save_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.left:
                    print("[입력] ← 실패 폐기를 선택했습니다.")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping data recording...")
                    events["stop_recording"] = True
                    events["exit_early"] = True
            except Exception as e:
                print(f"Error handling key press: {e}")
            if key in key_dict:
                key_dict[key] = True

        def on_release(key):
            try:
                if key == keyboard.Key.enter:
                    if not is_recorded:
                        print_record_controls(recording=False)
                    else:
                        print_record_controls(recording=True)
                    # is_recorded = True
            except Exception as e:
                print(f"Error handling key release: {e}")
            if key in key_dict:
                key_dict[key] = False

        listener, events = init_keyboard_listener(events=events, on_press=on_press, on_release=on_release)
        print("\n********** Episode Record Loop Start **********")
        print_record_controls(recording=False)
    else:
        input('⌨   Press Enter to start record >>> ')
        # 실제 상대 영점 설정은 아래 공통 에피소드 시작 블록에서 팔로워
        # 관측값과 안전 대기를 포함해 한 번만 수행합니다.
        is_recorded = True
        print('\n********** Episode Record Loop Start **********')

    frame_callback = None
    async_episode_saver = AsyncEpisodeSaver(dataset) if async_save else None
    if async_episode_saver is not None:
        print('Async episode saving is enabled.')

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            time.sleep(0.01)
            if is_evt:
                if not is_recorded and key_dict[keyboard.Key.space]:
                    is_recorded = True

            if teleop is not None and isinstance(teleop, UFMockTeleop):
                if events["stop_recording"]:
                    continue
                teleop.configure(events=events)
                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    input('\n⌨   Press Enter to regenerate random target location >>>>> ')
                    continue
                if events["stop_recording"]:
                    continue
                is_recorded = True

            if is_recorded:
                events["rerecord_episode"] = False
                events["save_episode"] = False
                events["exit_early"] = False
                if is_uf_teleop:
                    # 프로그램 시작/Space 직후의 자세 확인은 기존 안전 속도 3°/s입니다.
                    # 저장 후 자동 복귀 속도 8°/s와 혼동하지 않습니다.
                    try:
                        robot.configure()
                    except Exception as exc:
                        # C31/C19 같은 컨트롤러 오류로 준비에 실패해도 프로그램을
                        # 죽이지 않고, 오류 코드를 안내한 뒤 Space 대기로 돌아간다.
                        is_recorded = False
                        print(f"❌ [에피소드 시작 취소] 로봇 준비 실패: {exc}")
                        print("   xArm 오류(C번호)가 로그에 보이면 해당 복구 절차(C31/C19)를 먼저 실행하세요.")
                        if is_evt:
                            print_record_controls(recording=False)
                        continue
                    episode_start_countdown(cfg.start_countdown_s)
                    obs = robot.get_observation()
                    try:
                        teleop.set_teleop_enabled(True, obs)
                    except RuntimeError as exc:
                        # 리더가 흔들렸거나 첫 action 차이가 크면 어떤 텔레옵 명령도
                        # 보내지 않고 다시 Space 대기 상태로 돌아갑니다.
                        teleop.set_teleop_enabled(False)
                        is_recorded = False
                        print(f"❌ [에피소드 시작 취소] {exc}")
                        print("   GELLO를 편한 자세로 잡고 고정한 뒤 Space를 다시 누르세요.")
                        if is_evt:
                            print_record_controls(recording=False)
                        continue
                # 준비/카운트다운 중에 미리 눌린 →/←는 판정 오동작을 일으키므로
                # 녹화 시작 직전에 버린다. 판정 키는 녹화 시작 후부터 유효하다.
                events["rerecord_episode"] = False
                events["save_episode"] = False
                events["exit_early"] = False
                log_say(f"Recording episode {_current_episode_index(dataset)}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    frame_callback=frame_callback,
                )
            else:
                continue
            if events['stop_recording']:
                break

            # 20초가 끝났거나 →/←로 녹화를 멈춘 시점이다. 이 시점에는 절대
            # 자동 저장하지 않는다. → 성공 또는 ← 실패의 명시적인 판정을 기다린다.
            if not events["save_episode"] and not events["rerecord_episode"]:
                wait_for_episode_decision(events, is_evt)
            if events["stop_recording"]:
                break

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["save_episode"] = False
                events["exit_early"] = False
                if is_uf_teleop:
                    teleop.set_teleop_enabled(False)
                episode_buffer = _get_episode_buffer(dataset)
                frame_count = _episode_buffer_size(episode_buffer)
                episode_index = _episode_buffer_index(episode_buffer)
                duration_s = frame_count / cfg.dataset.fps
                if frame_count > 0:
                    if async_episode_saver is None:
                        dataset.clear_episode_buffer()
                    else:
                        empty_episode_buffer = _create_empty_episode_buffer(dataset, episode_index, episode_buffer)
                        _set_episode_buffer(dataset, empty_episode_buffer)
                print("❌ [실패 에피소드 폐기 완료]")
                print(f"   에피소드: {episode_index} / 기록 프레임: {frame_count}개 / 기록 시간: 약 {duration_s:.1f}초")
                print("   결과: 관절값·action·영상은 학습 데이터셋에 저장하지 않았습니다.")
                if is_uf_teleop:
                    return_to_start_after_decision(robot, cfg, "실패 폐기")
                is_recorded = False
                if is_evt:
                    print_record_controls(recording=False)
                else:
                    input('\n⌨   Press Enter to rerecord this episode >>>>> ')
                    is_recorded = True
                continue

            if events["save_episode"] and is_recorded and not events['stop_recording']:
                # 프레임이 없는 에피소드를 저장하려 하면 LeRobot이 예외로 죽는다.
                # 준비 단계 취소 직후 등으로 버퍼가 비어 있으면 저장을 건너뛴다.
                empty_buffer = _episode_buffer_size(_get_episode_buffer(dataset)) == 0
                if empty_buffer:
                    print("⚠️ [저장 건너뜀] 기록된 프레임이 없어 이번 판정을 무시합니다. Space로 다시 시작하세요.")
                    events["save_episode"] = False
                    is_recorded = False
                    if is_uf_teleop:
                        teleop.set_teleop_enabled(False)
                    if is_evt:
                        print_record_controls(recording=False)
                    continue
                episode_index = _current_episode_index(dataset)
                log_say(f"Save episode {episode_index}", cfg.play_sounds)
                if is_uf_teleop:
                    teleop.set_teleop_enabled(False)
                if async_episode_saver is None:
                    dataset.save_episode()
                    log_say(f"[Finish] Save episode {episode_index}", cfg.play_sounds)
                else:
                    queued_episode_index = async_episode_saver.submit_current_episode()
                    if queued_episode_index is not None:
                        log_say(f"[Queued] Save episode {queued_episode_index}", cfg.play_sounds)

                recorded_episodes += 1
                # 성공 저장 또는 실패 폐기 판정 직후에만 초기 자세로 자동 복귀한다.
                # configure()는 기본 그리퍼를 열고, 현재 자세가 시작 자세와 다를 때만
                # 설정된 8°/s로 관절 이동을 수행한다.
                # ESC로 전체 수집을 끝낸 경우에는 이 블록에 도달하지 않으므로
                # 사용자가 원하지 않는 종료 직전의 추가 모션은 만들지 않는다.
                if is_uf_teleop:
                    return_to_start_after_decision(robot, cfg, "성공 저장")
                is_recorded = False
                events["save_episode"] = False
                if is_evt:
                    print_record_controls(recording=False)
                else:
                    input('⌨   Press Enter to record at the next episode >>>>> ')
                    is_recorded = True

        if async_episode_saver is not None:
            print('Waiting for pending async episode saves.')
            async_episode_saver.close()

    print("\n********** Episode Record Loop Exit **********")

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if is_evt and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset

@parser.wrap()
def get_cfg(cfg: UFRecordConfig) -> UFRecordConfig:
    return cfg

def main():
    parser = argparse.ArgumentParser(description='configuration args')
    parser.add_argument('-r',
                       action='store_true', # specify --resume if resume needs to be True
                       default=False,
                       help='Whether contitue recording on existing dataset (default: False)')
    parser.add_argument('-a', '--async_save',
                       action='store_true',
                       default=False,
                       help='Enable async background saving (default: False)')
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    if args.r:
        cfg.resume = True
    cfg.play_sounds = False
    record(cfg, async_save=args.async_save)


if __name__ == "__main__":
    main()
