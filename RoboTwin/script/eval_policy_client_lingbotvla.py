import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

# Candidate instructions generated per episode, fixed at the benchmark's 100 so
# the instruction distribution does not depend on --test_num (episodes per task).
INSTRUCTION_POOL_SIZE = 100


def build_formatted_observation(observation, instruction, ee_mode):
    """Shape one RoboTwin observation into the dict the deploy server expects.

    Module-level so it can be tested against the real server without a simulator
    or a checkpoint -- the EE/joint choice is exactly the part that is silent
    when wrong, so it should not only be reachable from inside the eval loop.

    Joint mode sends the 14-dim `joint_action.vector` and nothing else.

    EE mode sends the four WXYZ endpose fields. The deploy server's
    `_build_ee_state` reads them as TOP-LEVEL observation keys (it does not look
    inside `observation.state`), so they are spread at the top level; the same
    dict is also placed under `observation.state` so a server that passes that
    key through unchanged still sees a complete state.
    """
    if ee_mode:
        endpose = dict(observation["endpose"])
        if endpose.get("left_endpose") is None:
            raise KeyError(
                "action_space='ee' needs the endpose block in the observation. "
                "Enable `data_type.endpose: true` in the RoboTwin task config "
                f"(got keys: {sorted(observation.keys())})."
            )
        state = endpose
    else:
        endpose = None
        state = observation["joint_action"]["vector"]

    formatted_observation = {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],  # H,W,3
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],

        "observation.state": state,
        "task": instruction,
    }
    if ee_mode:
        # The keys _build_ee_state() actually reads: 7-dim poses + scalar
        # grippers, WXYZ, verbatim from RoboTwin.
        formatted_observation.update(endpose)
    return formatted_observation


def resolve_eval_action_space(usr_args, server_metadata=None):
    """(ee_mode, action_type) for this run, preferring the SERVER's own declaration.

    The representation is a property of the checkpoint, and the deploy server
    resolves it from the checkpoint's lingbotvla_cli.yaml and publishes it in the
    websocket handshake. Reading it from there -- rather than from this client's
    `--action_space` flag -- is what makes the two sides agree by construction: a
    flag that disagrees with the checkpoint is otherwise accepted silently, and in
    the dangerous direction (`--action_space ee` against a joint checkpoint) the
    client would send endpose fields and call `take_action(action_type='ee')` with
    joint targets, failing every episode in a way that looks like a policy problem.

    Falls back to the flag only when the server published nothing, which is the
    case for servers predating the handshake. Absent/empty still means the joint
    path, so an omitted flag reproduces the original joint evaluation exactly.

    Returning the disagreement as a warning rather than a hard error is deliberate:
    the flag is the operator's explicit statement of intent, so the server's value
    wins on substance, but a mismatch is worth seeing in the log.
    """
    metadata = server_metadata or {}
    declared = str(metadata.get("action_space", "") or "").strip().lower()

    flag = str((usr_args or {}).get("action_space", "") or "").strip().lower()

    if declared in ("joint", "ee"):
        if flag and flag != declared:
            print(
                f"\033[33mWARNING: --action_space={flag!r} disagrees with the "
                f"checkpoint, which was trained with action_space={declared!r}. "
                f"Using the checkpoint's value.\033[0m"
            )
        elif not flag:
            # No flag to confirm it, so say where the value came from -- this is
            # what makes an EE eval run explicable from the log alone.
            print(
                f"\033[36maction_space={declared!r} read from the server handshake "
                f"(checkpoint's lingbotvla_cli.yaml)\033[0m"
            )
        action_space = declared
    else:
        action_space = flag
        if declared or metadata:
            print(
                "\033[33mWARNING: the server published no action_space; falling back "
                "to the --action_space flag. Update the deploy server if this "
                "checkpoint is ee.\033[0m"
            )

    ee_mode = action_space == "ee"
    return ee_mode, ("ee" if ee_mode else "qpos")


def build_eval_policy(usr_args, save_dir):
    """The policy object for this run.

    Without `--sam_mask` (the default) this is the plain WebsocketClientPolicy,
    exactly as before. With `--sam_mask True` cam_high goes through the
    mask-Qwen (Qwen3.8-27B) -> SAM3 wrapper first; see script/deploy/sam_mask/.
    """
    if str(usr_args.get("sam_mask", False)).strip().lower() in ("1", "true", "yes", "on"):
        from script.deploy.sam_mask import build_policy_client
        return build_policy_client(usr_args, save_dir=save_dir)
    from script.deploy.websocket_client_policy import WebsocketClientPolicy
    return WebsocketClientPolicy(port=usr_args['port'])


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None
    video_fps = str(usr_args.get("video_fps", 10))

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if usr_args.get("output_dir"):
        save_dir = Path(usr_args["output_dir"]) / task_name
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 命令行 --eval_video_log 优先于 YAML 配置
    if "eval_video_log" in usr_args:
        args["eval_video_log"] = usr_args["eval_video_log"]

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))  # episodes per task; 100 = the benchmark
    topk = 1

    # model = get_model(usr_args)
    # from IPython import embed;embed()
    model = build_eval_policy(usr_args, save_dir)

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   video_fps=video_fps,
                                   instruction_type=instruction_type,
                                   usr_args=usr_args)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


# Exit code for an unrecoverable scene-setup failure; the launcher retries any
# non-zero exit, so the value only has to be distinctive in its "[fail] exit=" line.
FATAL_SETUP_EXIT_CODE = 75


def is_fatal_setup_error(e):
    """True for scene-setup errors that no later seed can recover from.

    When curobo's Warp kernels fail to JIT-compile (an NVRTC precompiled-header
    race between sim processes sharing WARP_CACHE_PATH), Robot.set_planner()
    raises before assigning left_planner. The Robot object is cached on the task,
    so every later setup_demo goes through Robot.reset(), which reads
    self.left_planner first and raises AttributeError -- for every seed, forever.
    The seed loop below would then spin until the launcher's stall watchdog
    (default 3600 s) kills it. A fresh process recompiles and recovers.
    """
    msg = str(e)
    if "CUDA kernel build failed" in msg:
        return True
    return isinstance(e, AttributeError) and "_planner" in msg


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                video_fps="10",
                instruction_type=None,
                usr_args = None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    # eval_func = eval_function_decorator(policy_name, "eval")
    # reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                print(f"error occurs ! {e}")
                if is_fatal_setup_error(e):
                    print(f"\033[31mfatal scene-setup error at seed {now_seed} "
                          f"({succ_seed}/{test_num} episodes done); exiting with code "
                          f"{FATAL_SETUP_EXIT_CODE} so the launcher retries in a fresh process\033[0m",
                          flush=True)
                    sys.stderr.flush()
                    # os._exit, not sys.exit: a half-initialised sapien/curobo process
                    # can hang in interpreter teardown, which would hold the slot again.
                    os._exit(FATAL_SETUP_EXIT_CODE)
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        # The third argument caps how many candidate instructions are generated, not
        # how many episodes run. Upstream passes test_num (=100) here; tying it to a
        # smaller test_num would shrink the pool np.random.choice draws from, so
        # instructions would no longer match a 100-episode benchmark run.
        results = generate_episode_descriptions(args["task_name"], episode_info_list, INSTRUCTION_POOL_SIZE)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    os.environ["FFMPEG_BINARY"],
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    video_fps,
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        path_to_pi_model = None
        if usr_args is not None and "new_ckpt_path" in usr_args:
            path_to_pi_model = usr_args['new_ckpt_path']
        ret = model.infer(dict(reset = True, robo_name=usr_args['robo_name'], path_to_pi_model=path_to_pi_model))
        
        # `action_space: ee` runs need two things the joint runs do not: the
        # 16-dim end-effector state instead of the 14-dim joint vector, and
        # `take_action(..., action_type='ee')` so the target pose goes through
        # IK/motion planning rather than joint TO PP.
        #
        # Which one applies is the checkpoint's decision, not this script's: the
        # server resolves it from the run's lingbotvla_cli.yaml and published it in
        # the websocket handshake above. `--action_space` is only a fallback for
        # servers old enough not to publish it, and a mismatch is logged.
        server_metadata = None
        if hasattr(model, "get_server_metadata"):
            try:
                server_metadata = model.get_server_metadata()
            except Exception as e:
                print(f"\033[33mCould not read server metadata ({e}); using --action_space\033[0m")
        ee_mode, eval_action_type = resolve_eval_action_space(usr_args, server_metadata)
        if ee_mode:
            print("\033[94mEval action_space: ee (endpose state, take_action(action_type='ee'))\033[0m")
        # sam_frames=all only: after every executed step, render cam_high once
        # more and feed it to the SAM3 tracker. Rendering does not touch physics.
        feed_every_step = getattr(model, "wants_every_step", False)
        # Eval video (launcher --enable_video): take_action() records TASK_ENV.now_obs,
        # and only get_obs() refreshes that cache.
        #   1 -> no refresh inside a chunk: each step repeats the chunk's VLA query
        #        frame, so the video holds still for a chunk, then jumps. No extra cost.
        #   2 -> refresh after every step: a continuous video, at one extra
        #        all-camera render per step.
        refresh_video = (TASK_ENV.eval_video_path is not None and str(
            (usr_args or {}).get("video_every_step", False)).strip().lower() in ("1", "true", "yes", "on"))

        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim and not succ:
            observation = TASK_ENV.get_obs()
            # from IPython import embed;embed()

            formatted_observation = build_formatted_observation(
                observation, TASK_ENV.get_instruction(), ee_mode)

            # print(formatted_observation["observation.images.cam_high"].shape)
            # ========= debug image =========
            # import torch, torchvision
            # from PIL import Image
            # imgs_np = []
            # for k in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
            #     arr = formatted_observation['image'][k].squeeze(0)  # (H,W,3), uint8
            #     print(arr.max(), arr.min())
            #     imgs_np.append(arr)

            # concat = np.concatenate(imgs_np, axis=1)
            # if TASK_ENV.take_action_cnt % 25 ==0:
            #     Image.fromarray(concat).save(f'combined_{TASK_ENV.take_action_cnt}.png')
            # ========= debug image =========

            # from IPython import embed;embed()
            ret = model.infer(formatted_observation) #(TASK_ENV, model, observation)
            action, latency = ret['action'], ret['server_timing']
            if len(action.shape) == 2:
                for act_idx, act in enumerate(action):
                    TASK_ENV.take_action(act, action_type=eval_action_type)
                    if TASK_ENV.eval_success:
                        succ = True
                        break
                    # The frame after the chunk's last action is the next VLA query
                    # frame: the outer loop's get_obs() renders it (and infer() feeds
                    # it to SAM3), so skip it here. One render serves both consumers.
                    if (act_idx < len(action) - 1
                            and TASK_ENV.take_action_cnt < TASK_ENV.step_lim
                            and (refresh_video or feed_every_step)):
                        step_obs = TASK_ENV.get_obs()  # also refreshes now_obs for the video
                        if feed_every_step:
                            model.observe_step(step_obs["observation"]["head_camera"]["rgb"])
            else:
                TASK_ENV.take_action(action, action_type=eval_action_type)
                if TASK_ENV.eval_success:
                    succ = True
            
            print(f"infer time {latency}")


        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        result_tag = "success" if succ else "failure"
        if hasattr(model, "end_episode"):  # sam_mask: close SAM3 session, save masked cam_high
            model.end_episode(TASK_ENV.test_num, succ)

        old_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"
        new_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_{result_tag}.mp4"

        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"Video saved: {new_name}")


        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
