#!/usr/bin/env python3
"""Resolve and validate the portable SAM evaluation configuration."""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, resolve, runtime_env, writable


def owned(value):
    value = resolve(value)
    if not value.is_relative_to(base_path()):
        raise ValueError(f"Project code paths must remain inside the submission: {value}")
    return value


def settings():
    cfg = config()
    values = {key.upper(): str(value) for key, value in cfg["evaluation"].items()}
    paths = {"MODEL_PATH": "vla_checkpoint", "INFERENCE_WORKDIR": "vla_project",
             "EVAL_WORKDIR": "robotwin", "OUTPUT_BASE": "eval_output",
             "QWEN3VL_PATH": "vla_backbone", "VLA_NORM_STATS": "vla_norm_stats",
             "MASK_QWEN_MODEL": "qwen_model", "SAM_CHECKPOINT": "sam_checkpoint"}
    for key, name in paths.items():
        # Model inputs may be read from another checkout. Code, simulation
        # assets that the launcher updates, and generated files stay local.
        resolver = owned if key in ("INFERENCE_WORKDIR", "EVAL_WORKDIR") else resolve
        if key == "OUTPUT_BASE":
            resolver = writable
        values[key] = str(resolver(os.environ.get(key, path(name))))
    model = Path(values["MODEL_PATH"])
    if (model / "hf_ckpt").is_dir():
        values["MODEL_PATH"] = str(model / "hf_ckpt")
    info = json.loads(path("checkpoint_sam_info").read_text())
    values.update(SAM_JUDGE=str(int(bool(info.get("judge")))),
                  OUTSIDE_SCALE=str(info["style"]["outside_scale"]),
                  EDGE_WIDTH=str(info["style"]["edge_width"]))
    values["MASK_QWEN_MODEL_NAME"] = cfg["qwen"]["model_name"]
    values["MASK_QWEN_MAX_MODEL_LEN"] = str(cfg["qwen"]["max_model_len"])
    values["MASK_QWEN_MAX_NUM_SEQS"] = str(cfg["qwen"]["max_num_seqs"])
    values["SAM_SERVER"] = str(owned(path("sam_source") / "tools/sam3_stream_server.py"))
    for key in list(values):
        if key not in paths:
            values[key] = os.environ.get(key, values[key])
    values["SAM_SERVER"] = str(owned(values["SAM_SERVER"]))
    for key in ("RUN_DIR", "WARP_CACHE_PATH"):
        if os.environ.get(key):
            values[key] = str(writable(os.environ[key]))
    for key in ("RUN_ID", "JOB_TOKEN", "TASK_CONFIG"):
        value = os.environ.get(key, values.get(key, ""))
        if value and (not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", "..")):
            raise ValueError(f"Invalid {key}: {value}")
    if os.environ.get("TASK_NAMES"):
        names = os.environ["TASK_NAMES"].split(",")
        if any(not re.fullmatch(r"[a-z0-9_]+", name) for name in names) or len(names) != len(set(names)):
            raise ValueError("TASK_NAMES must contain distinct, comma-separated task names")
        values["NUM_TASKS"] = str(len(names))
    writable(values["OUTPUT_BASE"])
    values["SUBMIT_ROOT"] = str(base_path())
    values["ROLE_RUNNER"] = str(base_path() / "tools/run_python.py")
    # Do not import user-level numpy or allow Hub downloads during evaluation.
    env = runtime_env("eval-launch", create=False)
    for key in env:
        if key == "WARP_CACHE_PATH" and key not in os.environ:
            continue  # script_sam.sh adds the model/config hash under this run's cache
        if key not in os.environ or env[key] != os.environ[key]:
            values.setdefault(key, env[key])
    values["PYTHONNOUSERSITE"] = "1"
    values["HF_HUB_OFFLINE"] = values["TRANSFORMERS_OFFLINE"] = "1"
    return values


def check(values):
    for role in ("vla", "robotwin", "sam3", "vllm"):
        if not os.access(interpreter(role), os.X_OK):
            raise FileNotFoundError(f"base.json interpreters.{role}: {interpreter(role)}")
    for name in ("vla_project", "robotwin", "vllm_source", "sam_source"):
        if not owned(path(name)).is_dir():
            raise FileNotFoundError(path(name))
    ckpt = Path(values["MODEL_PATH"])
    required = [ckpt / "config.json", ckpt / "model.safetensors.index.json",
                ckpt.parents[2] / "lingbotvla_cli.yaml", Path(values["VLA_NORM_STATS"]),
                Path(values["QWEN3VL_PATH"]) / "config.json", Path(values["SAM_CHECKPOINT"])]
    for model in (ckpt, Path(values["MASK_QWEN_MODEL"])):
        index = json.loads((model / "model.safetensors.index.json").read_text())
        required.extend(model / filename for filename in set(index["weight_map"].values()))
    for file in required:
        if not file.is_file() or not file.stat().st_size:
            raise FileNotFoundError(file)
    for key in ("NUM_TASKS", "TEST_NUM", "NUM_PER_GPU", "USE_LENGTH"):
        if int(values[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(values["NUM_TASKS"]) > 50:
        raise ValueError("NUM_TASKS must be <= 50")
    if int(values["USE_LENGTH"]) > json.loads((ckpt / "config.json").read_text())["chunk_size"]:
        raise ValueError("USE_LENGTH exceeds checkpoint chunk_size")
    if not (Path(values["EVAL_WORKDIR"]) / "task_config" / (values["TASK_CONFIG"] + ".yml")).is_file():
        raise ValueError("Unknown TASK_CONFIG")
    if len(os.fsencode(runtime_env("eval-vllm", create=False)["VLLM_RPC_BASE_PATH"])) + 37 > 107:
        raise ValueError("Bundle path is too long for vLLM Unix sockets; move the bundle to a shorter path")
    save_mask_video = os.environ.get("SAM_SAVE_VIDEO", values.get("SAM_SAVE_VIDEO", "")).lower()
    if int(values["ENABLE_VIDEO"]) != 0 or save_mask_video in ("1", "true"):
        check_video_encoder()


def check_video_encoder():
    """Check the actual H.264 encoder before allocating any GPU services."""
    binary = owned(path("ffmpeg"))
    if not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"base.json paths.ffmpeg is not executable: {binary}")
    result = subprocess.run(
        [str(binary), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=size=32x32:rate=1", "-frames:v", "1", "-threads", "1", "-c:v", "libx264", "-f", "null", "-"],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode:
        raise RuntimeError(f"Configured FFmpeg cannot encode H.264: {result.stderr.strip()}")


def check_ports(vla, sam, qwen, slots, masking):
    ports = list(range(vla, vla + slots))
    if masking:
        ports += list(range(sam, sam + slots)) + [qwen]
    if not ports or min(ports) < 1 or max(ports) > 65535:
        raise ValueError("Service ports must be in 1..65535")
    if len(ports) != len(set(ports)):
        raise ValueError("Configured VLA/SAM/Qwen service ports overlap")
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise ValueError(f"Service port {port} is occupied; choose free ports in base.json") from exc
    finally:
        for sock in sockets:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-ports", nargs=5, type=int,
                        metavar=("VLA", "SAM", "QWEN", "SLOTS", "MASKING"))
    args = parser.parse_args()
    if args.check_ports:
        check_ports(*args.check_ports)
        print("Evaluation service ports: OK")
        return
    values = settings()
    if args.check or args.shell:
        check(values)
    if args.shell:
        runtime_env("eval-launch")
        for key, value in values.items():
            print(f"export {key}={shlex.quote(value)}")
    else:
        print(json.dumps({"base_path": str(base_path()), "evaluation": values,
                          "interpreters": {r: str(interpreter(r)) for r in ("vla", "robotwin", "sam3", "vllm")}}, indent=2))
        if args.check:
            print("Evaluation files and settings: OK")


if __name__ == "__main__":
    main()
