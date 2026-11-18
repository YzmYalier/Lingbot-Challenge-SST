#!/usr/bin/env python3
"""Launch the checked-out Lingbot training recipe from the submission bundle."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, resolve, writable


def owned(value):
    result = resolve(value)
    if not result.is_relative_to(base_path()):
        raise ValueError(f"Training resources must remain inside the submission: {result}")
    return result


def environment():
    from run_python import environment as role_environment
    env = role_environment("vla", runtime_name="train-vla", inherit_warp_cache=False)
    env.update(HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    # Never read credentials from a checked-in script or an unrelated user HOME.
    env.setdefault("WANDB_MODE", "offline")
    for key, suffix in (("HF_DATASETS_CACHE", "datasets"), ("HF_HUB_CACHE", "hub"),
                        ("HUGGINGFACE_HUB_CACHE", "hub")):
        env[key] = str(Path(env["HF_HOME"]) / suffix)
    env.pop("TRANSFORMERS_CACHE", None)
    runtime = writable(path("runtime") / "training")
    for key, name in (("WANDB_DIR", "wandb"), ("WANDB_CACHE_DIR", "wandb-cache"),
                      ("WANDB_CONFIG_DIR", "wandb-config"), ("WANDB_DATA_DIR", "wandb-data"),
                      ("HF_LEROBOT_HOME", "lerobot")):
        target = runtime / name
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    return env


def resolve_config(value):
    """YAML resource references name keys in base.json.paths."""
    if isinstance(value, dict):
        return {k: resolve_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_config(v) for v in value]
    if isinstance(value, str) and value.startswith("submission://"):
        name, _, suffix = value[len("submission://"):].partition("/")
        return str(owned(path(name) / suffix))
    return value


def schema_fields(project):
    """Validate option names without importing CUDA/model libraries."""
    classes = {}
    for file in (project / "lingbotvla/utils/arguments.py", project / "tasks/vla/train_lingbotvla.py"):
        for node in ast.parse(file.read_text()).body:
            if isinstance(node, ast.ClassDef):
                inherited = set()
                for parent in node.bases:
                    if isinstance(parent, ast.Name):
                        inherited |= classes.get(parent.id, set())
                classes[node.name] = inherited | {n.target.id for n in node.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    return {section: classes[name] for section, name in (
        ("model", "ModelArguments"), ("data", "MyDataArguments"),
        ("train", "MyTrainingArguments"), ("eval", "EvalArguments"))}


def apply_overrides(cfg, arguments, fields):
    import yaml
    cfg = copy.deepcopy(cfg)
    i = 0
    while i < len(arguments):
        token = arguments[i]
        if not token.startswith("--") or "." not in token:
            raise ValueError(f"Expected --section.option VALUE, got {token!r}")
        option, equals, inline = token[2:].partition("=")
        section, key = option.split(".", 1)
        if section not in fields or key not in fields[section]:
            raise ValueError(f"Unknown training option for this branch: --{option}")
        i += 1
        values = [inline] if equals else []
        if not equals:
            while i < len(arguments) and not arguments[i].startswith("--"):
                values.append(arguments[i])
                i += 1
        if not values:
            value = True
        elif len(values) == 1:
            value = yaml.safe_load(values[0])
        else:
            value = [yaml.safe_load(v) for v in values]
        cfg.setdefault(section, {})[key] = value
    for section, values in cfg.items():
        if section not in fields or not isinstance(values, dict):
            raise ValueError(f"Unknown config section: {section}")
        unknown = set(values) - fields[section]
        if unknown:
            raise ValueError(f"Unsupported {section} options on this branch: {sorted(unknown)}")
    return cfg


def topology(train, env):
    nodes = int(env.get("NNODES", env.get("HOST_NUM", "1")))
    rank = int(env.get("NODE_RANK", "0"))
    if env.get("NPROC_PER_NODE"):
        gpus = int(env["NPROC_PER_NODE"])
    elif "CUDA_VISIBLE_DEVICES" in env:
        devices = [d.strip() for d in env["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]
        gpus = len(devices) if devices != ["-1"] else 0
    elif env.get("HOST_GPU_NUM"):
        gpus = int(env["HOST_GPU_NUM"])
    else:
        try:
            gpus = len(subprocess.check_output(["nvidia-smi", "-L"], text=True,
                                               stderr=subprocess.DEVNULL).splitlines())
        except (OSError, subprocess.CalledProcessError):
            gpus = 0
    if nodes < 1 or gpus < 1 or not 0 <= rank < nodes:
        raise ValueError("Invalid node/GPU topology. For a CPU config check, set NPROC_PER_NODE=8.")
    world = nodes * gpus
    micro, batch = int(train["micro_batch_size"]), int(train["global_batch_size"])
    if micro < 1 or batch < 1 or batch % (micro * world):
        raise ValueError(f"global_batch_size={batch} must be divisible by micro_batch_size={micro} * world={world}")
    if any(train.get(key, 1) != 1 for key in ("tensor_parallel_size", "pipeline_parallel_size",
                                            "context_parallel_size", "ulysses_parallel_size")):
        raise ValueError("The submission launcher supports data-parallel training only")
    shard = int(train.get("data_parallel_shard_size", world))
    if shard < 1:
        shard = world
    if world % shard:
        raise ValueError(f"world={world} must be a multiple of data_parallel_shard_size={shard}")
    addr = env.get("MASTER_ADDR", "127.0.0.1" if nodes == 1 else "")
    if not addr:
        raise ValueError("Multi-node training requires MASTER_ADDR on every node")
    port = int(env.get("MASTER_PORT", "62500"))
    if not 1 <= port <= 65535:
        raise ValueError("MASTER_PORT must be in 1..65535")
    train.update(gradient_accumulation_steps=batch // (micro * world),
                 data_parallel_shard_size=shard, data_parallel_replicate_size=world // shard)
    return dict(nnodes=nodes, nproc_per_node=gpus, node_rank=rank,
                master_addr=addr, master_port=port, world_size=world)


def check_paths(cfg):
    """Resolve all path-valued options, including command-line overrides."""
    def walk(values):
        for key, value in values.items():
            if isinstance(value, dict):
                walk(value)
            elif value and key.endswith(("_path", "_dir", "_root", "_file")):
                if not isinstance(value, str):
                    raise ValueError(f"Expected one local path for {key}")
                values[key] = str(owned(value))
    walk(cfg)
    output = writable(cfg["train"]["output_dir"])
    protected = [owned(cfg["model"]["model_path"]), owned(cfg["data"]["train_path"]),
                 owned(path("raw")), owned(path("vla_backbone")), owned(path("vla_checkpoint"))]
    for resource in protected:
        if output.is_relative_to(resource) or resource.is_relative_to(output):
            raise ValueError(f"Training output overlaps an input or evaluation checkpoint: {output}")


def make_plan(yaml_file=None, overrides=(), env=None):
    import yaml
    project = owned(path("vla_project"))
    profile = json.loads((project / "training_profile.json").read_text())
    selected = yaml_file or profile["config"]
    candidate = Path(selected)
    if not candidate.is_absolute():
        candidate = project / candidate if (project / candidate).is_file() else resolve(candidate)
    candidate = owned(candidate)
    cfg = apply_overrides(yaml.safe_load(candidate.read_text()), overrides, schema_fields(project))
    cfg = resolve_config(cfg)
    check_paths(cfg)
    topo = topology(cfg["train"], os.environ if env is None else env)
    return {"profile": profile["name"], "yaml": str(candidate), "project": str(project),
            "python": str(interpreter("vla")), "topology": topo, "config": cfg}


def require_file(file):
    file = owned(file)
    if not file.is_file() or not file.stat().st_size:
        raise FileNotFoundError(f"Missing training resource: {file}")
    return file


def check_assets(plan, *, data=True):
    cfg = plan["config"]
    if not os.access(interpreter("vla"), os.X_OK):
        raise FileNotFoundError(f"base.json interpreters.vla: {interpreter('vla')}")
    model = Path(cfg["model"]["model_path"])
    require_file(model / "config.json")
    index = json.loads(require_file(model / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        require_file(model / shard)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        require_file(Path(cfg["model"]["tokenizer_path"]) / name)
    require_file(cfg["data"]["norm_stats_file"])
    require_file(Path(cfg["data"]["robot_config_root"]) / "robotwin.yaml")
    align = cfg["train"].get("align_params", {})
    for field in ("moge_path", "morgbd_path"):
        if align.get("depth", {}).get(field):
            require_file(align["depth"][field])
    if align.get("use_future_video"):
        for field in ("ckpt_path", "config_path"):
            require_file(align["video"][field])
    if data:
        check_dataset(cfg)


def check_dataset(cfg):
    """Check full metadata and files before model allocation; never accept a smoke subset."""
    sam = Path(cfg["data"]["train_path"])
    try:
        metadata = json.loads(require_file(sam / "meta/sam_info.json").read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Training data is not ready. Run bash process_data.sh first. {exc}") from exc
    import numpy as np
    import pandas as pd
    from lingbotvla.data.vla_data.submission_paths import raw_source_path
    if metadata.get("subset") or metadata.get("mode") != "multi":
        raise ValueError("Training requires the complete SAM multi dataset")
    raw = raw_source_path(metadata.get("source"), sam)
    if raw != owned(path("raw")):
        raise ValueError(f"SAM raw teacher source must match base.json paths.raw: {raw}")
    if not json.loads(require_file(raw / "repair_report.json").read_text()).get("complete"):
        raise ValueError("Raw frame repair is incomplete; finish process_data.sh first")
    frames = []
    for root in (sam, raw):
        info = json.loads(require_file(root / "meta/info.json").read_text())
        for filename in ("tasks.parquet", "stats.json"):
            require_file(root / "meta" / filename)
        if (info.get("total_episodes"), info.get("total_frames"), info.get("fps")) != (2500, 548893, 15):
            raise ValueError(f"Expected 2500 episodes / 548893 frames / 15 fps: {root}")
        files = sorted((root / "meta/episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(root / "meta/episodes")
        eps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("episode_index")
        if list(eps.episode_index) != list(range(2500)) or int(eps.length.sum()) != 548893:
            raise ValueError(f"Incomplete episode metadata: {root}")
        keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        wrists = {"observation.images.cam_left_wrist", "observation.images.cam_right_wrist"}
        if root == sam:
            expected = [f"observation.images.cam_high_{i}" for i in range(1, 101)]
            if metadata.get("video_keys") != expected or any(len(x) != 100 for x in eps.tasks):
                raise ValueError("SAM needs 100 ordered instruction/video slots per episode")
            if set(keys) != set(expected) | wrists:
                raise ValueError("SAM needs 100 top cameras and both wrist cameras")
        elif set(keys) != {"observation.images.cam_high"} | wrists:
            raise ValueError("Raw teacher data needs top and both wrist cameras")
        seen = set()
        for _, row in eps.iterrows():
            for key in keys:
                prefix = f"videos/{key}/"
                begin, end = row[prefix + "from_timestamp"], row[prefix + "to_timestamp"]
                if not np.isclose((end - begin) * 15, row.length, atol=1e-3):
                    raise ValueError(f"Frame count mismatch: {root}, episode {row.episode_index}, {key}")
                file = root / info["video_path"].format(video_key=key,
                    chunk_index=int(row[prefix + "chunk_index"]), file_index=int(row[prefix + "file_index"]))
                if file not in seen:
                    require_file(file)
                    seen.add(file)
            file = root / info["data_path"].format(chunk_index=int(row["data/chunk_index"]),
                                                  file_index=int(row["data/file_index"]))
            if file not in seen:
                require_file(file)
                seen.add(file)
        frames.append(eps)
    for column in ("episode_index", "length", "dataset_from_index", "dataset_to_index"):
        if not np.array_equal(frames[0][column], frames[1][column]):
            raise ValueError(f"SAM and raw teacher disagree on {column}")


def command(plan, rendered):
    topo = plan["topology"]
    return [plan["python"], "-B", "-m", "torch.distributed.run",
            *[f"--{key.replace('_', '-')}={topo[key]}" for key in
              ("nnodes", "nproc_per_node", "node_rank", "master_addr", "master_port")],
            str(base_path() / "tools/train_worker.py"),
            str(Path(plan["project"]) / "tasks/vla/train_lingbotvla.py"), str(rendered)]


def main():
    # Bootstrap through the configured Conda interpreter without activating or editing it.
    if os.environ.get("SUBMISSION_TRAIN_BOOTSTRAPPED") != str(base_path()):
        env = environment()
        env["SUBMISSION_TRAIN_BOOTSTRAPPED"] = str(base_path())
        executable = str(interpreter("vla"))
        os.execve(executable, [executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], env)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("YAML_FILE"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Print resolved config; no data/GPU/model loading")
    modes.add_argument("--check", action="store_true", help="Check complete data and model resources without training")
    modes.add_argument("--check-models", action="store_true", help="Check model resources before data processing finishes")
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ["--"]:
        overrides.pop(0)
    plan = make_plan(args.config, [*shlex.split(os.environ.get("EXTRA_ARGS", "")), *overrides])
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    check_assets(plan, data=not args.check_models)
    if args.check or args.check_models:
        print(f"Training {'models' if args.check_models else 'resources and full dataset'}: OK ({plan['profile']})")
        return 0
    import yaml
    output = writable(plan["config"]["train"]["output_dir"])
    logs = writable(path("training_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    run = f"{Path(plan['yaml']).stem}_n{plan['topology']['node_rank']}_{uuid.uuid4().hex[:12]}"
    rendered = logs / (run + ".yaml")
    rendered.write_text(yaml.safe_dump(plan["config"], sort_keys=False))
    cmd = command(plan, rendered)
    print(f"Training output: {output}\nLog: {logs / (run + '.log')}\n{shlex.join(cmd)}", flush=True)
    with (logs / (run + ".log")).open("w") as log:
        process = subprocess.Popen(cmd, cwd=plan["project"], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        def stop(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = process.wait()
        return code if code >= 0 else 128 - code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Training preflight failed: {exc}", file=sys.stderr)
        sys.exit(2)
