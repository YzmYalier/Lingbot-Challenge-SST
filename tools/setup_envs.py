#!/usr/bin/env python3
"""Create isolated runtime environments at the paths configured in base.json."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, interpreter, runtime_env, writable


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("role", choices=["inputs", "data", "sam3", "vllm"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.role == "vllm":
        from setup_vllm import main as setup_vllm
        return setup_vllm(["--dry-run"] if args.dry_run else [])
    root = base_path()
    exe = interpreter(args.role)
    writable(exe.parent)
    prefix = exe.parent.parent
    if exe.name != "python" or exe.parent.name != "bin":
        raise ValueError("setup_envs expects <environment>/bin/python")
    if prefix.exists() and not (prefix / "pyvenv.cfg").exists():
        raise ValueError(f"Refusing to install into an existing non-venv directory: {prefix}")
    python_version = "3.12"
    commands = []
    if not exe.exists():
        commands.append(["uv", "venv", "--seed", "--python", python_version, str(prefix)])
    pip = ["uv", "pip", "install", "--python", str(exe)]
    if args.role in ("inputs", "data", "sam3"):
        command = pip + ["-r", str(root / "environments" / f"requirements-{args.role}.txt")]
        if args.role == "sam3":
            command += ["--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"]
        commands.append(command)
        if args.role == "sam3":
            commands.append(pip + ["--no-deps", "-e", str(root / "sam3")])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            env = runtime_env("setup-" + args.role)
            subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
