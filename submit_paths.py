"""Resolve every submission-owned path from the single root configuration."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().with_name("base.json")


def config():
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def base_path() -> Path:
    root = Path(config()["base_path"]).expanduser()
    if not root.is_absolute():
        root = CONFIG_FILE.parent / root
    root = root.resolve()
    if root != CONFIG_FILE.parent:
        raise ValueError("base.json points to a different checkout. Run: python3 tools/configure.py --relocate")
    return root


def resolve(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return (value if value.is_absolute() else base_path() / value).resolve()


def path(name: str) -> Path:
    return resolve(config()["paths"][name])


def interpreter(role: str) -> Path:
    value = Path(config()["interpreters"][role]).expanduser()
    if not value.is_absolute():
        value = base_path() / value
    # Keep the executable's final symlink: resolving venv/bin/python to the
    # system Python loses pyvenv.cfg and all packages installed in the venv.
    return value.parent.resolve() / value.name


def writable(value: str | Path) -> Path:
    value = resolve(value)
    if not value.is_relative_to(base_path()) or value == base_path():
        raise ValueError(f"Writes must remain inside base_path: {value}")
    return value


def runtime_env(name: str, *, create=True) -> dict[str, str]:
    """Keep caches and temporary files inside the submission, including subprocesses."""
    import socket
    runtime = writable(path("runtime") / socket.gethostname() / name)
    env = dict(os.environ)
    directories = {
        "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "HF_HOME": "cache/huggingface",
        "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor", "CUDA_CACHE_PATH": "cache/cuda",
        "VLLM_CACHE_ROOT": "cache/vllm", "VLLM_RPC_BASE_PATH": "rpc",
        "FLASHINFER_WORKSPACE_BASE": "cache/flashinfer", "NUMBA_CACHE_DIR": "cache/numba",
        "MPLCONFIGDIR": "cache/matplotlib", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "WARP_CACHE_PATH": "cache/warp", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
        "PIP_CACHE_DIR": "cache/pip", "UV_CACHE_DIR": "cache/uv", "CONDA_PKGS_DIRS": "cache/conda",
        "UV_PYTHON_INSTALL_DIR": "cache/uv-python",
    }
    for key, relative in directories.items():
        target = runtime / relative
        if create:
            target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    # vLLM uses Unix-domain sockets, whose filesystem path is limited to 107 bytes.
    # Keep these under the bundle, but much shorter than the normal cache hierarchy.
    tag = hashlib.sha256((socket.gethostname() + "/" + name).encode()).hexdigest()[:12]
    ipc = writable(base_path() / ".ipc" / tag)
    if create:
        ipc.mkdir(parents=True, exist_ok=True)
    env["VLLM_RPC_BASE_PATH"] = str(ipc)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    # Do not inherit an unrelated checkout's Python path.
    env["SUBMISSION_ROOT"] = str(base_path())
    env["FFMPEG_BINARY"] = str(path("ffmpeg"))
    env["PATH"] = str(path("ffmpeg").parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join(map(str, [
        base_path() / "tools/python_bootstrap", base_path(),
        path("sam_source"), path("vllm_source"),
    ]))
    return env
