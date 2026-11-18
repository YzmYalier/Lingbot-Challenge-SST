#!/usr/bin/env python3
"""Run a configured interpreter with submission code and private caches."""
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, interpreter, path, runtime_env, writable


def environment(role, *, runtime_name=None, inherit_warp_cache=True):
    env = runtime_env(runtime_name or "eval-" + role)
    root = base_path()
    sources = [root / "tools/python_bootstrap", root]
    if role == "vla":
        project = path("vla_project")
        sources += [project, project / "lingbotvla/models/vla/vision_models/MoGe",
                    project / "lingbotvla/models/vla/vision_models/lingbot-depth"]
    elif role == "robotwin":
        sources += [path("robotwin"), path("robotwin") / "envs/curobo/src"]
        graphics = path("graphics_lib")
        env["LD_LIBRARY_PATH"] = str(graphics) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        # SAPIEN ships a Vulkan loader and NVIDIA manifests. EGL dispatch is
        # bundled separately because minimal CUDA containers often omit it.
        prefix = interpreter(role).parent.parent
        for library in prefix.glob("lib/python*/site-packages/sapien/vulkan_library"):
            env["VK_ICD_FILENAMES"] = str(library / "nvidia_icd.json")
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(library / "10_nvidia.json")
            break
    elif role == "sam3":
        sources += [path("sam_source")]
    elif role == "vllm":
        sources += [path("vllm_source")]
    for source in sources:
        if not source.resolve().is_relative_to(root):
            raise ValueError(f"Project code must be inside the submission: {source}")
    env["PYTHONPATH"] = os.pathsep.join(map(str, sources))
    env["SUBMISSION_ROOT"] = str(root)
    env["PATH"] = str(interpreter(role).parent) + os.pathsep + env["PATH"]
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.0.0"
    if inherit_warp_cache and os.environ.get("WARP_CACHE_PATH"):
        env["WARP_CACHE_PATH"] = str(writable(os.environ["WARP_CACHE_PATH"]))
    if role == "robotwin":
        # NVRTC's intermediate PCH files are not safe for concurrent writers.
        # A fresh directory also isolates retries from an interrupted compiler.
        warp_cache = writable(Path(env["WARP_CACHE_PATH"]) / ("process-" + uuid.uuid4().hex))
        warp_cache.mkdir(parents=True, exist_ok=True)
        env["WARP_CACHE_PATH"] = str(warp_cache)
    env["__GL_SHADER_DISK_CACHE_PATH"] = env["XDG_CACHE_HOME"] + "/nvidia"
    env["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    for key in ("PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(key, None)
    return env


if __name__ == "__main__":
    role, *args = sys.argv[1:]
    executable = str(interpreter(role))
    os.execve(executable, [executable, "-B", *args], environment(role))
