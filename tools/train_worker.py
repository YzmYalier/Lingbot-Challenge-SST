#!/usr/bin/env python3
"""Give each torchrun worker private compiler caches before importing PyTorch."""
import os
from pathlib import Path
import runpy
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import writable


def isolate_compiler_caches():
    rank = int(os.environ["LOCAL_RANK"])
    # Concurrent compilers must not share partially written cubin files.
    # A unique worker namespace also separates overlapping jobs and restarts.
    namespace = f"worker-{rank}-{uuid.uuid4().hex[:12]}"
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
        directory = writable(Path(os.environ[key]) / namespace)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)


def main():
    script = writable(sys.argv[1])
    if not script.is_file():
        raise FileNotFoundError(script)
    isolate_compiler_caches()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
