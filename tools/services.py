#!/usr/bin/env python3
"""Launch only the Qwen / SAM3 services owned by this invocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_paths import base_path, config, interpreter, path, runtime_env, writable


def qwen_url():
    q = config()["qwen"]
    return f'http://{q["host"]}:{q["port"]}/v1/chat/completions'


def qwen_command():
    q = config()["qwen"]
    return [str(interpreter("vllm")), "-B", "-m", "vllm.entrypoints.cli.main", "serve",
            str(path("qwen_model")), "--served-model-name", q["model_name"],
            "--host", q["host"], "--port", str(q["port"]),
            "--max-model-len", str(q["max_model_len"]),
            "--gpu-memory-utilization", str(q["gpu_memory_utilization"]),
            "--max-num-seqs", str(q["max_num_seqs"]), "--gdn-prefill-backend", "triton",
            "--tensor-parallel-size", str(q.get("tensor_parallel_size", 1)),
            "--limit-mm-per-prompt", '{"image":2,"video":0}', "--enable-prefix-caching"]


def stream_command(url):
    s = config()["inference"]
    return [str(interpreter("sam3")), "-B", str(base_path() / "sam3/tools/sam3_stream_server.py"),
            "--host", s["host"], "--port", str(s["port"]),
            "--checkpoint", str(path("sam_checkpoint")), "--memory-window", str(s["memory_window"]),
            "--score-threshold-detection", "0.3", "--new-det-thresh", "0.5",
            "--judge-url", url, "--judge-model", config()["qwen"]["model_name"],
            "--judge-threshold", "0.5", "--judge-outside-scale", "0.2",
            "--judge-regrow", "2.0", "--judge-regrow-min-area", "0.002", "--judge-timeout", "120"]


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def check_qwen(url):
    endpoint = url.split("/v1")[0].rstrip("/") + "/v1/models"
    names = {item["id"] for item in get_json(endpoint)["data"]}
    if config()["qwen"]["model_name"] not in names:
        raise RuntimeError(f"Endpoint does not serve the expected Qwen model: {names}")


class Service:
    def __init__(self, name, command, settings):
        self.name, self.command, self.settings = name, command, settings
        self.process = None
        self.log = None

    def start(self):
        host, port = self.settings["host"], int(self.settings["port"])
        with socket.socket() as sock:
            try:
                sock.bind((host, port))
            except OSError as exc:
                raise RuntimeError(f"{self.name} port {host}:{port} is busy. To use an existing Qwen, pass --qwen-url.") from exc
        env = runtime_env(f"{self.name}-{port}")
        if self.name == "qwen" and len(os.fsencode(env["VLLM_RPC_BASE_PATH"])) + 37 > 107:
            raise ValueError("Submission path is too long for vLLM Unix sockets; relocate the bundle to a shorter path")
        env["CUDA_VISIBLE_DEVICES"] = str(self.settings["gpu"])
        env["OMP_NUM_THREADS"] = "8" if self.name == "qwen" else "4"
        logfile = writable(path("runtime") / socket.gethostname() / f"{self.name}-{port}" / "service.log")
        self.log = logfile.open("a")
        print(f"Starting {self.name}: GPU={env['CUDA_VISIBLE_DEVICES']}; log={logfile}", flush=True)
        try:
            self.process = subprocess.Popen(self.command, cwd=base_path(), env=env,
                                            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + int(self.settings["startup_timeout"])
            last_report = 0
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"{self.name} exited with {self.process.returncode}; see {logfile}")
                try:
                    if self.name == "qwen":
                        check_qwen(f"http://{host}:{port}")
                    elif not get_json(f"http://{host}:{port}/health").get("loaded"):
                        raise RuntimeError("SAM3 is still loading")
                    print(f"{self.name} ready at http://{host}:{port}", flush=True)
                    return self
                except (OSError, ValueError, KeyError, RuntimeError):
                    pass
                if time.monotonic() - last_report > 30:
                    print(f"Waiting for {self.name}; log={logfile}", flush=True)
                    last_report = time.monotonic()
                time.sleep(2)
            raise TimeoutError(f"{self.name} startup timed out; see {logfile}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["qwen", "inference"])
    ap.add_argument("--qwen-url", help="Use an already running Qwen endpoint (must allow two images)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    url = args.qwen_url or qwen_url()
    commands = ([] if args.qwen_url else [("qwen", qwen_command(), config()["qwen"])])
    if args.mode == "inference":
        commands.append(("sam3", stream_command(url), config()["inference"]))
    if not commands:
        ap.error("qwen mode starts a new service; omit --qwen-url")
    for name, command, settings in commands:
        print(f"CUDA_VISIBLE_DEVICES={settings['gpu']} {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    running = []
    try:
        if args.qwen_url:
            check_qwen(url)
        for name, command, settings in commands:
            running.append(Service(name, command, settings).start())
        print("Services ready. Keep this command running; Ctrl-C stops the services it started.", flush=True)
        while True:
            for service in running:
                if service.process.poll() is not None:
                    raise RuntimeError(f"{service.name} exited unexpectedly")
            time.sleep(2)
    finally:
        for service in reversed(running):
            service.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
