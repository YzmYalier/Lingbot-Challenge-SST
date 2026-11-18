"""CPU tests for numeric identity, original instruction order and safe publication."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import submit_paths
from tools import prepare_data_inputs as P
from tools.remote_zip import RangeReader


class InputPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke = submit_paths.writable(submit_paths.path("input_work") / "cpu-smoke")
        cls.smoke.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.smoke)
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def source(self, ep):
        state = (np.arange(42).reshape(3, 14) + ep * 100).astype("float32")
        action = state + 0.25
        stream = io.BytesIO()
        with h5py.File(stream, "w") as h:
            for group, array in (("state", state), ("action", action)):
                for name, sl in zip(P.PARTS, (slice(0, 6), slice(6, 7), slice(7, 13), slice(13, 14))):
                    h.create_dataset(f"{group}/{name}", data=array[:, sl])
        instruction = json.dumps({"seen": [f"instruction {i % 7}" for i in range(100)],
                                  "unseen": ["unseen"]}, indent=3).encode()
        signature, length = P.hdf5_signature(stream.getvalue())
        self.assertEqual(signature, P.trajectory_signature(state, action))
        record = {"task": "adjust_bottle", "relative": f"aloha_agilex/data/episode_{ep:07d}.hdf5",
                  "signature": signature, "length": length,
                  "instruction_sha256": P.sha256(instruction),
                  "instruction_base64": base64.b64encode(instruction).decode()}
        return state, action, record

    def test_cpu_zip_to_exact_episode_mapping(self):
        sources = [self.source(0), self.source(1)]
        rows = []
        for ep, (state, action, _) in enumerate(sources):
            rows.extend({"episode_index": ep, "frame_index": i,
                         "observation.state": state[i].tolist(), "action": action[i].tolist()}
                        for i in range(len(state)))
        table = pa.Table.from_pylist(rows)
        stream = io.BytesIO()
        pq.write_table(table, stream)
        archive = self.root / "reference.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("raw/meta/info.json", json.dumps({"codebase_version": "v3.0",
                        "total_episodes": 2, "total_frames": 6, "fps": 15}))
            z.writestr("raw/data/chunk-000/file-000.parquet", stream.getvalue())
        entry = {"size": archive.stat().st_size, "sha256": P.sha256(archive.read_bytes())}
        with patch.object(P, "EPISODES", 2), patch.object(P, "FRAMES", 6), \
             patch.object(P, "path", return_value=archive), \
             patch.object(P, "validate_reference", side_effect=lambda rs: P_validate(rs, episodes=2, frames=6)):
            reference = P.reference_signatures(entry, self.root)
        # Raw archive order differs from target episode order, on purpose.
        result = P.match_records(reference, [sources[1][2], sources[0][2]], episodes=2, frames=6)
        self.assertEqual(result[0]["relative"], sources[0][2]["relative"])
        self.assertEqual(result[1]["relative"], sources[1][2]["relative"])
        data = base64.b64decode(result[0]["instruction_base64"])
        self.assertEqual(P.validate_instruction(data)["seen"], [f"instruction {i % 7}" for i in range(100)])

    def test_full_state_and_action_both_determine_identity(self):
        state, action, row = self.source(0)
        for which in (0, 1):
            arrays = [state.copy(), action.copy()]
            arrays[which][-1, -1] += 1
            self.assertNotEqual(P.trajectory_signature(*arrays), row["signature"])

    def test_duplicate_or_missing_raw_match_is_rejected(self):
        _, _, row = self.source(0)
        ref = [{"episode_index": 0, "length": 3, "signature": row["signature"]}]
        with self.assertRaises(ValueError):
            P.match_records(ref, [], episodes=1, frames=3)
        with self.assertRaises(ValueError):
            P.match_records(ref, [row, row], episodes=1, frames=3)
        with self.assertRaises(ValueError):
            P.match_records(ref, [dict(row, signature="0" * 64)], episodes=1, frames=3)

    def test_instruction_integrity_and_no_padding(self):
        for seen in (["a"] * 99, ["a"] * 101, ["{object}"] * 100, [""] * 100):
            with self.assertRaises(ValueError):
                P.validate_instruction(json.dumps({"seen": seen}))
        _, _, row = self.source(0)
        row["instruction_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            P.match_records([{"episode_index": 0, "length": 3, "signature": row["signature"]}],
                            [row], episodes=1, frames=3)

    def test_no_partial_official_publication(self):
        with self.assertRaisesRegex(ValueError, "complete original dataset"):
            P.publish({0: {"length": 3}}, {}, {})

    def test_paths_cannot_escape_inputs(self):
        for task, rel in (("../task", "robot/data/a.hdf5"), ("task", "../data/a.hdf5"),
                          ("task", "/robot/data/a.hdf5"), ("task", "robot/data/../../a.hdf5")):
            with self.assertRaises(ValueError):
                P.instruction_relative(task, rel)

    def test_venv_executable_symlink_is_preserved(self):
        executable = self.root / "bin/python"
        executable.parent.mkdir()
        executable.symlink_to(sys.executable)
        with patch.object(submit_paths, "config", return_value={"interpreters": {"inputs": str(executable)}}):
            self.assertEqual(submit_paths.interpreter("inputs"), executable)

    def test_corrupt_cache_is_not_reused(self):
        file = self.root / "cached.json"
        P.save_cache(file, {"v": 1}, [{"id": 1}])
        self.assertEqual(P.read_cache(file, {"v": 1}), [{"id": 1}])
        self.assertIsNone(P.read_cache(file, {"v": 2}))
        file.write_text('{}')
        self.assertIsNone(P.read_cache(file, {"v": 1}))

    def test_render_and_judge_have_distinct_outside_scales(self):
        info = json.loads(submit_paths.path("checkpoint_sam_info").read_text())
        self.assertEqual(info["style"], {"name": "outline", "outside_scale": .1,
                                      "edge_width": 2, "edge_color": [255, 255, 255]})
        self.assertEqual(info["judge"]["outside_scale"], .2)
        self.assertEqual(info["judge"]["threshold"], .5)

    def test_eval_false_judge_does_not_enable_it(self):
        from tools import evaluation_config as E
        original_path = E.path
        file = self.root / "sam_info.json"
        for value, expected in ((False, "0"), (None, "0"), ({"threshold": .5}, "1")):
            file.write_text(json.dumps({"style": {"outside_scale": .1, "edge_width": 2}, "judge": value}))
            with patch.object(E, "path", side_effect=lambda k: file if k == "checkpoint_sam_info" else original_path(k)):
                self.assertEqual(E.settings()["SAM_JUDGE"], expected)

    def test_remote_zip_exact_ranges_and_member_crc(self):
        content = b"numeric-fixture" * 15000
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("task/data/a.hdf5", content)
        archive = stream.getvalue()
        requests = []

        def open_range(request, timeout):
            start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
            response = io.BytesIO(archive[start:end+1])
            response.status = 206
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(archive)}"}
            response.geturl = lambda: request.full_url
            requests.append((start, end))
            return response

        url = "https://huggingface.co/datasets/test/resolve/fixed/archive.zip"
        with patch("tools.remote_zip.urllib.request.urlopen", side_effect=open_range):
            with RangeReader(url, len(archive), self.root / "http", retries=1) as reader:
                with zipfile.ZipFile(reader) as z:
                    member = z.getinfo("task/data/a.hdf5")
                    reader.member_prefetch[member.header_offset] = min(len(archive), member.compress_size+16384)
                    self.assertEqual(z.read(member), content)
            count = len(requests)
            with RangeReader(url, len(archive), self.root / "http", retries=1) as reader:
                with zipfile.ZipFile(reader) as z:
                    reader.member_prefetch[0] = min(len(archive), len(content)+16384)
                    self.assertEqual(z.read("task/data/a.hdf5"), content)
            self.assertEqual(len(requests), count)  # resume uses verified disk ranges

    def test_server_ignoring_range_is_rejected(self):
        response = io.BytesIO(b"wrong")
        response.status, response.headers = 200, {}
        with patch("tools.remote_zip.urllib.request.urlopen", return_value=response):
            with RangeReader("https://huggingface.co/datasets/test/resolve/fixed/a.zip", 5,
                             self.root / "bad-http", retries=1) as reader:
                with self.assertRaisesRegex(OSError, "exact requested byte range"):
                    reader.read(5)

    def test_adjacent_prefetched_members_work_without_network(self):
        content = b"numeric-fixture" * 15000
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as z:
            for i in range(3):
                z.writestr(f"task/data/{i}.hdf5", content)
        archive = stream.getvalue()

        def open_range(request, timeout):
            start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
            response = io.BytesIO(archive[start:end+1])
            response.status = 206
            response.headers = {"Content-Range": f"bytes {start}-{end}/{len(archive)}"}
            response.geturl = lambda: request.full_url
            return response

        with patch("tools.remote_zip.urllib.request.urlopen", side_effect=open_range):
            with RangeReader("https://huggingface.co/datasets/test/resolve/fixed/adjacent.zip",
                             len(archive), self.root / "adjacent", retries=1) as reader:
                with zipfile.ZipFile(reader) as z:
                    for member in z.infolist():
                        start = member.header_offset
                        count = min(len(archive)-start, member.compress_size+16384)
                        reader.member_prefetch[start] = count
                        reader._range(start, count)
                    reader.memory.clear()
                    with patch("tools.remote_zip.urllib.request.urlopen", side_effect=AssertionError("Unexpected network")):
                        for member in z.infolist():
                            self.assertEqual(z.read(member), content)


P_validate = P.validate_reference

if __name__ == "__main__":
    unittest.main()
