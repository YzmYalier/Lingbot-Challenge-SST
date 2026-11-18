"""Ownership, sharding, GPU reservation and rendezvous tests (no models)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import rebuild_robotwin_sam_multinode as M


class MultinodeTests(unittest.TestCase):
    def test_all_original_ids_owned_once(self):
        for nodes in (1, 2, 3, 8):
            shards = [M.shard_episodes(range(2500), rank, nodes) for rank in range(nodes)]
            flat = sum(shards, [])
            self.assertEqual(sorted(flat), list(range(2500)))
            self.assertEqual(len(set(flat)), 2500)

    def test_gpu_zero_reserved_and_allocation_respected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2,5'}, clear=True):
            self.assertEqual(M.sam_gpus(), '2,5')
            self.assertEqual(M.sam_gpus('5'), '5')
            for selected in ('0', '1', '2,2', ''):
                with self.assertRaises(ValueError):
                    M.sam_gpus(selected)
        with patch.dict(os.environ, {'HOST_GPU_NUM': '4'}, clear=True):
            self.assertEqual(M.sam_gpus(), '1,2,3')
        for visible in ('0', '1,2', 'GPU-abc'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}, clear=True):
                with self.assertRaises(ValueError):
                    M.sam_gpus()

    def test_single_writer_and_readonly_cache_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lock'
            with M.locked(path):
                for shared in (False, True):
                    with self.assertRaisesRegex(RuntimeError, 'in use'):
                        with M.locked(path, shared=shared):
                            self.fail('acquired active lock')
            with M.locked(path, shared=True):
                with M.locked(path, shared=True):
                    pass

    def test_workspace_pins_node_count_and_refuses_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(work=root / 'work', out=root / 'out')
            M.initialize(args, {'nodes': 2}, {})
            M.initialize(args, {'nodes': 2}, {})
            with self.assertRaisesRegex(ValueError, 'changed'):
                M.initialize(args, {'nodes': 3}, {})
            args.work = root / 'unknown'
            args.work.mkdir()
            (args.work / 'existing').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'Unrecognized'):
                M.initialize(args, {}, {})
            self.assertEqual((args.work / 'existing').read_text(), 'keep')

    def test_barrier_requires_all_shards_and_propagates_failure(self):
        class Progress:
            detail = ''
            def stage(self, *args, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(work=root, nodes=2, started=time.time(), wait_timeout=.001, report_every=.001)
            for rank in range(2):
                (root / 'nodes' / f'rank{rank}').mkdir(parents=True)
            M.R.write_json(root / 'nodes/rank0/complete.json', {'fingerprint': 'a'})
            with self.assertRaises(TimeoutError):
                M.wait_shards(args, 'a', Progress())
            M.R.write_json(root / 'nodes/rank1/failure.json',
                           dict(fingerprint='a', time=time.time(), error='Qwen failed'))
            with self.assertRaisesRegex(RuntimeError, 'Qwen failed'):
                M.wait_shards(args, 'a', Progress())
            M.R.write_json(root / 'nodes/rank1/complete.json', {'fingerprint': 'wrong'})
            with self.assertRaises(RuntimeError):
                M.wait_shards(args, 'a', Progress())
            M.R.write_json(root / 'nodes/rank1/complete.json', {'fingerprint': 'a'})
            M.wait_shards(args, 'a', Progress())

    def test_publication_never_cleans_another_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(work=root / 'work', out=root / 'out')
            partial = root / 'out.partial'
            partial.mkdir()
            (partial / 'keep').write_text('other run')
            with self.assertRaisesRegex(ValueError, 'unowned'):
                with M.publication(args, 'a'):
                    self.fail('accepted unowned output')
            self.assertEqual((partial / 'keep').read_text(), 'other run')
            args.out = root / 'new'
            with M.publication(args, 'a'):
                pass
            with M.publication(args, 'a'):
                pass
            args.work = root / 'different-work'
            with self.assertRaisesRegex(ValueError, 'another rebuild'):
                with M.publication(args, 'a'):
                    self.fail('accepted another owner')

    def test_resume_promotion_discards_old_judges_but_keeps_fresh_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = SimpleNamespace(judge_json=lambda ep: root / f'judge{ep}.json',
                                   done_json=lambda ep: root / f'done{ep}.json')
            for ep in (0, 1):
                M.R.write_json(work.done_json(ep), {})
            M.R.write_json(work.judge_json(0), dict(reused_final_masks_from='old', phrases={}))
            M.R.write_json(work.judge_json(1), dict(phrases={'new': {}}))
            M.clear_promoted_reuse(work, [0, 1])
            self.assertFalse(work.done_json(0).exists())
            self.assertFalse(work.judge_json(0).exists())
            self.assertTrue(work.done_json(1).exists())
            self.assertEqual(M.read_json(work.judge_json(1)), {'phrases': {'new': {}}})


if __name__ == '__main__':
    unittest.main()
