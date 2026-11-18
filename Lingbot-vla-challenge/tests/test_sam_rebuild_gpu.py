"""GPU starvation regression, deliberate sharing, and strict cache migration."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import rebuild_robotwin_sam as R
import rebuild_robotwin_sam_multinode as M
import sam_rebuild_gpu as G


class GPUSchedulingTests(unittest.TestCase):
    def test_queue_feeds_all_gpus_and_keeps_assembly_workers(self):
        info = {'judge': dict(model='qwen', threshold=.5, outside_scale=.2, regrow=2, regrow_min_area=.002)}
        for count in (1, 7, 8):
            args = SimpleNamespace(sam3_gpus=','.join(map(str, range(count))), compose_workers=64,
                                   pipeline='pipeline', raw='raw', out='out', work='work', qwen_url='')
            with patch.object(G, 'cpu_capacity', return_value={'available_cpus': 160}):
                argv = R.pipeline_argv(args, info, 'masks', [93])
            def value(key):
                return int(argv[argv.index('--' + key) + 1])
            self.assertEqual(value('max-queued'), 4 * count)
            self.assertGreater(value('max-queued'), count)
            self.assertEqual(value('prep-threads'), min(8, 2 * count))
            self.assertEqual(value('compose-workers'), 8)
            self.assertEqual(value('judge-concurrency'), 16)
            self.assertIn('--judge', argv)
            assembly = R.pipeline_argv(args, info, 'assemble')
            self.assertEqual(assembly[assembly.index('--compose-workers') + 1], '64')

    def test_overrides_and_invalid_values(self):
        with patch.dict(os.environ, {'DATAFIX_MAX_QUEUED': '40', 'DATAFIX_PREP_WORKERS': '6'}, clear=True):
            parser = argparse.ArgumentParser()
            G.add_arguments(parser)
            args = parser.parse_args(['--judge-concurrency', '8'])
        args.sam3_gpus, args.compose_workers = '0,1', 64
        config = G.scheduling(args)
        self.assertEqual(config['max_queued'], 40)
        self.assertEqual(config['prep_workers'], 6)
        self.assertEqual(config['judge_concurrency'], 8)
        args.max_queued = 0
        with self.assertRaises(ValueError):
            G.scheduling(args)
        args.max_queued, args.sam3_gpus = 'auto', '1,1'
        with self.assertRaises(ValueError):
            G.scheduling(args)

    def test_sharing_includes_allocated_zero_only_when_requested(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2,5'}, clear=True):
            self.assertEqual(M.sam_gpus(None, True), '0,2,5')
            self.assertEqual(M.sam_gpus('0,5', True), '0,5')
            self.assertEqual(M.sam_gpus('5', True), '5')
            with self.assertRaises(ValueError):
                M.sam_gpus('0,1', True)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=True):
            self.assertEqual(M.sam_gpus(None, True), '0')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,2'}, clear=True):
            with self.assertRaises(ValueError):
                M.sam_gpus(None, True)

    def test_memory_guard_does_not_query_dedicated_sam(self):
        with patch.object(G.subprocess, 'check_output', return_value='49128\n') as query:
            G.check_shared_gpu(['1', '2'])
            query.assert_not_called()
            G.check_shared_gpu(['0', '1'])
            query.return_value = '8000\n'
            with self.assertRaisesRegex(RuntimeError, '16 GiB'):
                G.check_shared_gpu(['0'])

    def test_only_audited_scheduler_changes_migrate(self):
        helper = Path(G.__file__).resolve()
        expected = dict(version=4, signature={'judge': 'same'}, inputs_sha256={
            str(helper): 'new-gpu-helper', str(helper.with_name('rebuild_robotwin_sam.py')): 'new-scheduler',
            str(helper.with_name('sam_rebuild_cpu.py')): 'same-cpu-code', '/raw/metadata': 'original'})
        for version, digest in G.AUDITED_REBUILD.items():
            previous = copy.deepcopy(expected)
            previous['version'] = version
            previous['inputs_sha256'].pop(str(helper))
            if version == 2:
                previous['inputs_sha256'].pop(str(helper.with_name('sam_rebuild_cpu.py')))
            previous['inputs_sha256'][str(helper.with_name('rebuild_robotwin_sam.py'))] = digest
            self.assertTrue(G.compatible_single_manifest(previous, expected))
            for field in ('source', 'script', 'judge'):
                changed = copy.deepcopy(previous)
                if field == 'source':
                    changed['inputs_sha256']['/raw/metadata'] = 'changed'
                elif field == 'script':
                    changed['inputs_sha256'][str(helper.with_name('rebuild_robotwin_sam.py'))] = 'unknown'
                else:
                    changed['signature']['judge'] = 'different'
                self.assertFalse(G.compatible_single_manifest(changed, expected))
        self.assertTrue(G.compatible_single_manifest(expected, expected))


if __name__ == '__main__':
    unittest.main()
