"""Real spawn processes, failure cleanup, CPU quota and audited cache migration."""
from contextlib import closing
import copy
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import sam_rebuild_cpu as C


def pid_job(number):
    time.sleep(.05)
    return number, os.getpid(), signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def fail_job(number):
    raise ValueError('deliberate worker error')


def crash_job(number):
    os._exit(7)


def fail_initializer():
    raise ValueError('deliberate init error')


class CPUProcessTests(unittest.TestCase):
    def test_auto_uses_allocated_budget_and_native_threads(self):
        with patch.object(C, 'cpu_capacity', return_value={'available_cpus': 160}):
            self.assertEqual(C.resolve_workers('auto', 2), 64)
            self.assertEqual(C.resolve_workers('auto', 4), 32)
            self.assertEqual(C.resolve_workers('12', 2), 12)
        with patch.object(C, 'cpu_capacity', return_value={'available_cpus': .5}):
            self.assertEqual(C.resolve_workers('auto', 2), 1)
        with self.assertRaises(ValueError):
            C.resolve_workers(0)
        with self.assertRaises(ValueError):
            C.resolve_workers('auto', 0)

    def test_affinity_and_cgroup_quota_are_both_respected(self):
        with patch.object(os, 'sched_getaffinity', return_value=set(range(12))):
            with patch.object(Path, 'read_text', return_value='250000 100000'):
                self.assertEqual(C.cpu_capacity()['available_cpus'], 2.5)
            with patch.object(Path, 'read_text', return_value='max 100000'):
                self.assertEqual(C.cpu_capacity()['available_cpus'], 12)

    def test_jobs_run_in_distinct_processes_not_parent_threads(self):
        before = {p.pid for p in mp.active_children()}
        results = list(C.spawn_results(pid_job, range(24), 3))
        self.assertEqual(sorted(r[0] for r in results), list(range(24)))
        self.assertEqual(len({r[1] for r in results}), 3)
        self.assertNotIn(os.getpid(), {r[1] for r in results})
        self.assertTrue(all(r[2] for r in results))
        self.assertEqual({p.pid for p in mp.active_children()}, before)

    def test_python_error_initializer_error_and_worker_death_propagate(self):
        before = {p.pid for p in mp.active_children()}
        for function, initializer, error in ((fail_job, None, 'deliberate worker error'),
                                             (pid_job, fail_initializer, 'deliberate init error'),
                                             (crash_job, None, 'exited unexpectedly')):
            with self.assertRaisesRegex(RuntimeError, error):
                list(C.spawn_results(function, range(10), 2, initializer=initializer))
            self.assertEqual({p.pid for p in mp.active_children()}, before)

    def test_parent_interrupt_closes_only_its_own_children(self):
        before = {p.pid for p in mp.active_children()}
        with self.assertRaises(KeyboardInterrupt):
            with closing(C.spawn_results(pid_job, range(1000), 2)) as results:
                next(results)
                raise KeyboardInterrupt
        self.assertEqual({p.pid for p in mp.active_children()}, before)

    def test_only_known_v2_scheduling_manifest_can_migrate(self):
        helper = Path(C.__file__).resolve()
        script = helper.with_name('rebuild_robotwin_sam.py')
        expected = dict(version=3, raw='/same/raw', signature={'judge': 'same'},
                        inputs_sha256={str(script): 'current', str(helper): 'helper', '/data/meta': 'unchanged'})
        previous = copy.deepcopy(expected)
        previous['version'] = 2
        del previous['inputs_sha256'][str(helper)]
        previous['inputs_sha256'][str(script)] = C.LEGACY_REBUILD_SHA256
        self.assertTrue(C.compatible_single_manifest(previous, expected))
        self.assertTrue(C.compatible_single_manifest(expected, expected))
        for mutate in (lambda m: m.update(raw='/different'),
                       lambda m: m['inputs_sha256'].update({'/data/meta': 'changed'}),
                       lambda m: m['inputs_sha256'].update({str(script): 'unknown'})):
            wrong = copy.deepcopy(previous)
            mutate(wrong)
            self.assertFalse(C.compatible_single_manifest(wrong, expected))


if __name__ == '__main__':
    unittest.main()
