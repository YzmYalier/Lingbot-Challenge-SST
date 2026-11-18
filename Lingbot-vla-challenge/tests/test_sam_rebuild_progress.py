"""Progress reports must remain live during long work and preserve failures."""
import concurrent.futures as cf
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from sam_rebuild_progress import Progress


class ProgressTests(unittest.TestCase):
    def test_heartbeat_and_cached_work_not_counted_as_throughput(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            with Progress(d, interval=.02) as progress:
                progress.stage('sam_judge_render', 100, completed=90, cached=90)
                self.assertIsNone(progress.snapshot()['eta_seconds'])
                # Event confirms that progress keeps refreshing without completed episodes.
                heartbeat = threading.Event()
                progress.refresh = lambda: (heartbeat.set(), 90)[1]
                self.assertTrue(heartbeat.wait(timeout=2))
                self.assertEqual(json.loads((Path(d) / 'progress.json').read_text())['completed'], 90)
                progress.refresh = None
                progress.advance()
                self.assertGreater(progress.snapshot()['eta_seconds'], 0)
            state = json.loads((Path(d) / 'progress.json').read_text())
            self.assertEqual(state['status'], 'finished')
            self.assertEqual(state['completed'], 91)
            self.assertEqual(state['cached'], 90)
            self.assertIn('[sam_judge_render]', (Path(d) / 'progress.log').read_text())

    def test_failed_futures_are_not_reported_as_successes(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'encoding failed'):
                with Progress(d) as progress:
                    progress.stage('assemble_videos', 3)
                    good, bad, cancelled = cf.Future(), cf.Future(), cf.Future()
                    good.set_result(None)
                    bad.set_exception(RuntimeError('encoding failed'))
                    cancelled.cancel()
                    for f in (good, bad, cancelled):
                        progress.future_done(f)
                    self.assertEqual(progress.snapshot()['completed'], 1)
                    self.assertEqual(progress.snapshot()['failed'], 2)
                    bad.result()
            state = json.loads((Path(d) / 'progress.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['completed'], 1)
            self.assertIn('encoding failed', state['detail'])


if __name__ == '__main__':
    unittest.main()
