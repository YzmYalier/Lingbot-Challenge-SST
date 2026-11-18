"""Offline checks for resumable downloads, mirror hashes and parallel reporting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_public_weights as d
from submit_paths import path, writable


class DownloadTests(unittest.TestCase):
    def setUp(self):
        root = writable(path('public_weights_state') / 'tests')
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b'preserve-exact-model-bytes'
        self.entry = dict(url='https://huggingface.co/owner/model/resolve/commit/file',
                          size=len(self.payload), md5=hashlib.md5(self.payload).hexdigest(),
                          sha256=hashlib.sha256(self.payload).hexdigest(), group='qwen3_8',
                          destination=dict(path_key='qwen_model', relative='test'))

    def response(self, payload, status=200, content_range=None):
        response = io.BytesIO(payload)
        response.status = status
        response.headers = {'Content-Range': content_range} if content_range else {}
        return response

    def test_resumes_only_valid_byte_range(self):
        target = self.root / 'model'
        target.with_suffix('.part').write_bytes(self.payload[:5])
        response = self.response(self.payload[5:], 206, f'bytes 5-{len(self.payload)-1}/{len(self.payload)}')
        with patch.object(d, 'request_file', return_value=response) as request:
            result = d.download(self.entry, target, retries=1, timeout=1, opener=None)
        self.assertTrue(result['ok'])
        self.assertEqual(request.call_args.args[1], 5)
        self.assertEqual(target.read_bytes(), self.payload)

    def test_invalid_range_never_appends(self):
        target = self.root / 'model'
        partial = target.with_suffix('.part')
        partial.write_bytes(self.payload[:5])
        with patch.object(d, 'request_file', return_value=self.response(b'wrong', 206, 'bytes 0-4/25')):
            with self.assertRaises(d.DownloadError):
                d.download(self.entry, target, retries=1, timeout=1, opener=None)
        self.assertEqual(partial.read_bytes(), self.payload[:5])
        self.assertFalse(target.exists())

    def test_mirror_wrong_bytes_cannot_be_published(self):
        target = self.root / 'model'
        entry = dict(self.entry, url='https://modelscope.cn/models/owner/model/resolve/commit/file')
        with patch.object(d, 'request_file', return_value=self.response(b'x' * len(self.payload))):
            with self.assertRaises(d.DownloadError):
                d.download(entry, target, retries=1, timeout=1, opener=None)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_suffix('.part').exists())

    def test_parallel_mirror_keeps_hashes_and_reports_failure(self):
        entries = [dict(self.entry, destination=dict(path_key='qwen_model', relative=str(i)),
                        mirrors={'modelscope': f'https://modelscope.cn/models/owner/model/resolve/commit/{i}'})
                   for i in range(4)]
        manifest = dict(sources=[dict(id='qwen3_8')], files=entries)
        observed = []
        def fetch(i, entry, total, args):
            observed.append(entry)
            return dict(path=entry['destination']['relative'], ok=i != 3)
        with patch.object(d, 'load_manifest', return_value=manifest), \
             patch.object(d, 'path', return_value=self.root), \
             patch.object(d, 'fetch_entry', side_effect=fetch), contextlib.redirect_stdout(io.StringIO()):
            code = d.main(['--only', 'qwen3_8', '--workers', '3', '--mirror', 'modelscope'])
        report = json.loads((self.root / 'last_run.json').read_text())
        self.assertEqual(code, 1)
        self.assertEqual(report['failed'], 1)
        self.assertEqual(len(report['files']), 4)
        for entry in observed:
            self.assertTrue(entry['url'].startswith('https://modelscope.cn/'))
            self.assertEqual(entry['sha256'], self.entry['sha256'])
            self.assertEqual(entry['md5'], self.entry['md5'])


if __name__ == '__main__':
    unittest.main()
