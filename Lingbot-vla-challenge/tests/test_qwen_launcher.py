"""Exercise the real shell launcher with an isolated CPU HTTP fake, never a GPU."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/start_qwen_gpu0.sh'


class QwenLauncherTests(unittest.TestCase):
    def test_backend_isolation_readiness_reuse_and_foreign_port(self):
        with tempfile.TemporaryDirectory(prefix='datafix-qwen-test-') as directory:
            root = Path(directory)
            fake = root / 'fake_vllm'
            fake.write_text(f'#!{sys.executable}\n' + '''import os,sys,json
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
args=sys.argv[1:]
assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
assert args[args.index('--gdn-prefill-backend')+1]=='triton'
assert json.loads(args[args.index('--limit-mm-per-prompt')+1])=={'image':2,'video':0}
assert os.environ['FLASHINFER_WORKSPACE_BASE'].startswith(str(Path.cwd()))
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{}')
HTTPServer(('127.0.0.1',int(args[args.index('--port')+1])),Handler).serve_forever()
''')
            fake.chmod(0o755)
            (root / 'model').mkdir()
            (root / 'model/config.json').write_text('{}')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            runtime = root / 'runtime'
            env = dict(os.environ, DATAFIX_PYTHON=sys.executable, DATAFIX_VLLM=str(fake),
                       QWEN_MODEL=str(root / 'model'), DATAFIX_QWEN_WORK=str(runtime),
                       QWEN_PORT=str(port), QWEN_START_TIMEOUT='30', QWEN_GDN_PREFILL_BACKEND='triton')
            pid = None
            try:
                dry = subprocess.run(['bash', str(SCRIPT), '--dry-run'], env=env, text=True, capture_output=True)
                self.assertEqual(dry.returncode, 0, dry.stderr)
                self.assertIn('--gdn-prefill-backend triton', dry.stdout)
                self.assertFalse(runtime.exists())
                first = subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True, timeout=40)
                pid = int((runtime / f'qwen_{port}.pid').read_text())
                self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                again = subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True, timeout=10)
                self.assertEqual(again.returncode, 0, again.stderr)
                self.assertIn('Reusing healthy', again.stdout)
                self.assertEqual(int((runtime / f'qwen_{port}.pid').read_text()), pid)
                foreign = subprocess.run(['bash', str(SCRIPT)],
                         env=dict(env, DATAFIX_QWEN_WORK=str(root / 'different-runtime')),
                         text=True, capture_output=True, timeout=10)
                self.assertNotEqual(foreign.returncode, 0)
                self.assertIn('unavailable', foreign.stderr)
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                    self.assertEqual(response.status, 200)
            finally:
                if pid is not None:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
