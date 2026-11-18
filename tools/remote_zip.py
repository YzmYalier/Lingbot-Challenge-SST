"""Read pinned ZIP members by verified HTTP ranges, without downloading videos.

The source revision and archive size are pinned by the caller. ZipFile verifies
each fully read member's CRC. Partial HTTP reads do NOT verify the whole archive's
SHA256; callers must not claim otherwise. Cache files stay below the supplied
submission-owned work directory.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, cache, *, timeout=60, retries=4):
        super().__init__()
        if not url.startswith("https://huggingface.co/datasets/"):
            raise ValueError("Expected a pinned Hugging Face dataset URL")
        self.url, self.size, self.position = url, int(size), 0
        self.cache = Path(cache) / hashlib.sha256(url.encode()).hexdigest()[:20]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries = timeout, retries
        self.memory = OrderedDict()
        self.downloaded = 0
        self.redirect_url = None
        self.member_prefetch = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if target < 0:
            raise ValueError("Negative seek")
        self.position = target
        return target

    def _range(self, start, length):
        key = (start, length)
        for (offset, count), data in self.memory.items():
            if offset <= start and start + length <= offset + count:
                return data[start - offset:start - offset + length]
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        target = self.cache / f"{start}-{length}.bin"
        checksum = target.with_suffix(".sha256")
        if target.is_file() and checksum.is_file():
            data = target.read_bytes()
            if len(data) == length and hashlib.sha256(data).hexdigest() == checksum.read_text().strip():
                return self._remember(key, data)
        # An adjacent member's prefetch can already cover the next ZIP header.
        # ZipFile then asks for just the compressed body. Reuse a complete
        # prefetched member on disk even though that request has a different key.
        for offset, count in self.member_prefetch.items():
            if (offset, count) == key or not (offset <= start and start + length <= offset + count):
                continue
            covered = self.cache / f"{offset}-{count}.bin"
            digest = covered.with_suffix(".sha256")
            if covered.is_file() and digest.is_file():
                data = covered.read_bytes()
                if len(data) == count and hashlib.sha256(data).hexdigest() == digest.read_text().strip():
                    self._remember((offset, count), data)
                    return data[start - offset:start - offset + length]
        end = start + length - 1
        expected = f"bytes {start}-{end}/{self.size}"
        # Distinct query strings prevent HTTP caches returning another Range's data.
        url = self.url + f"?download=true&input_range={start}-{end}"
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(self.redirect_url or url, headers={
                    "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                    "Cache-Control": "no-cache",
                    "User-Agent": "robotwin-input-preparation/1",
                })
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise OSError("Server did not honor the exact requested byte range")
                    data = response.read(length + 1)
                    self.redirect_url = response.geturl()
                if len(data) != length:
                    raise OSError("Incomplete HTTP range")
                self.downloaded += length
                temp = target.with_suffix(f".tmp-{os.getpid()}")
                temp.write_bytes(data)
                temp.replace(target)
                checksum.write_text(hashlib.sha256(data).hexdigest() + "\n")
                return self._remember(key, data)
            except (OSError, urllib.error.URLError):
                # Signed CDN URLs expire; obtain a fresh one from the pinned origin.
                self.redirect_url = None
                if attempt + 1 == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    def _remember(self, key, data):
        self.memory[key] = data
        while len(self.memory) > 8:
            self.memory.popitem(last=False)
        return data

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        # Coalesce small ZIP/HDF5 header reads; large members remain single reads.
        start = self.position
        for (offset, count), cached in self.memory.items():
            if offset <= start and start + size <= offset + count:
                self.position += size
                return cached[start - offset:start - offset + size]
        if start in self.member_prefetch:
            count = self.member_prefetch[start]
            data = self._range(start, count)[:size]
            self.position += len(data)
            return data
        if size < 65536:
            aligned = (start // 65536) * 65536
            length = min(self.size - aligned, max(65536, start + size - aligned))
            data = self._range(aligned, length)[start - aligned:start - aligned + size]
        else:
            data = self._range(start, size)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
