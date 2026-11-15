"""POSIX shared-memory segments, named by the host.

This uses CPython's ``_posixshmem`` (the module behind ``multiprocessing.shared_memory``) directly
rather than ``SharedMemory``: on Python 3.11 and 3.12 ``SharedMemory`` registers every segment with
the resource tracker, which starts a helper process and unlinks segments the host still owns when
the worker exits. The host owns every segment's lifetime (see PROTOCOL.md).
"""

from __future__ import annotations

import mmap
import os

import _posixshmem


class Segment:
    """A mapped segment. ``view`` is a writable memoryview of its first ``size`` bytes."""

    def __init__(self, name: str, size: int, create: bool) -> None:
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = _posixshmem.shm_open(name, flags, mode=0o600)
        try:
            if create:
                # A zero-length mapping isn't allowed; an empty result still gets one page.
                os.ftruncate(fd, max(size, 1))
            self._map = mmap.mmap(fd, max(size, 1))
        finally:
            os.close(fd)
        self.name = name
        self.size = size
        self.view = memoryview(self._map)[:size]

    def close(self) -> bool:
        """Unmaps the segment. Returns False if Python objects still point into it (the user kept
        a reference to the input); the mapping then stays until they're collected."""
        try:
            self.view.release()
            self._map.close()
        except BufferError:
            return False
        return True
