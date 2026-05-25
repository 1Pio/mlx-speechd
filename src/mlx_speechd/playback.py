from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from .audio import as_float32_mono


@dataclass
class PlaybackHandle:
    request_id: int
    sample_rate: int
    queue: "queue.Queue[np.ndarray | None]" = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    chunks_written: int = 0

    def push(self, audio: object) -> None:
        if not self.stop_event.is_set():
            self.queue.put(as_float32_mono(audio))

    def finish(self) -> None:
        self.queue.put(None)

    def stop(self) -> None:
        self.stop_event.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(None)


class PlaybackManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handles: dict[int, PlaybackHandle] = {}
        self._fake = os.environ.get("MSD_PLAYBACK", "").lower() == "fake"

    def start(self, request_id: int, sample_rate: int) -> PlaybackHandle:
        handle = PlaybackHandle(request_id=request_id, sample_rate=sample_rate)
        target = self._run_fake if self._fake else self._run_sounddevice
        handle.thread = threading.Thread(target=target, args=(handle,), daemon=True)
        with self._lock:
            self._handles[request_id] = handle
        handle.thread.start()
        return handle

    def push(self, request_id: int, audio: object, sample_rate: int) -> None:
        with self._lock:
            handle = self._handles.get(request_id)
            if handle is None:
                handle = self.start(request_id, sample_rate)
        handle.push(audio)

    def finish(self, request_id: int) -> None:
        with self._lock:
            handle = self._handles.get(request_id)
        if handle is not None:
            handle.finish()

    def stop(self, request_id: int | None = None) -> list[int]:
        with self._lock:
            if request_id is None:
                ids = list(self._handles)
            else:
                ids = [request_id] if request_id in self._handles else []
            handles = [self._handles[rid] for rid in ids]
        for handle in handles:
            handle.stop()
        return ids

    def active_ids(self) -> list[int]:
        with self._lock:
            return list(self._handles)

    def active_count(self) -> int:
        return len(self.active_ids())

    def _drop(self, request_id: int) -> None:
        with self._lock:
            self._handles.pop(request_id, None)

    def _run_fake(self, handle: PlaybackHandle) -> None:
        try:
            while True:
                item = handle.queue.get()
                if item is None:
                    return
                if handle.stop_event.is_set():
                    continue
                handle.chunks_written += 1
        finally:
            self._drop(handle.request_id)

    def _run_sounddevice(self, handle: PlaybackHandle) -> None:  # pragma: no cover - live smoke
        import sounddevice as sd

        try:
            with sd.OutputStream(
                samplerate=handle.sample_rate,
                channels=1,
                dtype="float32",
                latency="low",
            ) as stream:
                while True:
                    item = handle.queue.get()
                    if item is None:
                        return
                    if handle.stop_event.is_set():
                        continue
                    samples = as_float32_mono(item)
                    if samples.size:
                        stream.write(samples.reshape(-1, 1))
                        handle.chunks_written += 1
        finally:
            self._drop(handle.request_id)
