from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import as_float32_mono


@dataclass
class PlaybackHandle:
    request_id: int
    sample_rate: int
    queue: "queue.Queue[np.ndarray | None]" = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    ready_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    chunks_written: int = 0
    error: str | None = None

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
        self._backend_lock = threading.RLock()
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

    def prepare(self, request_id: int, sample_rate: int) -> None:
        with self._lock:
            handle = self._handles.get(request_id)
            if handle is None:
                handle = self.start(request_id, sample_rate)
        if not handle.ready_event.wait(timeout=3.0):
            raise PlaybackError("audio output did not become ready within 3 seconds")
        if handle.error:
            raise PlaybackError(handle.error)

    def push(self, request_id: int, audio: object, sample_rate: int) -> None:
        self.prepare(request_id, sample_rate)
        with self._lock:
            handle = self._handles.get(request_id)
        if handle is None:
            raise PlaybackError("audio output stopped before audio could be queued")
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
        handle.ready_event.set()
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
            with self._open_output_stream(sd, handle) as stream:
                handle.ready_event.set()
                while True:
                    item = handle.queue.get()
                    if item is None:
                        return
                    if handle.stop_event.is_set():
                        continue
                    samples = resample_linear(as_float32_mono(item), handle.sample_rate, int(stream.samplerate))
                    if samples.size:
                        stream.write(samples.reshape(-1, 1))
                        handle.chunks_written += 1
        except Exception as exc:
            handle.error = f"{type(exc).__name__}: {exc}"
            handle.ready_event.set()
        finally:
            self._drop(handle.request_id)

    def _open_output_stream(self, sd: Any, handle: PlaybackHandle) -> Any:  # pragma: no cover - live smoke
        errors: list[Exception] = []
        for attempt in range(2):
            try:
                output_rate = self._default_output_sample_rate(sd) or handle.sample_rate
                return sd.OutputStream(
                    samplerate=output_rate,
                    channels=1,
                    dtype="float32",
                    latency="low",
                )
            except Exception as exc:
                errors.append(exc)
                if attempt == 0:
                    self._reset_sounddevice(sd)
                    continue
                raise PlaybackError("; after PortAudio reset: ".join(str(error) for error in errors)) from exc
        raise PlaybackError("audio output stream could not be opened")

    @staticmethod
    def _default_output_sample_rate(sd: Any) -> int | None:  # pragma: no cover - live smoke
        try:
            device = sd.query_devices(kind="output")
            return int(float(device.get("default_samplerate") or 0)) or None
        except Exception:
            return None

    def _reset_sounddevice(self, sd: Any) -> None:  # pragma: no cover - live smoke
        with self._backend_lock:
            try:
                sd.stop()
            except Exception:
                pass
            terminate = getattr(sd, "_terminate", None)
            initialize = getattr(sd, "_initialize", None)
            if callable(terminate) and callable(initialize):
                try:
                    terminate()
                except Exception:
                    pass
                initialize()


class PlaybackError(RuntimeError):
    pass


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if samples.size == 0 or source_rate == target_rate:
        return samples
    if source_rate <= 0 or target_rate <= 0:
        return samples
    duration = samples.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.ascontiguousarray(np.interp(target_x, source_x, samples).astype(np.float32))
