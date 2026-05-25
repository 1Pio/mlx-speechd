from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np

from mlx_speechd.daemon import SpeechDaemon
from mlx_speechd.engine import BaseEngine
from mlx_speechd.models import AudioChunk, ModelState, SpeechRequest
from mlx_speechd.playback import PlaybackError, PlaybackManager


class SlowWarmEngine(BaseEngine):
    def __init__(self) -> None:
        self.warm_started = threading.Event()
        self.warm_done = threading.Event()
        self.unloaded_during_warm = False
        self.unload_count = 0
        self._state = ModelState(alias="cv06-q8", loaded=False, warmed=False)

    def state(self) -> ModelState:
        return self._state

    def load(self, model: str, *, overlap_load: bool = False) -> ModelState:
        self._state = ModelState(alias=model, loaded=True, warmed=False)
        return self._state

    def unload(self) -> None:
        if self.warm_started.is_set() and not self.warm_done.is_set():
            self.unloaded_during_warm = True
        self.unload_count += 1
        self._state = ModelState()

    def warm(self, request: SpeechRequest) -> ModelState:
        self.warm_started.set()
        time.sleep(1.2)
        self.warm_done.set()
        self._state = ModelState(alias=request.model, loaded=True, warmed=True)
        return self._state

    def stream(self, request: SpeechRequest, *, hidden: bool = False) -> Iterator[AudioChunk]:
        yield AudioChunk(audio=np.zeros(10, dtype=np.float32), sample_rate=24000)


class BlockingStreamEngine(BaseEngine):
    def __init__(self) -> None:
        self.release_first = threading.Event()
        self.first_started = threading.Event()
        self.streamed_texts: list[str] = []
        self._state = ModelState(alias="cv06-q8", loaded=True, warmed=True)

    def state(self) -> ModelState:
        return self._state

    def load(self, model: str, *, overlap_load: bool = False) -> ModelState:
        return self._state

    def unload(self) -> None:
        self._state = ModelState()

    def stream(self, request: SpeechRequest, *, hidden: bool = False) -> Iterator[AudioChunk]:
        self.streamed_texts.append(request.text)
        if request.text == "first":
            self.first_started.set()
            assert self.release_first.wait(timeout=2)
        yield AudioChunk(audio=np.zeros(10, dtype=np.float32), sample_rate=24000)


def run_engine_loop(daemon: SpeechDaemon) -> threading.Thread:
    thread = threading.Thread(target=daemon._engine_loop, daemon=True)
    thread.start()
    return thread


def stop_engine_loop(daemon: SpeechDaemon, thread: threading.Thread) -> None:
    daemon._stop_event.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_idle_unload_is_serialized_after_slow_warm() -> None:
    engine = SlowWarmEngine()
    daemon = SpeechDaemon("/tmp/msd-test-unused.sock", engine=engine)
    loop_thread = run_engine_loop(daemon)
    response: dict[str, object] = {}

    dispatch_thread = threading.Thread(
        target=lambda: response.update(daemon.dispatch({"op": "up", "text": "Ready.", "model": "cv06-q8", "ttl": 1}))
    )
    dispatch_thread.start()
    assert engine.warm_started.wait(timeout=1)
    dispatch_thread.join(timeout=3)

    try:
        assert response["ok"] is True
        assert engine.unloaded_during_warm is False

        deadline = time.time() + 3
        while time.time() < deadline and engine.unload_count == 0:
            time.sleep(0.05)
        assert engine.unload_count == 1
    finally:
        stop_engine_loop(daemon, loop_thread)


def test_cancelled_queued_say_does_not_stream(monkeypatch) -> None:
    monkeypatch.setenv("MSD_PLAYBACK", "fake")
    engine = BlockingStreamEngine()
    daemon = SpeechDaemon("/tmp/msd-test-unused.sock", engine=engine)
    loop_thread = run_engine_loop(daemon)

    try:
        first_response: dict[str, object] = {}
        first_thread = threading.Thread(
            target=lambda: first_response.update(
                daemon._handle_say(SpeechRequest(op="say", text="first", wait=False, ttl=0))
            )
        )
        first_thread.start()
        assert engine.first_started.wait(timeout=1)

        second_response: dict[str, object] = {}
        second_thread = threading.Thread(
            target=lambda: second_response.update(
                daemon._handle_say(SpeechRequest(op="say", text="second", wait=False, ttl=0))
            )
        )
        second_thread.start()
        deadline = time.time() + 1
        while time.time() < deadline and 2 not in daemon.status()["active_request_ids"]:
            time.sleep(0.01)
        assert 2 in daemon.status()["active_request_ids"]

        third_response: dict[str, object] = {}
        third_thread = threading.Thread(
            target=lambda: third_response.update(
                daemon._handle_say(SpeechRequest(op="say", text="third", wait=True, ttl=0))
            )
        )
        third_thread.start()
        engine.release_first.set()
        first_thread.join(timeout=3)
        second_thread.join(timeout=3)
        third_thread.join(timeout=3)

        assert first_response["status"] == "cancelled"
        assert second_response["status"] == "cancelled"
        assert third_response["status"] == "done"
        assert engine.streamed_texts == ["first", "third"]
    finally:
        stop_engine_loop(daemon, loop_thread)


def test_say_reports_playback_prepare_failure(monkeypatch) -> None:
    engine = BlockingStreamEngine()
    daemon = SpeechDaemon("/tmp/msd-test-unused.sock", engine=engine)

    def fail_prepare(self: PlaybackManager, request_id: int, sample_rate: int) -> None:
        raise PlaybackError("PortAudioError: output unavailable")

    monkeypatch.setattr(PlaybackManager, "prepare", fail_prepare)

    response = daemon._handle_say(SpeechRequest(op="say", text="hello", wait=False, ttl=0))

    assert response["ok"] is False
    assert response["status"] == "error"
    assert "PortAudioError" in response["error"]
    assert daemon.status()["active_request_ids"] == []
    assert daemon.status()["queued_engine_jobs"] == 0
