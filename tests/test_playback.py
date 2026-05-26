import queue

import numpy as np

from mlx_speechd.playback import PlaybackError, PlaybackHandle, PlaybackManager, resample_linear


def test_finish_drains_queued_audio_before_sentinel(monkeypatch) -> None:
    monkeypatch.setenv("MSD_PLAYBACK", "fake")
    manager = PlaybackManager()
    handle = manager.start(1, 24000)

    for _ in range(20):
        handle.push(np.zeros(2400, dtype=np.float32))
    handle.finish()

    assert handle.thread is not None
    handle.thread.join(timeout=1)
    assert not handle.thread.is_alive()
    assert handle.chunks_written == 20
    assert manager.active_ids() == []


def test_natural_finish_does_not_mark_handle_stopped() -> None:
    handle = PlaybackHandle(request_id=1, sample_rate=24000)

    handle.push(np.zeros(10, dtype=np.float32))
    handle.finish()

    assert not handle.stop_event.is_set()
    assert handle.queue.get_nowait() is not None
    assert handle.queue.get_nowait() is None


def test_stop_discards_stale_queued_audio() -> None:
    handle = PlaybackHandle(request_id=1, sample_rate=24000)

    for _ in range(3):
        handle.push(np.zeros(10, dtype=np.float32))
    handle.stop()

    assert handle.stop_event.is_set()
    assert handle.queue.get_nowait() is None
    try:
        handle.queue.get_nowait()
    except queue.Empty:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("stop should leave only the wake-up sentinel queued")


def test_push_surfaces_output_stream_open_failure(monkeypatch) -> None:
    monkeypatch.delenv("MSD_PLAYBACK", raising=False)

    def fail_output(self: PlaybackManager, handle: PlaybackHandle) -> None:
        handle.error = "PortAudioError: test failure"
        handle.ready_event.set()
        self._drop(handle.request_id)

    monkeypatch.setattr(PlaybackManager, "_run_sounddevice", fail_output)
    manager = PlaybackManager()

    try:
        manager.push(1, np.zeros(10, dtype=np.float32), 24000)
    except PlaybackError as exc:
        assert "PortAudioError" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("playback open failure should be visible to the request")


def test_resample_linear_changes_sample_count() -> None:
    samples = np.linspace(-0.5, 0.5, 240, dtype=np.float32)

    resampled = resample_linear(samples, 24000, 48000)

    assert resampled.dtype == np.float32
    assert resampled.shape == (480,)


def test_open_output_stream_uses_device_default_rate_and_retries() -> None:
    manager = PlaybackManager()
    handle = PlaybackHandle(request_id=1, sample_rate=24000)

    class FakeStream:
        def __init__(self, samplerate: int, channels: int, dtype: str, latency: str) -> None:
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.latency = latency

    class FakeSoundDevice:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.stopped = False
            self.terminated = False
            self.initialized = False

        def query_devices(self, kind: str) -> dict[str, float]:
            assert kind == "output"
            return {"default_samplerate": 44100.0}

        def OutputStream(self, samplerate: int, channels: int, dtype: str, latency: str) -> FakeStream:
            self.calls.append(samplerate)
            if len(self.calls) == 1:
                raise RuntimeError("first open failed")
            return FakeStream(samplerate, channels, dtype, latency)

        def stop(self) -> None:
            self.stopped = True

        def _terminate(self) -> None:
            self.terminated = True

        def _initialize(self) -> None:
            self.initialized = True

    sd = FakeSoundDevice()

    stream = manager._open_output_stream(sd, handle)

    assert stream.samplerate == 44100
    assert sd.calls == [44100, 44100]
    assert sd.stopped is True
    assert sd.terminated is True
    assert sd.initialized is True
