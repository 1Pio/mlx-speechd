import queue

import numpy as np

from mlx_speechd.playback import PlaybackHandle, PlaybackManager


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
