from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import SUPPORTED_FORMATS


def as_float32_mono(audio: object) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1 if arr.shape[0] >= arr.shape[-1] else 0)
    if arr.size == 0:
        return arr.astype(np.float32)
    peak = float(np.max(np.abs(arr)))
    if peak > 1.5:
        arr = arr / max(peak, 1.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def write_audio_file(output: str | Path, audio: object, sample_rate: int, fmt: str = "wav") -> Path:
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format: {fmt}")

    samples = as_float32_mono(audio)
    if fmt == "wav":
        sf.write(str(target), samples, sample_rate, subtype="PCM_16", format="WAV")
        return target

    with tempfile.TemporaryDirectory(prefix="msd-audio-") as tmp:
        wav = Path(tmp) / "input.wav"
        sf.write(str(wav), samples, sample_rate, subtype="PCM_16", format="WAV")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), str(target)],
            check=True,
        )
    return target


def concatenate(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([as_float32_mono(chunk) for chunk in chunks])
