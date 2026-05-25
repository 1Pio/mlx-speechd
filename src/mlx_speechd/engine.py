from __future__ import annotations

import gc
import math
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import as_float32_mono, concatenate, write_audio_file
from .constants import DEFAULT_INSTRUCT, DEFAULT_STREAMING_INTERVAL
from .models import AudioChunk, ModelState, SpeechRequest
from .normalize import normalize_model, normalize_voice, speed_to_instruction

# Xet transfers are fast when healthy, but on this Mac they have produced repeated
# incomplete TLS reads during first model fetches. Prefer the standard HF path.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class EngineError(RuntimeError):
    pass


@dataclass(slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int


class BaseEngine:
    def state(self) -> ModelState:
        raise NotImplementedError

    def load(self, model: str, *, overlap_load: bool = False) -> ModelState:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def warm(self, request: SpeechRequest) -> ModelState:
        for _ in self.stream(request, hidden=True):
            pass
        return self.state()

    def stream(self, request: SpeechRequest, *, hidden: bool = False) -> Iterator[AudioChunk]:
        raise NotImplementedError

    def synthesize(self, request: SpeechRequest) -> SynthesisResult:
        chunks: list[np.ndarray] = []
        sample_rate = 24000
        for chunk in self.stream(request, hidden=True):
            sample_rate = chunk.sample_rate
            if np.asarray(chunk.audio).size:
                chunks.append(as_float32_mono(chunk.audio))
        return SynthesisResult(audio=concatenate(chunks), sample_rate=sample_rate)

    def write(self, request: SpeechRequest) -> str:
        if not request.output:
            raise EngineError("output path is required")
        result = self.synthesize(request)
        return str(write_audio_file(request.output, result.audio, result.sample_rate, request.format))


class MlxAudioEngine(BaseEngine):
    def __init__(self) -> None:
        self._model: Any | None = None
        self._state = ModelState()
        self._sample_rate = 24000

    def state(self) -> ModelState:
        return ModelState(
            model_id=self._state.model_id,
            alias=self._state.alias,
            loaded=self._state.loaded,
            warmed=self._state.warmed,
            loading=self._state.loading,
            last_error=self._state.last_error,
        )

    def load(self, model: str, *, overlap_load: bool = False) -> ModelState:
        model_id, alias = normalize_model(model)
        if self._state.loaded and self._state.model_id == model_id:
            return self.state()

        if not overlap_load:
            self.unload()

        self._state = ModelState(model_id=model_id, alias=alias, loading=True)
        try:
            from mlx_audio.tts.utils import load_model

            new_model = load_model(model_id)
            if overlap_load and self._model is not None:
                old_model = self._model
                self._model = new_model
                del old_model
                self._clear_mlx_cache()
            else:
                self._model = new_model
            self._sample_rate = int(getattr(new_model, "sample_rate", 24000) or 24000)
            self._state = ModelState(model_id=model_id, alias=alias, loaded=True, warmed=False)
            return self.state()
        except Exception as exc:  # pragma: no cover - exercised in live smoke tests
            self._state.loading = False
            self._state.last_error = str(exc)
            raise

    def unload(self) -> None:
        if self._model is not None:
            model = self._model
            self._model = None
            del model
        gc.collect()
        self._clear_mlx_cache()
        self._state = ModelState()

    def warm(self, request: SpeechRequest) -> ModelState:
        self.load(request.model, overlap_load=request.overlap_load)
        hidden = SpeechRequest(
            op="warm",
            text="Ready.",
            model=request.model,
            voice=request.voice,
            language=request.language,
            instruct=request.instruct or DEFAULT_INSTRUCT,
            speed=request.speed,
            streaming_interval=request.streaming_interval,
            ttl=request.ttl,
            overlap_load=request.overlap_load,
        )
        for _ in self.stream(hidden, hidden=True):
            pass
        self._state.warmed = True
        return self.state()

    def stream(self, request: SpeechRequest, *, hidden: bool = False) -> Iterator[AudioChunk]:
        if request.method != "custom_voice":
            raise EngineError(f"unsupported method for V1: {request.method}")
        self.load(request.model, overlap_load=request.overlap_load)
        if self._model is None:
            raise EngineError("model is not loaded")

        speaker = normalize_voice(request.voice)
        instruct = speed_to_instruction(request.instruct or DEFAULT_INSTRUCT, request.speed)
        kwargs = {
            "text": request.text,
            "speaker": speaker,
            "language": request.language,
            "instruct": instruct,
            "stream": True,
            "streaming_interval": request.streaming_interval or DEFAULT_STREAMING_INTERVAL,
        }

        try:
            generator = self._model.generate_custom_voice(**kwargs)
        except TypeError:
            # Older examples sometimes use lang_code. Keep this fallback narrow.
            kwargs["lang_code"] = kwargs.pop("language")
            generator = self._model.generate_custom_voice(**kwargs)

        yielded = False
        for result in self._iter_results(generator):
            audio = as_float32_mono(getattr(result, "audio", result))
            sample_rate = int(getattr(result, "sample_rate", self._sample_rate) or self._sample_rate)
            yielded = True
            yield AudioChunk(audio=audio, sample_rate=sample_rate)

        if yielded:
            self._state.warmed = True

    def synthesize(self, request: SpeechRequest) -> SynthesisResult:
        if request.method != "custom_voice":
            raise EngineError(f"unsupported method for V1: {request.method}")
        self.load(request.model, overlap_load=request.overlap_load)
        if self._model is None:
            raise EngineError("model is not loaded")

        speaker = normalize_voice(request.voice)
        instruct = speed_to_instruction(request.instruct or DEFAULT_INSTRUCT, request.speed)
        kwargs = {
            "text": request.text,
            "speaker": speaker,
            "language": request.language,
            "instruct": instruct,
            "stream": False,
        }
        try:
            result = self._model.generate_custom_voice(**kwargs)
        except TypeError:
            kwargs["lang_code"] = kwargs.pop("language")
            result = self._model.generate_custom_voice(**kwargs)

        chunks: list[np.ndarray] = []
        sample_rate = self._sample_rate
        for item in self._iter_results(result):
            chunks.append(as_float32_mono(getattr(item, "audio", item)))
            sample_rate = int(getattr(item, "sample_rate", sample_rate) or sample_rate)
        self._state.warmed = True
        return SynthesisResult(audio=concatenate(chunks), sample_rate=sample_rate)

    @staticmethod
    def _iter_results(value: Any) -> Iterable[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return value
        if hasattr(value, "__iter__") and not isinstance(value, np.ndarray):
            return value
        return [value]

    @staticmethod
    def _clear_mlx_cache() -> None:
        try:
            import mlx.core as mx

            clear = getattr(mx, "clear_cache", None)
            if callable(clear):
                clear()
        except Exception:
            return


class FakeEngine(BaseEngine):
    def __init__(self) -> None:
        self._state = ModelState()

    def state(self) -> ModelState:
        return ModelState(
            model_id=self._state.model_id,
            alias=self._state.alias,
            loaded=self._state.loaded,
            warmed=self._state.warmed,
            loading=self._state.loading,
            last_error=self._state.last_error,
        )

    def load(self, model: str, *, overlap_load: bool = False) -> ModelState:
        model_id, alias = normalize_model(model)
        self._state = ModelState(model_id=model_id, alias=alias, loaded=True, warmed=self._state.warmed)
        return self.state()

    def unload(self) -> None:
        self._state = ModelState()

    def warm(self, request: SpeechRequest) -> ModelState:
        self.load(request.model)
        time.sleep(0.02)
        self._state.warmed = True
        return self.state()

    def stream(self, request: SpeechRequest, *, hidden: bool = False) -> Iterator[AudioChunk]:
        self.load(request.model)
        sr = 24000
        chunks = max(2, min(20, math.ceil(len(request.text or "x") / 8)))
        for index in range(chunks):
            time.sleep(0.05)
            t = np.linspace(0, 0.03, int(sr * 0.03), endpoint=False)
            wave = np.sin(2 * np.pi * (330 + index * 15) * t).astype(np.float32) * 0.02
            yield AudioChunk(audio=wave, sample_rate=sr, final=index == chunks - 1)
        self._state.warmed = True


def build_engine() -> BaseEngine:
    if os.environ.get("MSD_ENGINE", "").lower() == "fake":
        return FakeEngine()
    return MlxAudioEngine()
