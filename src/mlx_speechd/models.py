from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import (
    DEFAULT_FORMAT,
    DEFAULT_INSTRUCT,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_STREAMING_INTERVAL,
    DEFAULT_TTL_SECONDS,
    DEFAULT_VOICE,
)


@dataclass(slots=True)
class SpeechRequest:
    op: str
    text: str = ""
    method: str = "custom_voice"
    model: str = DEFAULT_MODEL_ALIAS
    voice: str = DEFAULT_VOICE
    language: str = DEFAULT_LANGUAGE
    instruct: str = DEFAULT_INSTRUCT
    speed: float = 1.0
    streaming_interval: float = DEFAULT_STREAMING_INTERVAL
    interrupt: bool = True
    wait: bool = False
    output: str | None = None
    format: str = DEFAULT_FORMAT
    ttl: int = DEFAULT_TTL_SECONDS
    overlap_load: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechRequest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "text": self.text,
            "method": self.method,
            "model": self.model,
            "voice": self.voice,
            "language": self.language,
            "instruct": self.instruct,
            "speed": self.speed,
            "streaming_interval": self.streaming_interval,
            "interrupt": self.interrupt,
            "wait": self.wait,
            "output": self.output,
            "format": self.format,
            "ttl": self.ttl,
            "overlap_load": self.overlap_load,
        }


@dataclass(slots=True)
class ModelState:
    model_id: str | None = None
    alias: str | None = None
    loaded: bool = False
    warmed: bool = False
    loading: bool = False
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "alias": self.alias,
            "loaded": self.loaded,
            "warmed": self.warmed,
            "loading": self.loading,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class AudioChunk:
    audio: Any
    sample_rate: int
    final: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
