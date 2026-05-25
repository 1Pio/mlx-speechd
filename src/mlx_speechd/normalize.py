from __future__ import annotations

from .constants import MODEL_ALIASES, KNOWN_VOICES


def normalize_model(model: str | None) -> tuple[str, str | None]:
    raw = (model or "").strip()
    if not raw:
        raw = "cv17-q8"
    key = raw.lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key], key
    return raw, None


def normalize_voice(voice: str | None) -> str:
    raw = (voice or "").strip()
    if not raw:
        return "Aiden"
    key = raw.lower()
    if key in KNOWN_VOICES:
        return KNOWN_VOICES[key]
    return raw


def speed_to_instruction(instruct: str, speed: float) -> str:
    if abs(speed - 1.0) < 0.001:
        return instruct
    pacing = "speaking slower" if speed < 1.0 else "speaking faster"
    return f"{instruct}, {pacing}, approximate speed {speed:g}"
