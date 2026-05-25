from __future__ import annotations

DEFAULT_MODEL_ALIAS = "cv17-q8"
DEFAULT_VOICE = "Aiden"
DEFAULT_LANGUAGE = "English"
DEFAULT_INSTRUCT = "neutral, clear, natural, brisk"
DEFAULT_STREAMING_INTERVAL = 0.24
DEFAULT_TTL_SECONDS = 240
DEFAULT_FORMAT = "wav"
DEFAULT_SOCKET_PATH = "/tmp/msd.sock"

MODEL_ALIASES = {
    "cv17-q8": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "cv17-q6": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
    "cv06-q8": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
}

KNOWN_VOICES = {
    "aiden": "Aiden",
    "ryan": "Ryan",
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
}

SUPPORTED_FORMATS = {"wav", "mp3"}
