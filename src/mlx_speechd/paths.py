from __future__ import annotations

import os
from pathlib import Path

from .constants import DEFAULT_SOCKET_PATH


def state_dir() -> Path:
    root = os.environ.get("XDG_STATE_HOME")
    if root:
        return Path(root) / "msd"
    return Path.home() / ".local" / "state" / "msd"


def log_path() -> Path:
    return state_dir() / "msd.log"


def pid_path() -> Path:
    return state_dir() / "msd.pid"


def socket_path(explicit: str | None = None) -> str:
    return explicit or os.environ.get("MSD_SOCKET", DEFAULT_SOCKET_PATH)
