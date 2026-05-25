from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path


def run_cli(socket_path: Path, *args: str, timeout: float = 10) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MSD_ENGINE"] = "fake"
    env["MSD_PLAYBACK"] = "fake"
    return subprocess.run(
        [sys.executable, "-m", "mlx_speechd.cli", "--socket", str(socket_path), *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def wait_socket(socket_path: Path, deadline: float = 5) -> None:
    end = time.time() + deadline
    while time.time() < end:
        if socket_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(str(socket_path))
                return
            except OSError:
                pass
        time.sleep(0.05)
    raise AssertionError("daemon socket did not appear")


def test_daemon_lifecycle_say_interrupt_and_hermes(tmp_path: Path) -> None:
    # macOS keeps AF_UNIX socket paths short; pytest tmp paths are often too long.
    socket_path = Path(f"/tmp/msd-test-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")
    env = os.environ.copy()
    env["MSD_ENGINE"] = "fake"
    env["MSD_PLAYBACK"] = "fake"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_speechd.cli",
            "--socket",
            str(socket_path),
            "serve",
            "--foreground",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        wait_socket(socket_path)

        up = run_cli(socket_path, "up", "--model", "CV06-Q8", "--json")
        assert up.returncode == 0, up.stderr
        up_payload = json.loads(up.stdout)
        assert up_payload["model"]["loaded"] is True
        assert up_payload["model"]["warmed"] is True
        assert up_payload["model"]["alias"] == "cv06-q8"

        first = run_cli(socket_path, "say", "first request that should be interrupted", "--json")
        assert first.returncode == 0, first.stderr
        first_id = json.loads(first.stdout)["request_id"]

        second = run_cli(socket_path, "say", "second request wins", "--wait", "--json")
        assert second.returncode == 0, second.stderr
        second_payload = json.loads(second.stdout)
        assert second_payload["request_id"] > first_id
        assert second_payload["status"] == "done"
        assert second_payload["chunks"] > 0

        one = run_cli(socket_path, "say", "overlap one long enough to observe", "--json")
        two = run_cli(socket_path, "say", "overlap two long enough to observe", "--no-interrupt", "--json")
        assert one.returncode == 0
        assert two.returncode == 0
        time.sleep(0.04)
        status = run_cli(socket_path, "status", "--json")
        status_payload = json.loads(status.stdout)
        assert len(status_payload["status"]["active_request_ids"]) >= 1

        input_file = tmp_path / "in.txt"
        output_file = tmp_path / "hermes.wav"
        input_file.write_text("Hermes file mode.", encoding="utf-8")
        hermes = run_cli(
            socket_path,
            "hermes",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--json",
        )
        assert hermes.returncode == 0, hermes.stderr
        assert output_file.exists()

        stop = run_cli(socket_path, "stop", "--json")
        assert stop.returncode == 0

        down = run_cli(socket_path, "down", "--json")
        assert down.returncode == 0
    finally:
        if socket_path.exists():
            socket_path.unlink()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
