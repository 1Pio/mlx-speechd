from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .constants import (
    DEFAULT_FORMAT,
    DEFAULT_INSTRUCT,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SOCKET_PATH,
    DEFAULT_STREAMING_INTERVAL,
    DEFAULT_TTL_SECONDS,
    DEFAULT_VOICE,
    SUPPORTED_FORMATS,
)
from .daemon import SpeechDaemon
from .engine import build_engine
from .models import SpeechRequest
from .paths import log_path, socket_path, state_dir
from .protocol import ProtocolError, request as socket_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msd", description="MLX Speech Daemon")
    parser.add_argument("--version", action="version", version=f"msd {__version__}")
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH, help="Unix socket path")
    sub = parser.add_subparsers(dest="command", required=True)

    say = sub.add_parser("say", help="stream and play text through the warm daemon")
    add_text_args(say)
    add_tts_args(say)
    say.add_argument("--no-interrupt", action="store_true", help="allow overlap instead of cancelling active playback")
    say.add_argument("--wait", action="store_true", help="block until the request completes")
    say.add_argument("--json", action="store_true", help="print JSON response")

    render = sub.add_parser("render", help="cold-render a complete file without keeping a daemon warm")
    add_text_args(render, positional=False)
    add_tts_args(render)
    render.add_argument("--output", "-o", required=True, help="output audio path")
    render.add_argument("--format", choices=sorted(SUPPORTED_FORMATS), default=DEFAULT_FORMAT)
    render.add_argument("--json", action="store_true")

    hermes = sub.add_parser("hermes", help="warm file-in/file-out mode with no playback")
    hermes.add_argument("--input", "-i", required=True, help="text input file")
    hermes.add_argument("--output", "-o", required=True, help="output audio path")
    add_tts_args(hermes)
    hermes.add_argument("--format", choices=sorted(SUPPORTED_FORMATS), default=DEFAULT_FORMAT)
    hermes.add_argument("--json", action="store_true")

    up = sub.add_parser("up", help="start daemon, load model, and run hidden warm synthesis")
    add_tts_args(up)
    up.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="show daemon and model state")
    status.add_argument("--json", action="store_true")

    stop = sub.add_parser("stop", help="interrupt active playback without stopping the daemon")
    stop.add_argument("--json", action="store_true")

    down = sub.add_parser("down", help="stop daemon and unload model")
    down.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the daemon")
    serve.add_argument("--foreground", action="store_true", help="run visibly in the current terminal")
    serve.add_argument("--autostarted", action="store_true", help=argparse.SUPPRESS)

    return parser


def add_text_args(parser: argparse.ArgumentParser, *, positional: bool = True) -> None:
    if positional:
        parser.add_argument("text_arg", nargs="?", help="text to speak")
    parser.add_argument("--text", help="text to speak")


def add_tts_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL_ALIAS, help="model alias or Hugging Face model id")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="CustomVoice speaker name")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT, help="natural-language speaking style/pacing")
    parser.add_argument("--speed", type=float, default=1.0, help="best-effort numeric pacing hint")
    parser.add_argument("--streaming-interval", type=float, default=DEFAULT_STREAMING_INTERVAL)
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL_SECONDS, help="idle unload/exit timeout in seconds")
    parser.add_argument("--overlap-load", action="store_true", help="load a new model before unloading the old one")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sock = socket_path(args.socket)

    try:
        if args.command == "serve":
            if not args.foreground:
                parser.error("V1 supports daemon execution through 'msd serve --foreground'")
            SpeechDaemon(sock, autostarted=args.autostarted).serve_foreground()
            return 0
        if args.command == "say":
            text = resolve_text(args)
            payload = tts_payload("say", args, text=text, interrupt=not args.no_interrupt, wait=args.wait)
            response = send_with_autostart(sock, payload, timeout=None if args.wait else 3.0)
            print_response(response, json_mode=args.json, terse_ok="accepted")
            return 0 if response.get("ok") else 1
        if args.command == "up":
            payload = tts_payload("up", args, text="Ready.")
            response = send_with_autostart(sock, payload, timeout=None)
            print_response(response, json_mode=args.json, terse_ok="warmed")
            return 0 if response.get("ok") else 1
        if args.command == "hermes":
            text = Path(args.input).expanduser().read_text(encoding="utf-8")
            payload = tts_payload("hermes", args, text=text, output=args.output, fmt=args.format, wait=True)
            response = send_with_autostart(sock, payload, timeout=None)
            print_response(response, json_mode=args.json, terse_ok=response.get("output"))
            return 0 if response.get("ok") else 1
        if args.command == "render":
            text = resolve_text(args)
            payload = SpeechRequest.from_dict(
                tts_payload("render", args, text=text, output=args.output, fmt=args.format)
            )
            engine = build_engine()
            try:
                output = engine.write(payload)
            finally:
                engine.unload()
            response = {"ok": True, "status": "done", "output": output}
            print_response(response, json_mode=args.json, terse_ok=output)
            return 0
        if args.command == "status":
            response = send_without_autostart(sock, {"op": "status"})
            print_status(response, json_mode=args.json)
            return 0 if response.get("ok", True) else 1
        if args.command == "stop":
            response = send_without_autostart(sock, {"op": "stop"})
            print_response(response, json_mode=args.json, terse_ok=response.get("status", "not_running"))
            return 0 if response.get("ok", True) else 1
        if args.command == "down":
            response = send_without_autostart(sock, {"op": "down"})
            print_response(response, json_mode=args.json, terse_ok=response.get("status", "not_running"))
            return 0 if response.get("ok", True) else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"msd: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unhandled command: {args.command}")
    return 2


def resolve_text(args: argparse.Namespace) -> str:
    text = args.text if args.text is not None else getattr(args, "text_arg", None)
    if not text:
        raise ValueError("text is required; pass positional text or --text")
    return text


def tts_payload(
    op: str,
    args: argparse.Namespace,
    *,
    text: str,
    interrupt: bool = True,
    wait: bool = True,
    output: str | None = None,
    fmt: str = DEFAULT_FORMAT,
) -> dict[str, Any]:
    return SpeechRequest(
        op=op,
        text=text,
        model=args.model,
        voice=args.voice,
        language=args.language,
        instruct=args.instruct,
        speed=args.speed,
        streaming_interval=args.streaming_interval,
        interrupt=interrupt,
        wait=wait,
        output=output,
        format=fmt,
        ttl=args.ttl,
        overlap_load=args.overlap_load,
    ).to_dict()


def send_without_autostart(sock: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return socket_request(sock, payload, timeout=2.0)
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError, ProtocolError):
        return {"ok": True, "status": "not_running", "daemon": "stopped"}


def send_with_autostart(sock: str, payload: dict[str, Any], *, timeout: float | None) -> dict[str, Any]:
    try:
        return socket_request(sock, payload, timeout=timeout)
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        start_daemon(sock)
    return socket_request(sock, payload, timeout=timeout)


def start_daemon(sock: str) -> None:
    stale = Path(sock)
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass
    state_dir().mkdir(parents=True, exist_ok=True)
    log = log_path().open("ab")
    subprocess.Popen(
        [sys.executable, "-m", "mlx_speechd.cli", "--socket", sock, "serve", "--foreground", "--autostarted"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    deadline = time.time() + 15
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            socket_request(sock, {"op": "status"}, timeout=1.0)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"daemon did not start; see {log_path()} ({last_error})")


def print_response(response: dict[str, Any], *, json_mode: bool, terse_ok: object | None) -> None:
    if json_mode:
        print(json.dumps(response, indent=2, sort_keys=True))
    elif response.get("ok"):
        print(terse_ok if terse_ok is not None else response.get("status", "ok"))
    else:
        print(f"error: {response.get('error', response)}", file=sys.stderr)


def print_status(response: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(response, indent=2, sort_keys=True))
        return
    if response.get("daemon") == "stopped" or response.get("status") == "not_running":
        print("daemon: stopped")
        return
    status = response.get("status", {})
    model = status.get("model", {})
    print(f"daemon: {status.get('daemon', 'unknown')}")
    print(f"socket: {status.get('socket')}")
    print(f"model: {model.get('alias') or model.get('model_id') or 'unloaded'}")
    print(f"loaded: {model.get('loaded', False)}")
    print(f"warmed: {model.get('warmed', False)}")
    active = status.get("active_request_ids") or []
    print(f"active: {','.join(map(str, active)) if active else 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
