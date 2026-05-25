from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from typing import Any


class ProtocolError(RuntimeError):
    pass


def encode_message(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


def decode_line(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid json: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("message must be a JSON object")
    return value


def iter_messages(conn: socket.socket) -> Iterator[dict[str, Any]]:
    buffer = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            if buffer.strip():
                yield decode_line(buffer)
            return
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line.strip():
                yield decode_line(line)


def send_message(conn: socket.socket, data: dict[str, Any]) -> None:
    conn.sendall(encode_message(data))


def request(socket_path: str, payload: dict[str, Any], timeout: float | None = 2.0) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        if timeout is not None:
            conn.settimeout(timeout)
        conn.connect(socket_path)
        send_message(conn, payload)
        conn.shutdown(socket.SHUT_WR)
        for message in iter_messages(conn):
            return message
    raise ProtocolError("daemon closed without response")
