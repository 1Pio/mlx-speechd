from __future__ import annotations

import os
import queue
import signal
import socket
import threading
import time
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from .constants import DEFAULT_TTL_SECONDS
from .engine import BaseEngine, build_engine
from .models import SpeechRequest
from .paths import pid_path, state_dir
from .playback import PlaybackManager
from .protocol import ProtocolError, iter_messages, send_message


@dataclass
class EngineJob:
    op: str
    request: SpeechRequest | None = None
    request_id: int | None = None
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] = field(default_factory=dict)


class SpeechDaemon:
    def __init__(self, socket_path: str, *, autostarted: bool = False, engine: BaseEngine | None = None) -> None:
        self.socket_path = socket_path
        self.autostarted = autostarted
        self.engine = engine or build_engine()
        self.playback = PlaybackManager()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._next_request_id = 1
        self._active: set[int] = set()
        self._cancelled: set[int] = set()
        self._last_activity = time.time()
        self._idle_timer: threading.Timer | None = None
        self._server_socket: socket.socket | None = None
        self._jobs: "queue.Queue[EngineJob]" = queue.Queue()

    def serve_foreground(self) -> None:
        self._install_signal_handlers()
        state_dir().mkdir(parents=True, exist_ok=True)
        pid_path().write_text(str(os.getpid()), encoding="utf-8")
        sock_path = Path(self.socket_path)
        if sock_path.exists():
            sock_path.unlink()

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            self._server_socket = server
            server.bind(self.socket_path)
            os.chmod(self.socket_path, 0o600)
            server.listen(64)
            server.settimeout(0.5)
            print(f"msd serve listening on {self.socket_path}", flush=True)
            accept_thread = threading.Thread(target=self._accept_loop, args=(server,), daemon=True)
            accept_thread.start()
            self._engine_loop()

        print("msd serve stopping", flush=True)
        self._cleanup_runtime()

    def _accept_loop(self, server: socket.socket) -> None:
        while not self._stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _engine_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run_job(job)
            except Exception as exc:
                job.response.update(ok=False, status="error", error=str(exc), trace=traceback.format_exc())
            finally:
                job.done.set()

    def _run_job(self, job: EngineJob) -> None:
        if job.op == "up":
            assert job.request is not None
            state = self.engine.warm(job.request)
            self._touch(job.request.ttl)
            job.response.update(ok=True, status="warmed", model=state.to_dict())
            return
        if job.op == "down":
            stopped = self.cancel_active()
            self.engine.unload()
            self._stop_event.set()
            self._close_server_socket()
            job.response.update(ok=True, status="down", stopped_request_ids=stopped)
            return
        if job.op == "hermes":
            assert job.request is not None and job.request_id is not None
            job.response.update(self._run_file(job.request_id, job.request))
            return
        if job.op == "say":
            assert job.request is not None and job.request_id is not None
            self._run_say(job.request_id, job.request, job.done, job.response)
            return
        raise ProtocolError(f"unknown engine job: {job.op}")

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            try:
                message = next(iter_messages(conn))
                response = self.dispatch(message)
                if response is not None:
                    send_message(conn, response)
            except StopIteration:
                return
            except Exception as exc:
                send_message(conn, {"ok": False, "error": str(exc), "trace": traceback.format_exc()})

    def dispatch(self, data: dict[str, Any]) -> dict[str, Any] | None:
        op = str(data.get("op", "")).lower()
        self._touch(int(data.get("ttl") or DEFAULT_TTL_SECONDS))
        if op == "status":
            return {"ok": True, "status": self.status()}
        if op == "stop":
            stopped = self.cancel_active()
            return {"ok": True, "status": "stopped", "stopped_request_ids": stopped}
        if op == "down":
            return self._submit(EngineJob(op="down"))
        if op == "up":
            req = SpeechRequest.from_dict(data)
            return self._submit(EngineJob(op="up", request=req))
        if op == "say":
            return self._handle_say(SpeechRequest.from_dict(data))
        if op == "hermes":
            return self._handle_file(SpeechRequest.from_dict(data))
        raise ProtocolError(f"unknown op: {op}")

    def _submit(self, job: EngineJob) -> dict[str, Any]:
        self._jobs.put(job)
        job.done.wait()
        return job.response

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = sorted(self._active)
            cancelled = sorted(self._cancelled)
        return {
            "daemon": "running",
            "socket": self.socket_path,
            "pid": os.getpid(),
            "autostarted": self.autostarted,
            "model": self.engine.state().to_dict(),
            "active_request_ids": active,
            "cancelled_request_ids": cancelled,
            "playback_active_ids": self.playback.active_ids(),
            "last_activity": self._last_activity,
        }

    def cancel_active(self) -> list[int]:
        with self._lock:
            ids = sorted(self._active)
            self._cancelled.update(ids)
        self.playback.stop()
        return ids

    def _handle_say(self, req: SpeechRequest) -> dict[str, Any]:
        request_id = self._new_request_id()
        if req.interrupt:
            self.cancel_active()
        with self._lock:
            self._active.add(request_id)
        done = threading.Event()
        job = EngineJob(op="say", request=req, request_id=request_id, done=done)
        self._jobs.put(job)
        if not req.wait:
            return {
                "ok": True,
                "request_id": request_id,
                "status": "accepted",
                "interrupt": req.interrupt,
            }
        done.wait()
        if job.response.get("ok") is False:
            return job.response
        return {
            "ok": True,
            "request_id": request_id,
            "status": job.response.get("status", "done"),
            "duration_ms": job.response.get("duration_ms"),
            "first_audio_ms": job.response.get("first_audio_ms"),
            "chunks": job.response.get("chunks", 0),
        }

    def _run_say(self, request_id: int, req: SpeechRequest, done: threading.Event, result: dict[str, Any]) -> None:
        started = time.monotonic()
        chunks = 0
        first_audio_ms: int | None = None
        try:
            for chunk in self.engine.stream(req):
                if self._is_cancelled(request_id):
                    break
                if first_audio_ms is None:
                    first_audio_ms = int((time.monotonic() - started) * 1000)
                self.playback.push(request_id, chunk.audio, chunk.sample_rate)
                chunks += 1
            self.playback.finish(request_id)
            result.update(
                ok=True,
                status="cancelled" if self._is_cancelled(request_id) else "done",
                chunks=chunks,
                first_audio_ms=first_audio_ms,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:
            result.update(ok=False, request_id=request_id, status="error", error=str(exc))
        finally:
            self._mark_complete(request_id)
            self._touch(req.ttl)
            done.set()

    def _handle_file(self, req: SpeechRequest) -> dict[str, Any]:
        request_id = self._new_request_id()
        with self._lock:
            self._active.add(request_id)
        return self._submit(EngineJob(op="hermes", request=req, request_id=request_id))

    def _run_file(self, request_id: int, req: SpeechRequest) -> dict[str, Any]:
        started = time.monotonic()
        try:
            output = self.engine.write(req)
            return {
                "ok": True,
                "request_id": request_id,
                "status": "done",
                "output": output,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "model": self.engine.state().to_dict(),
            }
        finally:
            self._mark_complete(request_id)
            self._touch(req.ttl)

    def _new_request_id(self) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id

    def _is_cancelled(self, request_id: int) -> bool:
        with self._lock:
            return request_id in self._cancelled

    def _mark_complete(self, request_id: int) -> None:
        with self._lock:
            self._active.discard(request_id)
            self._cancelled.discard(request_id)

    def _touch(self, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._last_activity = time.time()
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        if ttl <= 0:
            return
        self._idle_timer = threading.Timer(ttl, self._idle_expire)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_expire(self) -> None:
        with self._lock:
            if self._active:
                self._touch(DEFAULT_TTL_SECONDS)
                return
        self.playback.stop()
        self.engine.unload()

    def _install_signal_handlers(self) -> None:
        def stop(_signum: int, _frame: object) -> None:
            self._stop_event.set()
            self._close_server_socket()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, lambda _signum, _frame: None)

    def _close_server_socket(self) -> None:
        try:
            if self._server_socket is not None:
                self._server_socket.close()
        except OSError:
            pass

    def _cleanup_runtime(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self.playback.stop()
        self.engine.unload()
        for path in (Path(self.socket_path), pid_path()):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
