"""JSON-line transport for isolated live episodes."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from contextlib import suppress
from typing import Any

from rsenv.rlvr.contracts import ActionOutcome, InfrastructureError, Observation, TaskSpec
from rsenv.rlvr.tasks import verify_reset
from rsenv.rlvr.worlds import BRIDGE_DIR, WorldLease


class VerifiedBridgeClient:
    """Cold reset is owned by the lease; protocol errors poison the entire episode."""

    def __init__(self, lease: WorldLease, task: TaskSpec, episode_id: str) -> None:
        self.lease = lease
        self.task = task
        self.episode_id = episode_id
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self.responses: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=32)
        self.request_number = 0
        self.observation: Observation | None = None
        self.poisoned = False
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()

    def _reader(self) -> None:
        assert self._process and self._process.stdout
        try:
            for line in self._process.stdout:
                if len(line) > 2_000_000:
                    raise ValueError("oversized response")
                self.responses.put_nowait(json.loads(line))
        except (ValueError, queue.Full):
            pass
        finally:
            try:
                self.responses.put_nowait(None)
            except queue.Full:
                pass

    def _receive(self, request_id: str, timeout: float) -> dict[str, Any]:
        try:
            response = self.responses.get(timeout=timeout)
            if not isinstance(response, dict) or any(
                response.get(k) != v
                for k, v in {
                    "version": 1,
                    "episode_id": self.episode_id,
                    "world_id": self.lease.world_id,
                    "request_id": request_id,
                }.items()
            ):
                raise ValueError("invalid response")
            if response.get("status") == "infra_error":
                self.poisoned = True
                raise InfrastructureError(
                    "bridge_startup_failed" if request_id == "0" else "bridge_action_outcome_unknown"
                )
            if response.get("status") != ("ready" if request_id == "0" else "ok"):
                raise ValueError("invalid response status")
            return response
        except queue.Empty as exc:
            self.poisoned = True
            raise InfrastructureError("bridge_response_deadline") from exc
        except ValueError as exc:
            self.poisoned = True
            raise InfrastructureError("bridge_invalid_response") from exc

    def start(self) -> Observation:
        with self._lifecycle_lock:
            return self._start()

    def _start(self) -> Observation:
        started = time.monotonic()
        self.lease.acquire(self.episode_id, self.task.reset)
        assert self.lease.workdir is not None
        with (self.lease.workdir / "bridge.log").open("w") as diagnostics:
            self._process = subprocess.Popen(
                ["bun", "run", "rlvr/executor.ts"],
                cwd=BRIDGE_DIR,
                env=self.lease.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        remaining = self.lease.startup_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise InfrastructureError("startup_deadline")
        response = self._receive("0", remaining)
        self.observation = Observation.model_validate(response["state"])
        verify_reset(self.task, self.observation)
        return self.observation

    def rpc(self, operation: str, timeout: float, **arguments: Any) -> Any:
        with self._lock:
            if self.poisoned or self._process is None or self._process.poll() is not None:
                raise InfrastructureError("bridge_unavailable")
            self.request_number += 1
            request_id = str(self.request_number)
            command = {
                "version": 1,
                "episode_id": self.episode_id,
                "world_id": self.lease.world_id,
                "request_id": request_id,
                "operation": operation,
                "timeout_ms": min(30000, int(timeout * 1000)),
                **arguments,
            }
            payload = json.dumps(command) + "\n"
            if len(payload) > 4096:
                raise InfrastructureError("oversized_request")
            assert self._process and self._process.stdin

            # A daemon writer allows the caller's deadline to include a full/hung stdin pipe.
            def write() -> None:
                try:
                    self._process.stdin.write(payload)
                    self._process.stdin.flush()
                except (OSError, ValueError):
                    try:
                        self.responses.put_nowait(None)
                    except queue.Full:
                        pass

            threading.Thread(target=write, daemon=True).start()
            return self._receive(request_id, timeout + 0.5)["result"]

    def get_state(self) -> Observation:
        current = Observation.model_validate(self.rpc("state", 5))
        if self.observation and current.sequence <= self.observation.sequence:
            raise InfrastructureError("stale_observation")
        self.observation = current
        return current

    def act(self, name: str, arguments: dict[str, Any], timeout: float) -> ActionOutcome:
        if name not in self.task.allowed_tools:
            raise ValueError("Tool not enabled")
        raw = self.rpc("action", timeout, name=name, arguments=arguments)
        result = ActionOutcome.model_validate({**raw, "request_id": str(self.request_number)})
        if result.name != name:
            self.poisoned = True
            raise InfrastructureError("bridge_action_name_mismatch")
        self.observation = result.after
        return result

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop()

    def _stop(self) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            assert self._reader_thread is not None
            self._reader_thread.join(timeout=1)
            # A descendant can retain stdout. Never block cleanup on the reader's IO lock.
            for stream in (process.stdin, process.stdout if not self._reader_thread.is_alive() else None):
                if stream:
                    stream.close()
            self._process = None
        self.lease.close()
