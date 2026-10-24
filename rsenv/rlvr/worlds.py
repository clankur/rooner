"""Cold-reset real worlds with private persistence and owned process groups."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path
from typing import IO, Any

from rsenv.rlvr.contracts import InfrastructureError, ResetSpec

SDK_REVISION = "66ed33320e719e92883d5fbfc05a0b091b54373f"
BUN_VERSION = "1.3.14"
BRIDGE_DIR = Path(__file__).resolve().parents[1] / "bridge"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise InfrastructureError(f"engine_patch_mismatch:{path.name}")
    path.write_text(source.replace(old, new))


def doctor(sdk: Path = BRIDGE_DIR / "rs-sdk") -> dict[str, Any]:
    def revision() -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(sdk), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    bun_version = subprocess.check_output(["bun", "--version"], text=True).strip() if shutil.which("bun") else None
    browser_ready = False
    if bun_version and (BRIDGE_DIR / "node_modules/puppeteer").is_dir():
        try:
            browser_ready = (
                subprocess.run(
                    [
                        "bun",
                        "-e",
                        "import p from 'puppeteer'; import {accessSync, constants} from 'node:fs'; "
                        "accessSync(p.executablePath(), constants.X_OK);",
                    ],
                    cwd=BRIDGE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    checks = {
        "bun": bun_version == BUN_VERSION,
        "sdk_revision": sdk.exists() and revision() == SDK_REVISION,
        "engine_dependencies": (sdk / "server/engine/node_modules").is_dir(),
        "gateway_dependencies": (sdk / "server/gateway/node_modules").is_dir(),
        "engine_keys": all(
            (sdk / "server/engine/data/config" / name).is_file() for name in ("public.pem", "private.pem")
        ),
        "raw_wordenc": (sdk / "server/engine/data/raw/wordenc").is_file(),
        "packed_content": (sdk / "server/engine/data/pack/server/script.dat").is_file(),
        "webclient_bundle": (sdk / "server/webclient/out/bot/client.js").is_file(),
        "browser_dependencies": (BRIDGE_DIR / "node_modules/puppeteer").is_dir(),
        "browser_executable": browser_ready,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "sdk_revision": SDK_REVISION,
        "bun_version": bun_version,
        "game_rng": "uncontrolled",
        "tick_ms": 600,
        "xp_multiplier": 1,
        "remediation": []
        if browser_ready
        else [
            "Install Chrome: (cd rsenv/bridge && bun x puppeteer browsers install chrome). "
            "If using PUPPETEER_CACHE_DIR, use the same value for installation and execution."
        ],
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class OwnedProcesses:
    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []
        self.logs: list[IO[bytes]] = []

    def start(self, command: list[str], cwd: Path, env: dict[str, str], log: Path) -> subprocess.Popen:
        stream = log.open("ab")
        self.logs.append(stream)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=stream, start_new_session=True)
        self.processes.append(process)
        return process

    def close(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in reversed(self.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        self.processes.clear()
        for stream in self.logs:
            stream.close()
        self.logs.clear()


class WorldLease:
    """One worker owns one real engine; each acquisition creates a fresh world."""

    def __init__(self, root: Path, sdk: Path = BRIDGE_DIR / "rs-sdk", startup_seconds: float = 120) -> None:
        self.root = root.resolve()
        self.sdk = sdk.resolve()
        self.startup_seconds = startup_seconds
        self.world_id = uuid.uuid4().hex
        self.processes = OwnedProcesses()
        self.env: dict[str, str] = {}
        self.workdir: Path | None = None
        self._acquired = False

    def acquire(self, episode_id: str, spec: ResetSpec) -> None:
        if self._acquired:
            raise InfrastructureError("world_already_leased")
        runtime = doctor(self.sdk)
        if not runtime["ready"] and not runtime["checks"].get("browser_executable", True):
            raise InfrastructureError(
                "browser_executable_missing:run (cd rsenv/bridge && bun x puppeteer browsers install chrome); "
                "use the same PUPPETEER_CACHE_DIR for installation and execution"
            )
        if not runtime["ready"]:
            raise InfrastructureError("runtime_not_ready:run python -m rsenv.rlvr doctor")
        self._acquired = True
        self.workdir = self.root / self.world_id / episode_id
        self.workdir.mkdir(parents=True, exist_ok=False)
        try:
            engine = self._prepare(spec)
            ports: set[int] = set()
            while len(ports) < 5:
                ports.add(free_port())
            web, gateway, control, node, management = sorted(ports)
            self.env = {
                **os.environ,
                "RLVR_EPISODE_ID": episode_id,
                "RLVR_WORLD_ID": self.world_id,
                "RLVR_USERNAME": "rlvrbot",
                "RS_BOT_USERNAME": "rlvrbot",
                "RS_BOT_PASSWORD": secrets.token_hex(10),
                "RLVR_CONTROL_TOKEN": secrets.token_hex(32),
                "RLVR_CONTROL_PORT": str(control),
                "RLVR_CONTROL_URL": f"http://127.0.0.1:{control}",
                "RLVR_FIXTURE": str(self.workdir / "fixture.json"),
                "RLVR_WEB_URL": f"http://127.0.0.1:{web}",
                "RS_GATEWAY": f"ws://127.0.0.1:{gateway}",
                "RLVR_GATEWAY_URL": f"ws://127.0.0.1:{gateway}",
                "AGENT_PORT": str(gateway),
                "WEB_PORT": str(web),
                "WEB_MANAGEMENT_PORT": str(management),
                "NODE_PORT": str(node),
                "NODE_XPRATE": "1",
                "NODE_TICKRATE": "600",
                "NODE_DEBUG_SOCKET": "true",
                "NODE_RANDOM_EVENTS": "false",
                "LOGIN_SERVER": "false",
                "FRIEND_SERVER": "false",
                "LOGGER_SERVER": "false",
                "NODE_DEBUG": "false",
                "HEADLESS": "true",
            }
            self.processes.start(
                ["bun", "run", "gateway.ts"], self.sdk / "server/gateway", self.env, self.workdir / "gateway.log"
            )
            self.processes.start(["bun", "run", "src/app.ts"], engine, self.env, self.workdir / "engine.log")
            deadline = time.monotonic() + self.startup_seconds
            request = urllib.request.Request(
                self.env["RLVR_CONTROL_URL"] + "/health",
                headers={"Authorization": "Bearer " + self.env["RLVR_CONTROL_TOKEN"]},
            )
            while time.monotonic() < deadline:
                if any(p.poll() is not None for p in self.processes.processes):
                    raise InfrastructureError(f"world_start_failed:inspect {self.workdir}")
                try:
                    with urllib.request.urlopen(request, timeout=1) as response:
                        if json.load(response).get("ready"):
                            return
                except (OSError, ValueError):
                    time.sleep(0.1)
            raise InfrastructureError("world_start_deadline")
        except BaseException:
            self.close()
            raise

    def _prepare(self, spec: ResetSpec) -> Path:
        assert self.workdir is not None
        engine = self.workdir / "server/engine"
        engine.mkdir(parents=True)
        source = self.sdk / "server/engine"
        shutil.copytree(source / "src", engine / "src")
        shutil.copy2(source / "package.json", engine / "package.json")
        shutil.copy2(source / "tsconfig.json", engine / "tsconfig.json")
        shutil.copytree(source / "tools", engine / "tools")
        for name in ("node_modules", "view", "public"):
            (engine / name).symlink_to(source / name, target_is_directory=True)
        (engine / "data").mkdir()
        for name in ("pack", "symbols", "raw"):
            (engine / "data" / name).symlink_to(source / "data" / name, target_is_directory=True)
        (engine / "data/config").mkdir()
        for name in ("public.pem", "private.pem"):
            shutil.copy2(source / "data/config" / name, engine / "data/config" / name)
        (engine / "data/players").mkdir()
        (engine / "data/config/world.json").write_text(
            json.dumps(
                {
                    "build": {"liveReload": False, "startup": False},
                    "node": {"xpRate": 1, "profile": "rlvr"},
                }
            )
        )
        (engine.parent / "webclient").symlink_to(self.sdk / "server/webclient", target_is_directory=True)
        (engine.parent / "content").symlink_to(self.sdk / "server/content", target_is_directory=True)
        (self.workdir / "fixture.json").write_text(spec.model_dump_json())
        shutil.copy2(BRIDGE_DIR / "rlvr/engine_control.ts", engine / "src/rlvr_control.ts")
        replace_once(
            engine / "src/app.ts",
            "await World.start();",
            "await World.start();\nconst { startControl } = await import('./rlvr_control');\nstartControl();",
        )
        replace_once(
            engine / "src/web/websocket.ts",
            "new WebSocket('ws://localhost:7780')",
            "new WebSocket(process.env.RLVR_GATEWAY_URL!)",
        )
        # Only the leased account may log into this private world; normal game code is unchanged.
        replace_once(
            engine / "src/server/login/LoginThread.ts",
            "reply: 4,",
            "reply: username === process.env.RLVR_USERNAME ? 0 : 4,",
        )
        replace_once(
            engine / "src/engine/entity/PlayerLoading.ts",
            "export class PlayerLoading {",
            "import { initializePlayer } from '#/rlvr_control.js';\n\nexport class PlayerLoading {",
        )
        replace_once(
            engine / "src/engine/entity/PlayerLoading.ts",
            "            return player;",
            "            initializePlayer(player);\n            return player;",
        )
        return engine

    def close(self) -> None:
        self.processes.close()
        self._acquired = False
