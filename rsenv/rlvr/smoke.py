"""Small live-environment acceptance run; no model or synthetic game backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rsenv.rlvr.contracts import EpisodeResult, TaskSpec
from rsenv.rlvr.records import Recorder, atomic_json
from rsenv.rlvr.rollout import ScriptedPolicy
from rsenv.rlvr.runner import collect, summary
from rsenv.rlvr.tasks import verify_harvest


def check_phase(name: str, episodes: list[EpisodeResult]) -> None:
    for episode in episodes:
        if episode.status == "infra_invalid" or episode.receipt is None:
            raise ValueError(f"{name}: {episode.reason}")
        expected = 1.0 if name == "gather" else 0.0
        if episode.reward != expected or episode.status != ("success" if expected else "failure"):
            raise ValueError(f"{name}: unexpected outcome {episode.status}/{episode.reward}")
        if name == "equipment" and not any(
            t.outcome
            and t.outcome.name == "equipItem"
            and t.outcome.status == "completed"
            and t.outcome.after.equipment != t.outcome.before.equipment
            for t in episode.transitions
        ):
            raise ValueError("equipment: axe was not equipped")


def run_smoke(task: TaskSpec, directory: Path, workers: int, seed: int) -> dict[str, Any]:
    recorder = Recorder(directory)
    started = time.monotonic()
    episodes: list[EpisodeResult] = []
    report: dict[str, Any] = {"passed": False, "backend": "bridge", "phases": {}}
    try:
        # Gather again after mutations to exercise clean initial state across episodes.
        for name, policy, count in (
            ("equipment", ScriptedPolicy(reset_probe=True), 1),
            ("gather", ScriptedPolicy(), 2),
            ("negative_control", ScriptedPolicy(fail=True), 1),
        ):
            phase_start = time.monotonic()
            batch = collect(task, policy, count, workers, seed + len(episodes) * 1000, directory, recorder)
            episodes.extend(batch)
            report["phases"][name] = summary(batch, time.monotonic() - phase_start)
            check_phase(name, batch)
            print(json.dumps({"smoke_phase": name, "passed": True}), flush=True)
        receipts = [e.receipt for e in episodes]
        if len({r.state_hash for r in receipts}) != 1:
            raise ValueError("Initial state differs between independent episodes")
        if len({r.observation.world_id for r in receipts}) != len(episodes):
            raise ValueError("Episodes reused a world")
        # Read the durable artifacts rather than only rechecking in-memory results.
        replayed = 0
        for line in (directory / "episodes.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["kind"] != "episode":
                continue
            episode = EpisodeResult.model_validate(record["data"])
            verified = verify_harvest(task, episode.receipt, [t.outcome for t in episode.transitions if t.outcome])
            if verified != episode.verification or float(verified.success) != episode.reward:
                raise ValueError("Persisted reward replay mismatch")
            replayed += 1
        if replayed != len(episodes):
            raise ValueError("Missing persisted episodes")
        report.update(passed=True, replayed_episodes=replayed, reset_fingerprint=receipts[0].state_hash)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        atomic_json(directory / "smoke.json", report)
    return report
