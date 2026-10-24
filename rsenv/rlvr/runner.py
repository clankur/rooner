"""Shared scripted, evaluation and training collection; no implicit fake backend."""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rsenv.rlvr.bridge import VerifiedBridgeClient
from rsenv.rlvr.contracts import EpisodeResult, InfrastructureError, TaskSpec, fingerprint
from rsenv.rlvr.learning import build_examples
from rsenv.rlvr.records import Recorder, atomic_json
from rsenv.rlvr.rollout import Policy, rollout
from rsenv.rlvr.worlds import SDK_REVISION, WorldLease

if TYPE_CHECKING:
    from rsenv.rlvr.tinker_backend import TinkerBackend


def collect(
    task: TaskSpec, policy: Policy, count: int, workers: int, seed: int, directory: Path, recorder: Recorder
) -> list[EpisodeResult]:
    if count < 1 or workers < 1:
        raise ValueError("Positive episode and worker counts required")

    clients: list[VerifiedBridgeClient] = []
    clients_lock = threading.Lock()
    cancelled = threading.Event()

    def episode(index: int) -> EpisodeResult:
        lease = WorldLease(directory / "worlds")
        client = VerifiedBridgeClient(lease, task, uuid.uuid4().hex)
        with clients_lock:
            if cancelled.is_set():
                raise InfrastructureError("collection_cancelled")
            clients.append(client)
        return rollout(client, task, policy, seed + index * 1000, recorder)

    pool = ThreadPoolExecutor(max_workers=min(count, workers))
    try:
        futures = [pool.submit(episode, i) for i in range(count)]
        return [future.result() for future in futures]
    except BaseException:
        with clients_lock:
            cancelled.set()
            owned = list(clients)
        for client in owned:
            try:
                client.stop()
            except Exception:
                pass  # Continue closing other owned worlds; preserve the original failure.
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def collect_group(
    task: TaskSpec, policy: Policy, group_size: int, workers: int, seed: int, directory: Path, recorder: Recorder
) -> list[EpisodeResult]:
    if group_size < 2:
        raise ValueError("GRPO needs at least two group members")
    for attempt in range(3):
        episodes = collect(task, policy, group_size, workers, seed + attempt * 100_000, directory, recorder)
        invalid = [e.episode_id for e in episodes if e.status == "infra_invalid"]
        recorder.append(
            "group",
            {
                "attempt": attempt,
                "episodes": [e.episode_id for e in episodes],
                "invalid": invalid,
                "accepted": not invalid,
            },
        )
        if not invalid:
            return episodes
    raise InfrastructureError("group_infrastructure_retries_exhausted")


def summary(episodes: list[EpisodeResult], wall_seconds: float) -> dict[str, Any]:
    valid = [e for e in episodes if e.status != "infra_invalid"]
    successes = sum(e.status == "success" for e in valid)

    def percentiles(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return (
            {f"p{p}": ordered[min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)] for p in (50, 95, 99)}
            if ordered
            else {}
        )

    n = len(valid)
    p = successes / n if n else 0.0
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n) if n else 0.0
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else 0.0
    states = [t.observation for e in episodes for t in e.transitions]
    tick_seconds = []
    for e in episodes:
        if e.receipt and e.transitions:
            before, after = e.receipt.observation, e.transitions[-1].observation
            if after.tick > before.tick and before.runtime and after.runtime:
                tick_seconds.append(
                    (after.runtime["wall_seconds"] - before.runtime["wall_seconds"]) / (after.tick - before.tick)
                )
    return {
        "attempted": len(episodes),
        "valid": n,
        "successes": successes,
        "success_rate": p if n else None,
        "wilson_95": [center - radius, center + radius] if n else None,
        "infra_invalid": len(episodes) - n,
        "wall_seconds": wall_seconds,
        "valid_per_hour": n * 3600 / max(wall_seconds, 1e-9),
        "successes_per_hour": successes * 3600 / max(wall_seconds, 1e-9),
        "reset_seconds": percentiles([e.reset_seconds for e in episodes]),
        "episode_seconds": percentiles([e.elapsed_seconds for e in episodes]),
        "sampling_seconds": percentiles([t.sampling_seconds for e in episodes for t in e.transitions]),
        "action_seconds": percentiles(
            [t.outcome.elapsed_seconds for e in episodes for t in e.transitions if t.outcome]
        ),
        "generated_tokens": sum(len(t.turn.output_ids) for e in episodes for t in e.transitions),
        "prompt_tokens": sum(len(t.turn.prompt_ids) for e in episodes for t in e.transitions),
        "engine_peak_rss_bytes": max((s.runtime.get("engine_rss_bytes", 0) for s in states), default=0),
        "observed_seconds_per_tick": percentiles(tick_seconds),
        "billed_cost_usd": None,
        "failure_reasons": {
            reason: sum(e.reason == reason for e in episodes) for reason in sorted({e.reason for e in episodes})
        },
    }


def train(
    backend: TinkerBackend,
    tasks: list[TaskSpec],
    directory: Path,
    workers: int = 1,
    group_size: int = 4,
    iterations: int = 25,
    seed: int = 42,
    learning_rate: float = 1e-5,
    resume: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recorder = Recorder(directory)
    manifest_path = directory / "checkpoint.json"
    iteration = resume["iteration"] if resume else 0
    updates = resume["updates"] if resume else 0
    zero_groups = resume.get("zero_groups", 0) if resume else 0
    run_id = uuid.uuid4().hex[:12]
    policy = backend.sampling_policy(f"{run_id}-initial")
    training_config = {
        "lora_rank": backend.rank,
        "clip_low": 0.8,
        "clip_high": 1.2,
        "temperature": 1.0,
        "max_tokens": 512,
        "thinking": False,
        "kl_beta": 0.0,
        "renderer": policy.renderer,
        "advantage_normalization": "population_std",
        "trajectory_weighting": "mean_generated_tokens",
    }
    if resume and (resume.get("sdk_revision") != SDK_REVISION or resume.get("training_config") != training_config):
        raise InfrastructureError("resume_configuration_mismatch")
    committed = {
        "iteration": iteration,
        "updates": updates,
        "zero_groups": zero_groups,
        "training_path": resume["training_path"] if resume else backend.save_state(f"{run_id}-initial"),
        "sampler_path": policy.version,
        "seed": seed,
        "group_size": group_size,
        "learning_rate": learning_rate,
        "model": backend.model,
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "sdk_revision": SDK_REVISION,
        "training_config": training_config,
    }
    atomic_json(manifest_path, committed)
    for iteration in range(iteration, iterations):
        task = tasks[iteration % len(tasks)]
        episodes = collect_group(task, policy, group_size, workers, seed + iteration * 1_000_000, directory, recorder)
        examples = build_examples(episodes, group_size)
        recorder.append(
            "training_batch",
            {
                "episode_ids": [e.episode_id for e in episodes],
                "group_size": group_size,
                "conversion_hash": fingerprint([asdict(e) for e in examples]),
            },
        )
        if examples:
            # Resume always forks from the last committed training state. A pending remote branch is abandoned.
            atomic_json(
                directory / "pending-update.json",
                {"iteration": iteration, "parent": committed, "episode_ids": [e.episode_id for e in episodes]},
            )
            started = time.monotonic()
            metrics = backend.update(examples, learning_rate)
            training_path = backend.save_state(f"{run_id}-update-{updates + 1}")
            policy = backend.sampling_policy(f"{run_id}-update-{updates + 1}")
            updates += 1
            zero_groups = 0
            committed.update(training_path=training_path, sampler_path=policy.version)
            recorder.append(
                "update",
                {
                    "iteration": iteration,
                    "update": updates,
                    "metrics": metrics,
                    "elapsed_seconds": time.monotonic() - started,
                    "policy_version": policy.version,
                },
            )
        else:
            zero_groups += 1
            recorder.append("skipped_group", {"iteration": iteration, "reason": "zero_reward_variance"})
        committed.update(iteration=iteration + 1, updates=updates, zero_groups=zero_groups)
        atomic_json(manifest_path, committed)
        (directory / "pending-update.json").unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "iteration": iteration + 1,
                    "updates": updates,
                    "zero_groups": zero_groups,
                    "mean_reward": statistics.mean(e.reward for e in episodes),
                }
            ),
            flush=True,
        )
        if zero_groups >= 20:
            break
    return committed
