"""Explicit live-only CLI for environment validation, evaluation and GRPO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from types import FrameType

from rsenv.rlvr.contracts import EpisodeResult, TaskSpec, fingerprint
from rsenv.rlvr.learning import build_examples
from rsenv.rlvr.records import Recorder, atomic_json
from rsenv.rlvr.rollout import ScriptedPolicy
from rsenv.rlvr.runner import collect, summary, train
from rsenv.rlvr.tasks import verify_harvest
from rsenv.rlvr.worlds import doctor


def source_identity() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        [
            *root.joinpath("rsenv").glob("*.py"),
            *root.joinpath("rsenv/rlvr").glob("*.py"),
            *root.joinpath("rsenv/bridge/rlvr").glob("*.ts"),
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["doctor", "smoke", "scripted", "evaluate", "train", "profile", "reset-check", "replay"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, help="JSON array of TaskSpec objects")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--checkpoint", help="Tinker sampler path for evaluation")
    parser.add_argument("--resume", type=Path, help="Committed training checkpoint.json")
    parser.add_argument("--records", type=Path, help="episodes.jsonl to replay")
    parser.add_argument("--fail-policy", action="store_true")
    parser.add_argument("--greedy", action="store_true", help="Evaluation only")
    args = parser.parse_args(argv)
    if args.resume and args.command != "train":
        parser.error("--resume is for training; use --checkpoint for evaluation")
    if args.checkpoint and args.command != "evaluate":
        parser.error("--checkpoint accepts sampler weights for evaluation only")
    if args.greedy and args.command != "evaluate":
        parser.error("--greedy is evaluation-only")
    if args.command == "doctor":
        report = doctor()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ready"] else 1)
    if args.workers < 1 or args.episodes < 1 or args.iterations < 1 or args.group_size < 2:
        parser.error("workers, episodes and iterations must be positive; group-size must be at least 2")
    tasks = (
        [TaskSpec.model_validate(t) for t in json.loads(args.manifest.read_text())] if args.manifest else [TaskSpec()]
    )
    if not tasks:
        parser.error("Task manifest must not be empty")
    if args.command == "smoke":
        if args.manifest is None:
            tasks = [TaskSpec(id="gather_logs_1/smoke-v1", quantity=1, max_turns=6, episode_seconds=120)]
        if len(tasks) != 1:
            parser.error("smoke requires exactly one task")
    if args.command == "replay":
        if not args.records:
            parser.error("replay requires --records")
        by_hash = {fingerprint(t.model_dump(mode="json")): t for t in tasks}
        count = 0
        saved = {}
        batches = 0
        for line in args.records.read_text().splitlines():
            record = json.loads(line)
            if record["kind"] == "training_batch":
                batch = record["data"]
                examples = build_examples([saved[e] for e in batch["episode_ids"]], batch["group_size"])
                if fingerprint([asdict(e) for e in examples]) != batch["conversion_hash"]:
                    raise ValueError("Training-mask replay mismatch")
                batches += 1
            if record["kind"] != "episode":
                continue
            episode = EpisodeResult.model_validate(record["data"])
            saved[episode.episode_id] = episode
            if episode.status == "infra_invalid":
                continue
            task = by_hash[episode.task_hash]
            verified = verify_harvest(task, episode.receipt, [t.outcome for t in episode.transitions if t.outcome])
            if verified != episode.verification or float(verified.success) != episode.reward:
                raise ValueError(f"Verifier replay mismatch: {episode.episode_id}")
            count += 1
        print(json.dumps({"verified_episodes": count, "verified_training_batches": batches}))
        return
    directory = args.output or Path("outputs/rlvr") / uuid.uuid4().hex
    if args.resume and args.output is None:
        directory = args.resume.parent
    directory.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(directory)
    if not args.resume and (directory / "run.json").exists():
        parser.error("Output directory already contains a run; choose a new directory or --resume")
    if not args.resume:
        atomic_json(
            directory / "run.json",
            {
                "backend": "bridge",
                "command": args.command,
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_diff_hash": fingerprint(subprocess.check_output(["git", "diff"], text=True)),
                "source_identity": source_identity(),
                "protocol_version": 1,
                "packages": {p: importlib.metadata.version(p) for p in ("pydantic",)},
                "tasks": [t.model_dump(mode="json") for t in tasks],
                "seed": args.seed,
                "workers": args.workers,
                "model": args.model,
                "runtime": doctor(),
                "created_unix": time.time(),
            },
        )
    backend = None
    previous_term_handler = None
    if threading.current_thread() is threading.main_thread():

        def interrupt(signum: int, frame: FrameType | None) -> None:
            raise KeyboardInterrupt("RLVR collection interrupted")

        previous_term_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.command == "smoke":
            from rsenv.rlvr.smoke import run_smoke

            print(json.dumps(run_smoke(tasks[0], directory, args.workers, args.seed), indent=2))
            return
        if args.command in ("evaluate", "train"):
            from rsenv.rlvr.tinker_backend import TinkerBackend

            resumed = json.loads(args.resume.read_text()) if args.resume else None
            if resumed:
                tasks = [TaskSpec.model_validate(t) for t in resumed["tasks"]]
                args.seed, args.group_size, args.model = resumed["seed"], resumed["group_size"], resumed["model"]
            backend = TinkerBackend(
                args.model,
                seed=args.seed,
                training=args.command == "train",
                resume_path=resumed["training_path"] if resumed else args.checkpoint,
            )
            if args.command == "train":
                result = train(
                    backend,
                    tasks,
                    directory,
                    args.workers,
                    args.group_size,
                    args.iterations,
                    args.seed,
                    learning_rate=resumed["learning_rate"] if resumed else 1e-5,
                    resume=resumed,
                )
                print(json.dumps(result, indent=2))
                return
            policy = backend.sampling_policy("evaluation")
            if args.greedy:
                policy.temperature = 0.0
        else:
            policy = ScriptedPolicy(args.fail_policy, reset_probe=args.command == "reset-check")
        for workers in [1, 2, 4] if args.command == "profile" else [args.workers]:
            started = time.monotonic()
            episodes = []
            for task_index, task in enumerate(tasks):
                episodes.extend(
                    collect(task, policy, args.episodes, workers, args.seed + task_index * 100_000, directory, recorder)
                )
            report = {"workers": workers, **summary(episodes, time.monotonic() - started)}
            atomic_json(directory / f"summary-{workers}.json", report)
            print(json.dumps(report, indent=2), flush=True)
            if any(e.status == "infra_invalid" for e in episodes):
                raise SystemExit(1)
            if args.command == "reset-check":
                if any(
                    not any(
                        t.outcome and t.outcome.status == "completed" and t.outcome.name == "equipItem"
                        for t in e.transitions
                    )
                    for e in episodes
                ):
                    raise ValueError("Reset probe did not mutate equipment")
                hashes = {e.receipt.state_hash for e in episodes}
                if len(tasks) == 1 and len(hashes) != 1:
                    raise ValueError("Reset fingerprints differ")
    finally:
        if previous_term_handler is not None:
            signal.signal(signal.SIGTERM, previous_term_handler)
        if backend:
            backend.close()


if __name__ == "__main__":
    main()
