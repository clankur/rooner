"""Per-task and per-family baseline results, separate from training selection."""

from collections import Counter
from typing import Any

from rsenv.rlvr.contracts import EpisodeResult, TaskSpec, fingerprint
from rsenv.rlvr.runner import summary


def benchmark_report(tasks: list[TaskSpec], episodes: list[EpisodeResult], wall_seconds: float) -> dict[str, Any]:
    reports = []
    for task in tasks:
        batch = [e for e in episodes if e.task_hash == fingerprint(task.model_dump(mode="json"))]
        valid = [e for e in batch if e.status != "infra_invalid"]
        successes = sum(e.reward == 1 for e in valid)
        if len(valid) != len(batch) or not valid:
            assessment = "resolve_infrastructure"
        elif successes == len(valid):
            assessment = "possible_ceiling_needs_more_repeats"
        elif successes == 0:
            assessment = "unsolved_check_feasibility_and_traces"
        else:
            assessment = "mixed_outcomes_candidate_for_training"
        reports.append(
            {
                "id": task.id,
                "family": task.family,
                "difficulty": task.difficulty,
                "split": task.split,
                "assessment": assessment,
                **summary(batch, sum(e.elapsed_seconds for e in batch)),
            }
        )
    families = {}
    for family in sorted({t.family for t in tasks}):
        selected = [r for r in reports if r["family"] == family]
        rates = [r["success_rate"] for r in selected if r["success_rate"] is not None]
        families[family] = {
            "tasks": len(selected),
            "valid_episodes": sum(r["valid"] for r in selected),
            "macro_success_rate": sum(rates) / len(rates) if rates else None,
            "infra_invalid": sum(r["infra_invalid"] for r in selected),
        }
    return {
        "overall": summary(episodes, wall_seconds),
        "tasks": reports,
        "families": families,
        "assessments": dict(Counter(r["assessment"] for r in reports)),
        "timing": "simulation_paused_between_decisions",
        "game_rng": "uncontrolled",
        "training_started": False,
    }
