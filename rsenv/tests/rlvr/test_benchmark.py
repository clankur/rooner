from __future__ import annotations

from rsenv.rlvr.benchmark import benchmark_report
from rsenv.rlvr.contracts import EpisodeResult, TaskSpec, fingerprint


def result(task: TaskSpec, success: bool, index: int, invalid: bool = False) -> EpisodeResult:
    return EpisodeResult(
        episode_id=str(index),
        task_hash=fingerprint(task.model_dump(mode="json")),
        policy_version="base",
        seed=index,
        status="infra_invalid" if invalid else "success" if success else "failure",
        reason="transport" if invalid else "goal_complete" if success else "turn_limit",
        reward=None if invalid else float(success),
        receipt=None,
        transitions=(),
        verification=None,
        elapsed_seconds=1,
        reset_seconds=0,
    )


def test_family_score_weights_tasks_not_episode_counts() -> None:
    first = TaskSpec(id="first", family="gather")
    second = TaskSpec(id="second", family="gather")
    episodes = [result(first, True, i) for i in range(3)] + [result(second, False, 4)]
    report = benchmark_report([first, second], episodes, 2)
    assert report["overall"]["success_rate"] == 0.75
    assert report["families"]["gather"]["macro_success_rate"] == 0.5
    assert report["overall"]["wall_seconds"] == 2


def test_infrastructure_is_not_scored_as_gameplay_failure() -> None:
    task = TaskSpec()
    report = benchmark_report([task], [result(task, True, 0), result(task, False, 1, invalid=True)], 2)
    row = report["tasks"][0]
    assert row["success_rate"] == 1 and row["valid"] == 1 and row["infra_invalid"] == 1
    assert row["assessment"] == "resolve_infrastructure"
    assert report["training_started"] is False


def test_mixed_outcomes_are_candidates_not_proof_of_learning() -> None:
    task = TaskSpec()
    report = benchmark_report([task], [result(task, True, 0), result(task, False, 1)], 2)
    assert report["tasks"][0]["assessment"] == "mixed_outcomes_candidate_for_training"
    assert report["tasks"][0]["wilson_95"][0] < 0.5 < report["tasks"][0]["wilson_95"][1]
