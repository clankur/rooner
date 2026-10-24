"""Backend-neutral GRPO conversion with explicit trajectory weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rsenv.rlvr.contracts import EpisodeResult, InfrastructureError


@dataclass(frozen=True)
class TrainingExample:
    input_ids: tuple[int, ...]
    targets: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    advantages: tuple[float, ...]


def build_examples(episodes: list[EpisodeResult], group_size: int) -> list[TrainingExample]:
    if group_size < 2 or len(episodes) != group_size:
        raise InfrastructureError("invalid_group_size")
    first = episodes[0]
    ids: set[str] = set()
    for episode in episodes:
        if (
            episode.status == "infra_invalid"
            or episode.reward is None
            or episode.receipt is None
            or episode.task_hash != first.task_hash
            or episode.policy_version != first.policy_version
            or episode.receipt.state_hash != first.receipt.state_hash
            or episode.episode_id in ids
        ):
            raise InfrastructureError("invalid_group_membership")
        ids.add(episode.episode_id)
        if (
            episode.reward not in (0.0, 1.0)
            or episode.verification is None
            or episode.reward != float(episode.verification.success)
            or (episode.status == "success") != episode.verification.success
        ):
            raise InfrastructureError("unverified_group_reward")
    rewards = [float(e.reward) for e in episodes]
    mean = sum(rewards) / group_size
    variance = sum((r - mean) ** 2 for r in rewards) / group_size
    if not math.isfinite(variance):
        raise InfrastructureError("nonfinite_rewards")
    # Validate records even for zero-signal groups, so corruption cannot hide in skipped batches.
    examples: list[TrainingExample] = []
    renderers = {t.turn.renderer for e in episodes for t in e.transitions}
    if len(renderers) != 1:
        raise InfrastructureError("mixed_renderers")
    for episode, reward in zip(episodes, rewards, strict=True):
        tokens = sum(len(t.turn.output_ids) for t in episode.transitions)
        if not tokens:
            raise InfrastructureError("no_sampled_tokens")
        advantage = (reward - mean) / (math.sqrt(variance) + 1e-8)
        for transition in episode.transitions:
            turn = transition.turn
            if (
                not turn.prompt_ids
                or len(turn.output_ids) != len(turn.logprobs)
                or turn.policy_version != episode.policy_version
                or any(not math.isfinite(lp) or lp > 1e-5 for lp in turn.logprobs)
                or any(token < 0 for token in turn.prompt_ids + turn.output_ids)
                or turn.sampling.get("temperature", 1.0) != 1.0
            ):
                raise InfrastructureError("invalid_sampled_turn")
            if not turn.output_ids:
                continue
            full = turn.prompt_ids + turn.output_ids
            prefix = len(turn.prompt_ids) - 1
            examples.append(
                TrainingExample(
                    input_ids=full[:-1],
                    targets=full[1:],
                    behavior_logprobs=(0.0,) * prefix + turn.logprobs,
                    advantages=(0.0,) * prefix + (advantage / (tokens * group_size),) * len(turn.output_ids),
                )
            )
    return examples if variance > 0 else []
