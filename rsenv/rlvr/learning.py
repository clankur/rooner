"""Convert verified episodes to masked, equally weighted GRPO examples."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rsenv.rlvr.contracts import EpisodeResult, InfrastructureError


@dataclass(frozen=True)
class TrainingExample:
    """Aligned next-token arrays of length prompt_tokens + output_tokens - 1.

    Log-probabilities are natural logs. Prompt targets have zero advantage;
    generated targets share the episode advantage divided by its token count
    and the group size. Tinker sums these weighted token losses.
    """

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
    mean_reward = sum(rewards) / group_size
    reward_variance = sum((reward - mean_reward) ** 2 for reward in rewards) / group_size
    if not math.isfinite(reward_variance):
        raise InfrastructureError("nonfinite_rewards")
    reward_std = math.sqrt(reward_variance)
    # Validate records even for zero-signal groups, so corruption cannot hide in skipped batches.
    examples: list[TrainingExample] = []
    renderers = {t.turn.renderer for e in episodes for t in e.transitions}
    if len(renderers) != 1:
        raise InfrastructureError("mixed_renderers")
    for episode, reward in zip(episodes, rewards, strict=True):
        generated_tokens = sum(len(transition.turn.output_ids) for transition in episode.transitions)
        if not generated_tokens:
            raise InfrastructureError("no_sampled_tokens")
        advantage = (reward - mean_reward) / (reward_std + 1e-8)
        per_token_advantage = advantage / (generated_tokens * group_size)
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
            sequence_ids = turn.prompt_ids + turn.output_ids
            prompt_targets = len(turn.prompt_ids) - 1
            masked_prefix = (0.0,) * prompt_targets
            examples.append(
                TrainingExample(
                    input_ids=sequence_ids[:-1],
                    targets=sequence_ids[1:],
                    behavior_logprobs=masked_prefix + turn.logprobs,
                    advantages=masked_prefix + (per_token_advantage,) * len(turn.output_ids),
                )
            )
    return examples if reward_variance > 0 else []
