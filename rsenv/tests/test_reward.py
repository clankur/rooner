import pytest

from rsenv.env import compute_reward


@pytest.mark.parametrize(
    "actions,valid,xp,expected",
    [(0, 0, 0, -2.0), (10, 0, 0, -0.5), (10, 10, 0, 0.0), (1, 1, 25, 0.35)],
)
def test_legacy_reward_boundaries(actions: int, valid: int, xp: int, expected: float) -> None:
    result = compute_reward(
        total_xp_gained=xp,
        num_actions=actions,
        num_valid=valid,
        initial_skills={},
        final_skills={"Woodcutting": xp},
        action_xp_history=[("chopTree", {"Woodcutting": xp})] if xp else [],
    )
    assert result.reward == pytest.approx(expected)
    assert result.reward == sum(result.metrics.values())
    assert result.num_level_ups == 0
