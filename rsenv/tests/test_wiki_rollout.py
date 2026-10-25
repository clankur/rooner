import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import BaseModel

from rsenv import GameState, SimClient, Trajectory, rollout_trajectory
from rsenv.state import ActionResult
from rsenv.tools import WikiSearch
from rsenv.wiki import WikiIndex


class ByteTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.unk_token_id

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        return bytes(ids.tolist()).decode("utf-8")


class TestProcessor:
    __test__ = False

    def __init__(self) -> None:
        self.tokenizer = ByteTokenizer()
        self.schemas: list[list[dict]] = []

    def apply_chat_template(
        self, messages: list[dict], tools: list[dict], add_generation_prompt: bool, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        self.schemas.append(tools)
        # Deliberately serialize differently from sampled output to exercise the
        # tool-response boundary and loss mask, not just identical round trips.
        text = "".join(json.dumps(message) + "\n" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return {"input_ids": torch.tensor([list(text.encode("utf-8"))])}


class ScriptedModel:
    def __init__(self, turns: list[str]) -> None:
        self.turns = turns
        self.inputs: list[torch.Tensor] = []
        self.generated: list[str] = []

    def eval(self) -> None:
        pass

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        turn = self.turns[min(len(self.inputs), len(self.turns) - 1)]
        self.inputs.append(input_ids.clone())
        self.generated.append(turn)
        new_ids = torch.tensor([list(turn.encode("utf-8"))])
        return SimpleNamespace(
            sequences=torch.cat((input_ids, new_ids), dim=1),
            scores=[torch.zeros((1, 256)) for _ in range(new_ids.shape[1])],
        )


class RecordingClient(SimClient):
    def __init__(self) -> None:
        super().__init__()
        self.actions: list[str] = []

    def execute_action(self, name: str, args: BaseModel) -> ActionResult:
        self.actions.append(name)
        return super().execute_action(name, args)


def tool_call(name: str, **arguments: str) -> str:
    return "<think>Plan</think>\n<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def run_script(
    model: ScriptedModel,
    processor: TestProcessor,
    client: RecordingClient,
    wiki_root: Path,
    max_actions: int = 2,
    max_lookups: int = 1,
) -> Trajectory:
    return rollout_trajectory(
        model=model,
        processor=processor,
        initial_state=GameState(inventory={"Bronze axe": 1}, nearby=["Tree"]),
        max_actions=max_actions,
        max_lookups=max_lookups,
        wiki_data_path=str(wiki_root),
        max_new_tokens=512,
        temperature=0.7,
        device=torch.device("cpu"),
        client=client,
    )


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    (tmp_path / "learnings").mkdir()
    (tmp_path / "learnings" / "fletching.md").write_text("# Fletching\nKnife spawn at (3224, 3202).")
    return tmp_path


def test_lookup_preserves_actions_and_response_is_context_not_loss(
    wiki_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsenv import env

    reward = Mock(wraps=env.compute_reward)
    monkeypatch.setattr(env, "compute_reward", reward)
    model = ScriptedModel([tool_call("wikiSearch", query="knife"), tool_call("chopTree")])
    processor = TestProcessor()
    client = RecordingClient()
    traj = run_script(model, processor, client, wiki_root)

    assert traj.num_lookups == 1
    assert traj.num_actions == traj.num_valid_actions == 2
    assert client.actions == ["chopTree", "chopTree"]
    assert traj.total_xp == 50
    context = processor.tokenizer.decode(model.inputs[1][0])
    assert "Knife spawn at (3224, 3202)" in context
    assert "Wiki lookups remaining: 0" in context
    assert len(traj.full_ids) == len(traj.generation_mask)
    generated_ids = traj.full_ids[traj.generation_mask.bool()]
    assert processor.tokenizer.decode(generated_ids) == "".join(model.generated)
    assert len(traj.old_log_probs) == len(generated_ids)
    assert traj.total_gen_tokens == len(generated_ids)
    assert reward.call_args.kwargs["num_actions"] == reward.call_args.kwargs["num_valid"] == 2
    assert [name for name, xp in reward.call_args.kwargs["action_xp_history"]] == client.actions


def test_batched_lookup_and_action_keep_separate_budgets(wiki_root: Path) -> None:
    model = ScriptedModel(
        [
            tool_call("wikiSearch", query="knife") + tool_call("chopTree"),
            tool_call("chopTree"),
        ]
    )
    processor = TestProcessor()
    client = RecordingClient()
    traj = run_script(model, processor, client, wiki_root)
    assert (traj.num_lookups, traj.num_actions, traj.num_valid_actions) == (1, 2, 2)
    assert "Knife spawn" in processor.tokenizer.decode(model.inputs[1][0])
    assert processor.tokenizer.decode(traj.full_ids[traj.generation_mask.bool()]) == "".join(model.generated)


def test_exhausted_budget_rejects_search_but_allows_gameplay(wiki_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = Mock(wraps=WikiIndex(wiki_root))
    monkeypatch.setattr("rsenv.env.get_wiki_index", lambda path: index)
    model = ScriptedModel(
        [
            tool_call("wikiSearch", query="knife"),
            tool_call("wikiSearch", query="knife"),
            tool_call("chopTree"),
        ]
    )
    processor = TestProcessor()
    client = RecordingClient()
    traj = run_script(model, processor, client, wiki_root)
    assert index.search.call_count == traj.num_lookups == 1
    assert traj.num_actions == traj.num_valid_actions == 2
    assert "budget exhausted" in processor.tokenizer.decode(model.inputs[2][0])


def test_batched_calls_cannot_exceed_either_budget(wiki_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = Mock(wraps=WikiIndex(wiki_root))
    monkeypatch.setattr("rsenv.env.get_wiki_index", lambda path: index)
    model = ScriptedModel(
        [
            tool_call("wikiSearch", query="knife") * 3 + tool_call("chopTree") * 3,
        ]
    )
    client = RecordingClient()
    traj = run_script(model, TestProcessor(), client, wiki_root)
    assert index.search.call_count == traj.num_lookups == 1
    assert traj.num_actions == traj.num_valid_actions == len(client.actions) == 2


def test_repeated_lookups_terminate_without_awarding_xp(wiki_root: Path) -> None:
    model = ScriptedModel([tool_call("wikiSearch", query="knife")])
    client = RecordingClient()
    traj = run_script(model, TestProcessor(), client, wiki_root)
    assert len(model.inputs) == 2 + 1 + 1
    assert traj.num_lookups == 1
    assert traj.num_actions == traj.num_valid_actions == traj.total_xp == 0
    assert client.actions == []


def test_zero_budget_hides_tool_and_rejects_hallucinated_lookup(
    wiki_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = Mock()
    monkeypatch.setattr("rsenv.env.get_wiki_index", index)
    model = ScriptedModel([tool_call("wikiSearch", query="knife"), tool_call("chopTree")])
    processor = TestProcessor()
    traj = run_script(model, processor, RecordingClient(), wiki_root, max_lookups=0)
    assert traj.num_lookups == 0
    assert traj.num_actions == 2
    assert all(schema["function"]["name"] != "wikiSearch" for schemas in processor.schemas for schema in schemas)
    index.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_no_results_or_missing_corpus_spends_lookup_only(wiki_root: Path, missing: bool) -> None:
    root = wiki_root / "missing" if missing else wiki_root
    model = ScriptedModel([tool_call("wikiSearch", query="unfindable"), tool_call("chopTree")])
    processor = TestProcessor()
    traj = run_script(model, processor, RecordingClient(), root)
    assert traj.num_lookups == 1
    assert traj.num_actions == traj.num_valid_actions == 2
    context = processor.tokenizer.decode(model.inputs[1][0])
    assert "Wiki unavailable" in context if missing else 'results\\": []' in context


def test_zero_action_budget_does_not_generate(wiki_root: Path) -> None:
    model = ScriptedModel([tool_call("wikiSearch", query="knife")])
    traj = run_script(model, TestProcessor(), RecordingClient(), wiki_root, max_actions=0)
    assert model.inputs == []
    assert traj.num_actions == traj.num_lookups == 0


def test_negative_budget_rejected(wiki_root: Path) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        run_script(
            ScriptedModel([tool_call("chopTree")]), TestProcessor(), RecordingClient(), wiki_root, max_lookups=-1
        )


def test_simulator_does_not_treat_lookup_as_valid_game_action() -> None:
    assert not SimClient().execute_action("wikiSearch", WikiSearch(query="knife")).valid


def test_evaluation_passes_lookup_settings_and_reports_usage(wiki_root: Path) -> None:
    from eval import evaluate

    model = ScriptedModel([tool_call("wikiSearch", query="knife"), tool_call("chopTree")])
    processor = TestProcessor()
    results = evaluate(
        model=model,
        processor=processor,
        num_episodes=1,
        max_actions=1,
        max_lookups=2,
        wiki_data_path=str(wiki_root),
        max_new_tokens=512,
        temperature=0.7,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert results[0]["num_lookups"] == results[0]["num_actions"] == 1
    context = processor.tokenizer.decode(model.inputs[1][0])
    assert "Knife spawn at (3224, 3202)" in context
    assert "Wiki lookups remaining: 1" in context
