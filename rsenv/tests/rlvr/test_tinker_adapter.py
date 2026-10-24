from __future__ import annotations

import math
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from rsenv.rlvr.contracts import InfrastructureError, TaskSpec
from rsenv.rlvr.learning import TrainingExample
from rsenv.rlvr.tinker_backend import TinkerBackend, TinkerPolicy, bounded_call

if TYPE_CHECKING:
    from tinker.types import AdamParams, Datum, ModelInput, SamplingParams


class Future:
    def __init__(self, value: SimpleNamespace) -> None:
        self.value = value

    def result(self, timeout: float) -> SimpleNamespace:
        assert timeout > 0
        return self.value


def test_sampling_preserves_tool_tokens_and_behavior_probabilities() -> None:
    class Tokenizer:
        name_or_path = "test"
        eos_token_id = 99
        chat_template = "test-template"

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
            assert kwargs["enable_thinking"] is False and kwargs["tools"]
            assert kwargs["return_dict"] is False
            return [1, 2, 3]

        def decode(self, tokens: list[int], skip_special_tokens: bool) -> str:
            assert tokens == [4, 5] and not skip_special_tokens
            return '<tool_call>{"name":"chopTree","arguments":{}}</tool_call>'

    class Sampler:
        def sample(self, model_input: ModelInput, num_samples: int, sampling_params: SamplingParams) -> Future:
            assert num_samples == 1 and sampling_params.temperature == 1
            return Future(
                SimpleNamespace(
                    sequences=[SimpleNamespace(tokens=[4, 5, 99], logprobs=[-1, -2, -3], stop_reason="stop")]
                )
            )

    turn = TinkerPolicy(Sampler(), Tokenizer(), "version").sample([], TaskSpec(), 42, 10)
    assert turn.prompt_ids == (1, 2, 3) and turn.output_ids == (4, 5, 99)
    assert turn.logprobs == (-1, -2, -3) and "<tool_call>" in turn.text


def test_optimizer_payload_and_nonfinite_guard() -> None:
    class Trainer:
        def __init__(self) -> None:
            self.loss = 1.0
            self.steps = 0

        def forward_backward(self, data: list[Datum], loss_fn: str, loss_fn_config: dict[str, float]) -> Future:
            assert loss_fn == "ppo"
            assert loss_fn_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
            assert list(data[0].loss_fn_inputs["advantages"].data) == [0.0, 0.5]
            assert list(data[0].loss_fn_inputs["logprobs"].data) == [0.0, -1.0]
            return Future(SimpleNamespace(metrics={"loss:sum": self.loss}, loss_fn_outputs=[]))

        def optim_step(self, params: AdamParams) -> Future:
            assert params.learning_rate == 1e-5
            self.steps += 1
            return Future(SimpleNamespace(metrics=None))

    backend = object.__new__(TinkerBackend)
    backend.trainer = Trainer()
    batch = [TrainingExample((1, 2), (2, 3), (0.0, -1.0), (0.0, 0.5))]
    backend.update(batch)
    backend.trainer.loss = math.nan
    with pytest.raises(InfrastructureError, match="nonfinite"):
        backend.update(batch)
    assert backend.trainer.steps == 1


def test_setup_deadline_bounds_paused_sdk() -> None:
    release = threading.Event()
    try:
        with pytest.raises(InfrastructureError, match="tinker_setup_deadline"):
            bounded_call(release.wait, timeout=0.005)
    finally:
        release.set()


@pytest.mark.parametrize("error", [ValueError("setup failed"), TimeoutError("service timed out")])
def test_setup_preserves_service_errors(error: Exception) -> None:
    def fail() -> None:
        raise error

    with pytest.raises(type(error)) as caught:
        bounded_call(fail)
    assert caught.value is error
