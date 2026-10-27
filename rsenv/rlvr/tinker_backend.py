"""Tinker sampling and optimization."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from concurrent.futures import Future, wait
from functools import partial
from typing import TYPE_CHECKING, Literal, TypeVar

from rsenv.rlvr.contracts import InfrastructureError, SampledTurn, TaskSpec, fingerprint
from rsenv.rlvr.learning import TrainingExample
from rsenv.tools import TOOL_SCHEMAS

if TYPE_CHECKING:
    from tinker import SamplingClient, TrainingClient
    from transformers import PreTrainedTokenizerBase

T = TypeVar("T")


def bounded_call(call: Callable[[], T], timeout: float = 120) -> T:
    """Bound setup calls even when the SDK pauses indefinitely for billing."""
    result: Future[T] = Future()

    def run() -> None:
        try:
            result.set_result(call())
        except BaseException as exc:
            result.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    if not wait([result], timeout=timeout).done:
        raise InfrastructureError("tinker_setup_deadline:check_service_and_billing")
    return result.result()


class TinkerPolicy:
    def __init__(
        self,
        sampler: SamplingClient,
        tokenizer: PreTrainedTokenizerBase,
        version: str,
        max_tokens: int = 512,
        max_context: int = 65536,
        temperature: float = 1.0,
    ) -> None:
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.version = version
        self.max_tokens = max_tokens
        self.max_context = max_context
        self.temperature = temperature
        self.renderer = f"{tokenizer.name_or_path}:chat-template-v1:{fingerprint(tokenizer.chat_template)}"

    def sample(self, messages: list[dict[str, str]], task: TaskSpec, seed: int, timeout: float) -> SampledTurn:
        import tinker

        tools = [s for s in TOOL_SCHEMAS if s["function"]["name"] in task.allowed_tools]
        ids = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
        )
        if len(ids) + self.max_tokens > self.max_context:
            # This is an episode budget outcome, not an infrastructure outage.
            return SampledTurn(
                prompt_ids=tuple(ids),
                output_ids=(),
                logprobs=(),
                text="",
                stop_reason="context_limit",
                policy_version=self.version,
                renderer=self.renderer,
                seed=seed,
            )
        params = tinker.types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            top_k=-1,
            seed=seed,
            stop=[self.tokenizer.eos_token_id],
        )
        future = self.sampler.sample(tinker.types.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
        try:
            response = future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise InfrastructureError("sampling_deadline") from exc
        sequence = response.sequences[0]
        if sequence.logprobs is None or len(sequence.tokens) != len(sequence.logprobs):
            raise InfrastructureError("missing_sampling_probabilities")
        if any(not math.isfinite(p) for p in sequence.logprobs):
            raise InfrastructureError("nonfinite_sampling_probabilities")
        display_tokens = list(sequence.tokens)
        if display_tokens and display_tokens[-1] == self.tokenizer.eos_token_id:
            display_tokens.pop()
        return SampledTurn(
            prompt_ids=tuple(ids),
            output_ids=tuple(sequence.tokens),
            logprobs=tuple(sequence.logprobs),
            text=self.tokenizer.decode(display_tokens, skip_special_tokens=False),
            stop_reason=str(sequence.stop_reason),
            policy_version=self.version,
            renderer=self.renderer,
            seed=seed,
            sampling={
                "temperature": self.temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": self.max_tokens,
                "thinking": False,
            },
        )


class TinkerBackend:
    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-4B",
        rank: int = 16,
        seed: int = 42,
        resume_path: str | None = None,
        training: bool = True,
    ) -> None:
        import tinker

        self.service = tinker.ServiceClient(display_name="RuneScape RLVR")
        self.model = model
        self.rank = rank
        self.trainer: TrainingClient | None = None
        try:
            supported = {m.model_name for m in bounded_call(self.service.get_server_capabilities, 30).supported_models}
            if model not in supported:
                raise InfrastructureError(f"unsupported_tinker_model:{model}")
            if training:
                if resume_path:
                    self.trainer = bounded_call(
                        partial(self.service.create_training_client_from_state_with_optimizer, resume_path)
                    )
                else:
                    self.trainer = bounded_call(
                        partial(self.service.create_lora_training_client, base_model=model, rank=rank, seed=seed)
                    )
                self.tokenizer = self.trainer.get_tokenizer()
            else:
                if resume_path:
                    sampler = bounded_call(partial(self.service.create_sampling_client, model_path=resume_path))
                else:
                    sampler = bounded_call(partial(self.service.create_sampling_client, base_model=model))
                self.tokenizer = sampler.get_tokenizer()
                self.policy = TinkerPolicy(sampler, self.tokenizer, resume_path or model)
        except BaseException:
            try:
                self.close("errored")
            except Exception:
                pass  # Preserve the setup failure if service teardown also fails.
            raise

    def sampling_policy(self, name: str) -> TinkerPolicy:
        if self.trainer is None:
            return self.policy
        path = self.trainer.save_weights_for_sampler(name).result(timeout=120).path
        sampler = bounded_call(lambda: self.service.create_sampling_client(model_path=path))
        return TinkerPolicy(sampler, self.tokenizer, path)

    def save_state(self, name: str) -> str:
        assert self.trainer is not None
        return self.trainer.save_state(name).result(timeout=120).path

    def update(self, examples: list[TrainingExample], learning_rate: float = 1e-5) -> dict[str, float]:
        import tinker

        assert self.trainer is not None
        if not examples:
            raise ValueError("Cannot update a zero-signal group")
        data = [
            tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(list(e.input_ids)),
                loss_fn_inputs={
                    "target_tokens": tinker.types.TensorData(list(e.targets), dtype="int64", shape=[len(e.targets)]),
                    "logprobs": tinker.types.TensorData(
                        list(e.behavior_logprobs), dtype="float32", shape=[len(e.targets)]
                    ),
                    "advantages": tinker.types.TensorData(list(e.advantages), dtype="float32", shape=[len(e.targets)]),
                },
            )
            for e in examples
        ]
        result = self.trainer.forward_backward(
            data,
            loss_fn="ppo",
            loss_fn_config={
                "clip_low_threshold": 0.8,
                "clip_high_threshold": 1.2,
            },
        ).result(timeout=120)
        metrics = {k: float(v) for k, v in result.metrics.items()}
        if not metrics or any(not math.isfinite(v) for v in metrics.values()):
            raise InfrastructureError("nonfinite_training_metrics")
        for output in result.loss_fn_outputs:
            for tensor in output.values():
                if any(not math.isfinite(value) for value in tensor.data):
                    raise InfrastructureError("nonfinite_training_output")
        # Do not retry this operation at application level. The committed checkpoint is the recovery boundary.
        step = self.trainer.optim_step(tinker.types.AdamParams(learning_rate=learning_rate)).result(timeout=120)
        step_metrics = {f"optimizer/{k}": float(v) for k, v in (step.metrics or {}).items()}
        if any(not math.isfinite(v) for v in step_metrics.values()):
            raise InfrastructureError("nonfinite_optimizer_metrics")
        metrics.update(step_metrics)
        return metrics

    def close(self, status: Literal["success", "errored", "interrupted"] = "success") -> None:
        self.service.close(status).result(timeout=5)
