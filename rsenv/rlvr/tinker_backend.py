"""Tinker sampling and optimization; optional dependency, imported only on demand."""

from __future__ import annotations

import math
import queue
import threading
from typing import Any

from rsenv.rlvr.contracts import InfrastructureError, SampledTurn, TaskSpec, fingerprint
from rsenv.rlvr.learning import TrainingExample
from rsenv.tools import TOOL_SCHEMAS


def bounded_call(call: Any, timeout: float = 120) -> Any:
    """Bound SDK setup calls, including the SDK's automatic billing pause."""
    result: queue.Queue = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result.put((True, call()))
        except BaseException as exc:
            result.put((False, exc))

    threading.Thread(target=run, daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty as exc:
        raise InfrastructureError("tinker_setup_deadline:check_service_and_billing") from exc
    if not ok:
        raise value
    return value


class TinkerPolicy:
    def __init__(
        self,
        sampler: Any,
        tokenizer: Any,
        version: str,
        max_tokens: int = 512,
        max_context: int = 16384,
        temperature: float = 1.0,
    ) -> None:
        self.sampler, self.tokenizer, self.version = sampler, tokenizer, version
        self.max_tokens, self.max_context, self.temperature = max_tokens, max_context, temperature
        self.renderer = (
            f"{tokenizer.name_or_path}:chat-template-v1:{fingerprint(getattr(tokenizer, 'chat_template', None))}"
        )

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
                stop_reason="max_tokens",
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
        self.trainer = None
        try:
            supported = {m.model_name for m in bounded_call(self.service.get_server_capabilities, 30).supported_models}
            if model not in supported:
                raise InfrastructureError(f"unsupported_tinker_model:{model}")
            if training:
                self.trainer = bounded_call(
                    lambda: (
                        self.service.create_training_client_from_state_with_optimizer(resume_path)
                        if resume_path
                        else self.service.create_lora_training_client(base_model=model, rank=rank, seed=seed)
                    )
                )
                self.tokenizer = self.trainer.get_tokenizer()
            else:
                sampler = bounded_call(
                    lambda: (
                        self.service.create_sampling_client(model_path=resume_path)
                        if resume_path
                        else self.service.create_sampling_client(base_model=model)
                    )
                )
                self.tokenizer = sampler.get_tokenizer()
                self.policy = TinkerPolicy(sampler, self.tokenizer, resume_path or model)
        except BaseException:
            try:
                self.close()
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
        for output in getattr(result, "loss_fn_outputs", []):
            for tensor in output.values():
                if any(not math.isfinite(value) for value in tensor.data):
                    raise InfrastructureError("nonfinite_training_output")
        # Do not retry this operation at application level. The committed checkpoint is the recovery boundary.
        step = self.trainer.optim_step(tinker.types.AdamParams(learning_rate=learning_rate)).result(timeout=120)
        step_metrics = {f"optimizer/{k}": float(v) for k, v in (getattr(step, "metrics", None) or {}).items()}
        if any(not math.isfinite(v) for v in step_metrics.values()):
            raise InfrastructureError("nonfinite_optimizer_metrics")
        metrics.update(step_metrics)
        return metrics

    def close(self) -> None:
        self.service.close().result(timeout=5)
