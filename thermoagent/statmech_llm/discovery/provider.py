"""Pinned Qwen provider and deterministic structured test providers for V12."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from .core import AgentDecision, ProviderResult, decision_schema


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def schema_checksum() -> str:
    return sha256_text(json.dumps(decision_schema(), sort_keys=True, separators=(",", ":")))


class InvalidStructuredDecision(RuntimeError):
    """A model response remained invalid after the one allowed repair."""

    def __init__(self, message: str, result: Optional[ProviderResult] = None) -> None:
        super().__init__(message)
        self.result = result


class QwenStatmechProvider:
    def __init__(
        self,
        artifact_root: Path,
        repository_root: Path,
        inference_temperature: float = 0.72,
        top_p: float = 0.90,
        maximum_new_tokens: int = 128,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        repository = Path(repository_root).resolve()
        if self.artifact_root == repository or repository in self.artifact_root.parents:
            raise ValueError("raw model artifacts must remain outside the repository")
        if inference_temperature <= 0.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("invalid inference settings")
        self.inference_temperature = float(inference_temperature)
        self.top_p = float(top_p)
        self.maximum_new_tokens = int(maximum_new_tokens)
        self._model = None
        self._tokenizer = None
        self._call_index = 0
        self.accounting: Dict[str, float] = {
            "decision_requests": 0,
            "model_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "latency_seconds": 0.0,
            "first_pass_valid": 0,
            "repaired_valid": 0,
            "invalid_after_repair": 0,
            "model_loading_seconds": 0.0,
        }
        if self.artifact_root.exists():
            for path in self.artifact_root.glob("call_*.json"):
                try:
                    self._call_index = max(self._call_index, int(path.name.split("_", 2)[1]))
                except (IndexError, ValueError):
                    continue

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
        )
        started = time.perf_counter()
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        self._model.eval()
        self.accounting["model_loading_seconds"] += float(time.perf_counter() - started)

    @staticmethod
    def _parse(text: str) -> Mapping[str, object]:
        payload = json.loads(text.strip())
        if not isinstance(payload, dict):
            raise ValueError("response is not one JSON object")
        AgentDecision.from_mapping(payload)
        return payload

    def _generate(self, prompt: str, seed: int, temperature: float) -> Tuple[str, int, int, float]:
        self.load()
        import torch
        from transformers import set_seed

        set_seed(int(seed), deterministic=True)
        messages = [
            {"role": "system", "content": "Return exactly one valid JSON object and no other text."},
            {"role": "user", "content": prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self._tokenizer(rendered, return_tensors="pt")
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        generation = {
            "do_sample": bool(float(temperature) > 0.0),
            "max_new_tokens": self.maximum_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if generation["do_sample"]:
            generation.update({"temperature": float(temperature), "top_p": self.top_p})
        with torch.inference_mode():
            output = self._model.generate(**encoded, **generation)
        latency = time.perf_counter() - started
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        generated_tokens = int(output.shape[-1] - prompt_tokens)
        response = self._tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
        return response, prompt_tokens, generated_tokens, latency

    def _record(
        self,
        prompt: str,
        responses: Tuple[str, ...],
        seed: int,
        temperature: float,
        valid: bool,
        prompt_tokens: int,
        generated_tokens: int,
        latency_seconds: float,
        first_pass_valid: bool,
        repaired: bool,
    ) -> str:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._call_index += 1
        record = {
            "call_index": self._call_index,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "schema_sha256": schema_checksum(),
            "seed": int(seed),
            "inference_sampling_temperature": float(temperature),
            "top_p": self.top_p,
            "prompt": prompt,
            "responses": list(responses),
            "valid": bool(valid),
            "model_calls": len(responses),
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
            "latency_seconds": float(latency_seconds),
            "first_pass_valid": bool(first_pass_valid),
            "repaired": bool(repaired),
            "repair_sampling_temperature": 0.0 if len(responses) > 1 else None,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        serialized = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        digest = hashlib.sha256(serialized).hexdigest()
        destination = self.artifact_root / ("call_%08d_%s.json" % (self._call_index, digest[:12]))
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
        return digest

    def decide(self, prompt: str, seed: int, sampling_temperature: Optional[float] = None) -> ProviderResult:
        temperature = self.inference_temperature if sampling_temperature is None else float(sampling_temperature)
        self.accounting["decision_requests"] += 1
        first, prompt_tokens, generated_tokens, latency = self._generate(prompt, int(seed), temperature)
        self.accounting["model_calls"] += 1
        responses = [first]
        first_valid = True
        repaired = False
        try:
            payload = self._parse(first)
            self.accounting["first_pass_valid"] += 1
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            first_valid = False
            repair_prompt = (
                "The prior response failed validation (%s). Return exactly one corrected object matching this schema: %s\n"
                "PRIOR_RESPONSE=%s"
                % (str(error)[:160], json.dumps(decision_schema(), sort_keys=True), first[:1000])
            )
            # The one bounded repair is deliberately greedy.  This is an
            # engineering validity repair, not a second scientific draw.
            second, p2, g2, l2 = self._generate(
                repair_prompt, (int(seed) + 1000003) % (2 ** 32 - 1), 0.0
            )
            self.accounting["model_calls"] += 1
            responses.append(second)
            prompt_tokens += p2
            generated_tokens += g2
            latency += l2
            try:
                payload = self._parse(second)
                repaired = True
                self.accounting["repaired_valid"] += 1
            except (ValueError, TypeError, json.JSONDecodeError):
                self.accounting["invalid_after_repair"] += 1
                self.accounting["prompt_tokens"] += int(prompt_tokens)
                self.accounting["generated_tokens"] += int(generated_tokens)
                self.accounting["latency_seconds"] += float(latency)
                digest = self._record(
                    prompt, tuple(responses), seed, temperature, False,
                    prompt_tokens, generated_tokens, latency, first_valid, False,
                )
                failed = ProviderResult(
                    payload={},
                    first_pass_valid=first_valid,
                    repaired=False,
                    prompt_tokens=int(prompt_tokens),
                    generated_tokens=int(generated_tokens),
                    latency_seconds=float(latency),
                    raw_artifact_sha256=digest,
                )
                raise InvalidStructuredDecision("response remained invalid after one bounded repair", failed)
        digest = self._record(
            prompt, tuple(responses), seed, temperature, True,
            prompt_tokens, generated_tokens, latency, first_valid, repaired,
        )
        self.accounting["prompt_tokens"] += int(prompt_tokens)
        self.accounting["generated_tokens"] += int(generated_tokens)
        self.accounting["latency_seconds"] += float(latency)
        return ProviderResult(
            payload=payload,
            first_pass_valid=first_valid,
            repaired=repaired,
            prompt_tokens=int(prompt_tokens),
            generated_tokens=int(generated_tokens),
            latency_seconds=float(latency),
            raw_artifact_sha256=digest,
        )

    def environment_manifest(self) -> Dict[str, object]:
        output: Dict[str, object] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "quantization": "NF4 double quantization; BF16 computation",
            "backend": "Transformers AutoModelForCausalLM",
            "inference_sampling_temperature": self.inference_temperature,
            "top_p": self.top_p,
            "maximum_new_tokens": self.maximum_new_tokens,
            "schema_sha256": schema_checksum(),
            "accounting": dict(self.accounting),
        }
        try:
            import bitsandbytes
            import torch
            import transformers

            output.update(
                {
                    "bitsandbytes_version": bitsandbytes.__version__,
                    "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
            )
        except ImportError:
            output["runtime_dependencies_available"] = False
        return output


class FunctionalProvider:
    """Test provider whose callable, not the scheduler, selects each decision."""

    def __init__(self, function: Callable[[str, int], Mapping[str, object]]) -> None:
        self.function = function
        self.accounting: Dict[str, float] = {"decision_requests": 0, "model_calls": 0}

    def decide(self, prompt: str, seed: int, sampling_temperature: Optional[float] = None) -> ProviderResult:
        del sampling_temperature
        self.accounting["decision_requests"] += 1
        self.accounting["model_calls"] += 1
        payload = self.function(prompt, int(seed))
        AgentDecision.from_mapping(payload)
        digest = sha256_text(prompt + "\0" + json.dumps(payload, sort_keys=True))
        return ProviderResult(payload, True, False, len(prompt.split()), 8, 0.0, digest)


class KineticIsingProvider:
    """Transparent stochastic local-response control using only prompt-visible state."""

    def __init__(
        self,
        private_weight: float = 0.8,
        neighbor_weight: float = 0.7,
        persistence_weight: float = 0.45,
        belief_action_weight: float = 0.7,
    ) -> None:
        self.private_weight = float(private_weight)
        self.neighbor_weight = float(neighbor_weight)
        self.persistence_weight = float(persistence_weight)
        self.belief_action_weight = float(belief_action_weight)
        self.accounting: Dict[str, float] = {"decision_requests": 0, "model_calls": 0}

    @staticmethod
    def _spin(label: str) -> int:
        if label not in ("amber", "cobalt"):
            raise ValueError("invalid displayed state")
        return 1 if label == "amber" else -1

    def decide(self, prompt: str, seed: int, sampling_temperature: Optional[float] = None) -> ProviderResult:
        self.accounting["decision_requests"] += 1
        self.accounting["model_calls"] += 1
        envelope = json.loads(prompt.split("LOCAL_UPDATE=", 1)[1])
        view = envelope["authorized_local_state"]
        current_belief = self._spin(str(view["current_belief"]))
        current_action = self._spin(str(view["current_action"]))
        private_text = str(view["private_observation"])
        private = 1 if "amber" in private_text else (-1 if "cobalt" in private_text else 0)
        messages = list(view["delivered_neighbor_packets"])
        signals = [
            self._spin(str(packet["signal"]))
            for packet in messages
            if str(packet["signal"]) in ("amber", "cobalt")
        ]
        neighbor = float(np.mean(signals)) if signals else 0.0
        relevance = float(view["neighbor_relevance"])
        field = (
            self.private_weight * private
            + self.neighbor_weight * relevance * neighbor
            + self.persistence_weight * current_belief
            + 0.15 * float(view["current_local_workload"])
        )
        temperature = 1.0 if sampling_temperature is None else max(float(sampling_temperature), 0.05)
        probability_amber = 1.0 / (1.0 + np.exp(-2.0 * field / temperature))
        rng = np.random.default_rng(int(seed))
        belief = 1 if rng.random() < probability_amber else -1
        action_field = self.belief_action_weight * belief + self.persistence_weight * current_action
        probability_action_amber = 1.0 / (1.0 + np.exp(-2.0 * action_field / temperature))
        action = 1 if rng.random() < probability_action_amber else -1
        belief_label = "amber" if belief == 1 else "cobalt"
        action_label = "amber" if action == 1 else "cobalt"
        payload: Mapping[str, object] = {
            "belief_choice": belief_label,
            "action_choice": action_label,
            "confidence": float(max(probability_amber, 1.0 - probability_amber)),
            "commitment_status": "provisional" if action != current_action else "committed",
            "memory_state": "stable" if belief == current_belief else ("evidence_amber" if belief == 1 else "evidence_cobalt"),
            "outgoing_signal": belief_label,
            "tool_action": "execute_selected",
            "reason_code": "neighbor_messages" if signals else "private_observation",
        }
        AgentDecision.from_mapping(payload)
        digest = sha256_text(prompt + "\0" + json.dumps(payload, sort_keys=True))
        return ProviderResult(payload, True, False, len(prompt.split()), 8, 0.0, digest)
