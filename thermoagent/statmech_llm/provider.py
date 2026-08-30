"""Pinned cross-family structured providers for the V15 LLM experiment."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from thermoagent.statmech_llm.discovery.core import AgentDecision, ProviderResult, decision_schema
from thermoagent.statmech_llm.discovery.provider import InvalidStructuredDecision


@dataclass(frozen=True)
class ModelSpecification:
    key: str
    identifier: str
    revision: str


MODEL_SPECS = {
    "qwen": ModelSpecification(
        "qwen",
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "granite": ModelSpecification(
        "granite",
        "ibm-granite/granite-3.3-8b-instruct",
        "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    ),
}


def schema_checksum() -> str:
    serialized = json.dumps(decision_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class TransformersStatmechProvider:
    """One pinned model instance; the scheduler never supplies its decision."""

    def __init__(
        self,
        specification: ModelSpecification,
        artifact_root: Path,
        repository_root: Path,
        inference_temperature: float = 0.5,
        top_p: float = 0.9,
        maximum_new_tokens: int = 96,
    ) -> None:
        self.specification = specification
        self.artifact_root = Path(artifact_root).resolve()
        repository = Path(repository_root).resolve()
        if self.artifact_root == repository or repository in self.artifact_root.parents:
            raise ValueError("raw model artifacts must remain outside the repository")
        if float(inference_temperature) <= 0.0 or not 0.0 < float(top_p) <= 1.0:
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
            self.specification.identifier,
            revision=self.specification.revision,
            trust_remote_code=False,
        )
        started = time.perf_counter()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.specification.identifier,
            revision=self.specification.revision,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        self._model.eval()
        self.accounting["model_loading_seconds"] += float(time.perf_counter() - started)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _parse(text: str) -> Mapping[str, object]:
        value = json.loads(text.strip())
        if not isinstance(value, dict):
            raise ValueError("response is not one JSON object")
        AgentDecision.from_mapping(value)
        return value

    def _generate(self, prompt: str, seed: int, temperature: float) -> Tuple[str, int, int, float]:
        self.load()
        import torch
        from transformers import set_seed

        set_seed(int(seed), deterministic=True)
        messages = [
            {"role": "system", "content": "Return exactly one valid JSON object and no other text."},
            {"role": "user", "content": prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self._tokenizer(rendered, return_tensors="pt")
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation: Dict[str, object] = {
            "do_sample": bool(float(temperature) > 0.0),
            "max_new_tokens": self.maximum_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if generation["do_sample"]:
            generation.update({"temperature": float(temperature), "top_p": self.top_p})
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(**encoded, **generation)
        latency = float(time.perf_counter() - started)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        generated_tokens = int(output.shape[-1] - prompt_tokens)
        response = self._tokenizer.decode(
            output[0, prompt_tokens:], skip_special_tokens=True
        )
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
            "model_key": self.specification.key,
            "model_id": self.specification.identifier,
            "model_revision": self.specification.revision,
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
        destination = self.artifact_root / (
            "call_%08d_%s.json" % (self._call_index, digest[:12])
        )
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
        return digest

    def decide(
        self,
        prompt: str,
        seed: int,
        sampling_temperature: Optional[float] = None,
    ) -> ProviderResult:
        temperature = (
            self.inference_temperature
            if sampling_temperature is None
            else float(sampling_temperature)
        )
        self.accounting["decision_requests"] += 1
        first, prompt_tokens, generated_tokens, latency = self._generate(
            prompt, int(seed), temperature
        )
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
                "The prior response failed validation (%s). Return exactly one corrected object "
                "matching this schema: %s\nPRIOR_RESPONSE=%s"
                % (str(error)[:160], json.dumps(decision_schema(), sort_keys=True), first[:1000])
            )
            second, prompt_two, generated_two, latency_two = self._generate(
                repair_prompt,
                (int(seed) + 1000003) % (2 ** 32 - 1),
                0.0,
            )
            self.accounting["model_calls"] += 1
            responses.append(second)
            prompt_tokens += prompt_two
            generated_tokens += generated_two
            latency += latency_two
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
                    prompt,
                    tuple(responses),
                    seed,
                    temperature,
                    False,
                    prompt_tokens,
                    generated_tokens,
                    latency,
                    first_valid,
                    False,
                )
                result = ProviderResult(
                    {},
                    first_valid,
                    False,
                    int(prompt_tokens),
                    int(generated_tokens),
                    float(latency),
                    digest,
                )
                raise InvalidStructuredDecision(
                    "response remained invalid after one bounded repair", result
                )
        digest = self._record(
            prompt,
            tuple(responses),
            seed,
            temperature,
            True,
            prompt_tokens,
            generated_tokens,
            latency,
            first_valid,
            repaired,
        )
        self.accounting["prompt_tokens"] += int(prompt_tokens)
        self.accounting["generated_tokens"] += int(generated_tokens)
        self.accounting["latency_seconds"] += float(latency)
        return ProviderResult(
            payload,
            first_valid,
            repaired,
            int(prompt_tokens),
            int(generated_tokens),
            float(latency),
            digest,
        )

    def environment_manifest(self) -> Dict[str, object]:
        output: Dict[str, object] = {
            "model_key": self.specification.key,
            "model_id": self.specification.identifier,
            "model_revision": self.specification.revision,
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


__all__ = [
    "MODEL_SPECS",
    "ModelSpecification",
    "TransformersStatmechProvider",
    "schema_checksum",
]
