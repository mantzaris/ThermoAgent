"""Pinned Qwen structured provider with external-only raw-artifact storage.

Heavy dependencies are imported lazily so CPU theory and tests do not require a
Transformers installation.  The provider performs schema validation and permits
at most one repair generation.  It never infers states from arbitrary prose.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from .agents import ProviderResult, StructuredDecision, decision_schema_json


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_schema_manifest(prompt_templates: Tuple[str, ...]) -> Dict[str, object]:
    schema = decision_schema_json()
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "schema_sha256": _sha256_text(schema),
        "prompt_template_sha256": [_sha256_text(value) for value in prompt_templates],
    }


class QwenStructuredProvider:
    """One-device Qwen provider for controlled and autonomous agent turns."""

    def __init__(
        self,
        artifact_root: Path,
        repository_root: Path,
        inference_temperature: float = 0.65,
        top_p: float = 0.90,
        maximum_new_tokens: int = 220,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        repository = Path(repository_root).resolve()
        if self.artifact_root == repository or repository in self.artifact_root.parents:
            raise ValueError("raw Qwen artifacts must be outside the repository")
        if inference_temperature <= 0.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("invalid sampling settings")
        self.inference_temperature = float(inference_temperature)
        self.top_p = float(top_p)
        self.maximum_new_tokens = int(maximum_new_tokens)
        self._model = None
        self._tokenizer = None
        existing_indices = []
        if self.artifact_root.exists():
            for path in self.artifact_root.glob("call_*.json"):
                try:
                    existing_indices.append(int(path.name.split("_", 2)[1]))
                except (IndexError, ValueError):
                    continue
        # Preserve monotonically increasing external record identifiers when a
        # formally declared atomic batch is resumed after infrastructure loss.
        self._call_index = max(existing_indices, default=0)
        self.accounting = {
            "model_calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "latency_seconds": 0.0,
            "first_pass_valid": 0,
            "repaired_valid": 0,
            "invalid_after_repair": 0,
        }

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
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        self._model.eval()

    @staticmethod
    def _parse_exact_json(text: str) -> Mapping[str, object]:
        payload = json.loads(text.strip())
        if not isinstance(payload, dict):
            raise ValueError("model response is not a JSON object")
        StructuredDecision.from_mapping(payload)
        return payload

    def _generate(self, prompt: str, seed: int) -> Tuple[str, int, int, float]:
        self.load()
        import torch
        from transformers import set_seed

        set_seed(int(seed), deterministic=True)
        messages = [
            {"role": "system", "content": "Return exactly one valid JSON object and no other text."},
            {"role": "user", "content": prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(rendered, return_tensors="pt")
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                do_sample=True,
                temperature=self.inference_temperature,
                top_p=self.top_p,
                max_new_tokens=self.maximum_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - started
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        generated_tokens = int(output.shape[-1] - prompt_tokens)
        text = self._tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
        return text, prompt_tokens, generated_tokens, latency

    def _record_external(
        self,
        prompt: str,
        raw_responses: Tuple[str, ...],
        seed: int,
        valid: bool,
    ) -> str:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        record = {
            "call_index": self._call_index,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": int(seed),
            "inference_temperature": self.inference_temperature,
            "top_p": self.top_p,
            "prompt": prompt,
            "raw_responses": list(raw_responses),
            "valid": bool(valid),
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

    def decide(self, prompt: str, seed: int) -> ProviderResult:
        self._call_index += 1
        first, prompt_tokens, generated_tokens, latency = self._generate(prompt, int(seed))
        self.accounting["model_calls"] += 1
        raw = [first]
        first_valid = True
        repaired = False
        try:
            payload = self._parse_exact_json(first)
            self.accounting["first_pass_valid"] += 1
        except (ValueError, TypeError, json.JSONDecodeError) as first_error:
            first_valid = False
            repair_prompt = (
                "The prior response failed strict schema validation (%s). "
                "Return only one corrected JSON object satisfying: %s\nPRIOR_RESPONSE=%s"
                % (str(first_error)[:240], decision_schema_json(), first)
            )
            second, repair_prompt_tokens, repair_generated_tokens, repair_latency = self._generate(
                repair_prompt, int(seed) + 1000003
            )
            self.accounting["model_calls"] += 1
            raw.append(second)
            prompt_tokens += repair_prompt_tokens
            generated_tokens += repair_generated_tokens
            latency += repair_latency
            try:
                payload = self._parse_exact_json(second)
                repaired = True
                self.accounting["repaired_valid"] += 1
            except (ValueError, TypeError, json.JSONDecodeError):
                self.accounting["invalid_after_repair"] += 1
                self._record_external(prompt, tuple(raw), seed, False)
                raise ValueError("Qwen response remained invalid after one repair")
        raw_hash = self._record_external(prompt, tuple(raw), seed, True)
        self.accounting["prompt_tokens"] += int(prompt_tokens)
        self.accounting["generated_tokens"] += int(generated_tokens)
        self.accounting["latency_seconds"] += float(latency)
        return ProviderResult(
            payload=payload,
            first_pass_valid=first_valid,
            repaired=repaired,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_seconds=latency,
            raw_artifact_sha256=raw_hash,
        )

    def environment_manifest(self) -> Dict[str, object]:
        manifest: Dict[str, object] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "inference_temperature": self.inference_temperature,
            "top_p": self.top_p,
            "maximum_new_tokens": self.maximum_new_tokens,
            "quantization": "NF4, double quantization, BF16 compute",
            "backend": "Transformers AutoModelForCausalLM",
            "accounting": dict(self.accounting),
        }
        try:
            import bitsandbytes
            import torch
            import transformers

            manifest.update(
                {
                    "bitsandbytes_version": bitsandbytes.__version__,
                    "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
            )
        except ImportError:
            manifest["runtime_dependencies_available"] = False
        return manifest
