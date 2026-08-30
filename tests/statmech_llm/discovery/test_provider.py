import json

from thermoagent.statmech_llm.discovery.provider import QwenStatmechProvider


class _RepairProvider(QwenStatmechProvider):
    def __init__(self, artifact_root, repository_root):
        super().__init__(artifact_root, repository_root, maximum_new_tokens=32)
        self.temperatures = []

    def _generate(self, prompt, seed, temperature):
        del prompt, seed
        self.temperatures.append(temperature)
        if len(self.temperatures) == 1:
            return '{"invalid":true}', 12, 3, 0.1
        return json.dumps(
            {
                "belief_choice": "amber",
                "action_choice": "cobalt",
                "confidence": 0.6,
                "commitment_status": "revised",
                "memory_state": "conflict",
                "outgoing_signal": "amber",
                "tool_action": "execute_selected",
                "reason_code": "neighbor_messages",
            }
        ), 18, 20, 0.2


def test_bounded_repair_is_greedy_and_accounting_is_persisted(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    provider = _RepairProvider(tmp_path / "external", repository)
    result = provider.decide("prompt", 9, 0.72)
    assert provider.temperatures == [0.72, 0.0]
    assert result.repaired and not result.first_pass_valid
    record = json.loads(next((tmp_path / "external").glob("call_*.json")).read_text(encoding="utf-8"))
    assert record["model_calls"] == 2
    assert record["prompt_tokens"] == 30
    assert record["generated_tokens"] == 23
    assert record["repair_sampling_temperature"] == 0.0
