"""Nominal-only calibration and development direction diagnosis for DOET."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .environment import ScenarioConfig
from .runner import EpisodeRunner


APPLICATION_SIZES = {"commercial": 11, "humanitarian": 10}
DEVELOPMENT_REGIMES = (
    ("moderate", "reliable"),
    ("correlated", "intermittent"),
    ("compound", "partition"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _analysis_environment() -> Dict[str, str]:
    packages = (
        "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "torch",
    )
    values = {"python": platform.python_version()}
    for package in packages:
        try:
            values[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:
            values[package.replace("-", "_")] = "not-installed"
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _monitor_only_parameters() -> Dict[str, Any]:
    return {
        "trigger_type": "cusum",
        "direction": "absolute",
        "nominal_center": 0.5,
        "nominal_scale": 1.0,
        "rho": 0.0,
        "kappa": 0.0,
        "tau_on": 1_000_000.0,
        "tau_off": 999_999.0,
        "tau_crisis": 2_000_000.0,
        "minimum_dwell": 2,
        "cooldown": 0,
        "crisis_surprisal": 1_000_000.0,
        "propagation": "local",
        "quiet_decision_interval": 24,
        "quiet_gossip_rounds": 1,
        "quiet_gossip_period": 8,
    }


def _episode_rows(
    application: str,
    seed: int,
    disruption: str,
    communication: str,
    normalizers: Mapping[str, Any] = None,
    horizon: int = 24,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = ScenarioConfig(
        application=application,
        seed=int(seed),
        horizon=int(horizon),
        n_agents=APPLICATION_SIZES[application],
        private_information=0.8,
        objective_misalignment=0.8,
        communication=communication,
        disruption=disruption,
        decision_interval=4,
        communication_budget=200,
        topology="ring_plus_hubs",
    )
    runner = EpisodeRunner(
        config,
        "doet_rule",
        trigger_config=_monitor_only_parameters(),
        trigger_normalizers=normalizers,
    )
    result = runner.run(
        "doet-calibration-%s-%s-%s-s%d"
        % (application, disruption, communication, seed)
    )
    role_by_agent = {
        agent_id: agent.identity.role
        for agent_id, agent in runner.env.agents.items()
    }
    disruption_step = max(2, horizon // 3)
    rows: List[Dict[str, Any]] = []
    prior: Dict[str, float] = {}
    local_kpis: Dict[Tuple[str, int], float] = {}
    for event in runner.env.ledger.events:
        if event.kind != "observation_delivery":
            continue
        observation = event.payload["observation"]
        pressure = max(
            float(observation["backlog"])
            / max(float(observation["local_forecast"]), 1.0),
            float(observation["service_shortfall"]),
        )
        strain = 0.6 * float(observation["commitment_strain"]) + 0.4 * (
            1.0 - float(observation["communication_reliability"])
        )
        local_kpis[(str(event.payload["recipient"]), event.step)] = max(
            pressure,
            float(observation["impairment"]),
            strain,
        )
    for event in runner.env.ledger.events:
        if event.kind != "coordination_trigger":
            continue
        entropy = float(event.payload["signal"])
        previous = prior.get(event.actor, entropy)
        rows.append({
            "application": application,
            "role": role_by_agent[event.actor],
            "agent_id": event.actor,
            "topology": config.topology,
            "communication": communication,
            "disruption": disruption,
            "seed": seed,
            "step": event.step,
            "disruption_step": disruption_step,
            "disruption_active": int(
                disruption != "nominal" and event.step >= disruption_step
            ),
            "distributed_entropy": entropy,
            "entropy_change": entropy - previous,
            "local_kpi_composite": local_kpis[(event.actor, event.step)],
            "consensus_disagreement": float(
                event.payload["consensus_disagreement"]
            ),
        })
        prior[event.actor] = entropy
    return rows, {
        "run_id": result.run_id,
        "application": application,
        "seed": seed,
        "disruption": disruption,
        "communication": communication,
        "primary_outcome": result.metrics["primary_outcome"],
        "monitor_sketch_messages": result.metrics["monitor_sketch_messages"],
        "conservation_error": result.metrics["conservation_error"],
        "ledger_digest": runner.env.ledger.digest(),
    }


def _fit_normalizers(
    rows: Sequence[Mapping[str, Any]],
    field: str = "distributed_entropy",
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"applications": {}}
    for application in sorted({str(row["application"]) for row in rows}):
        application_rows = [
            row for row in rows if row["application"] == application
        ]
        values = np.asarray(
            [row[field] for row in application_rows],
            dtype=float,
        )
        app = {
            "default": {
                "center": float(values.mean()),
                "scale": float(max(values.std(ddof=1), 0.02)),
                "raw_standard_deviation": float(values.std(ddof=1)),
                "observations": int(len(values)),
            },
            "roles": {},
        }
        for role in sorted({str(row["role"]) for row in application_rows}):
            role_values = np.asarray([
                row[field]
                for row in application_rows if row["role"] == role
            ], dtype=float)
            app["roles"][role] = {
                "center": float(role_values.mean()),
                "scale": float(max(role_values.std(ddof=1), 0.02)),
                "raw_standard_deviation": float(role_values.std(ddof=1)),
                "observations": int(len(role_values)),
            }
        output["applications"][application] = app
    return output


def _normalizer(
    normalizers: Mapping[str, Any], application: str, role: str
) -> Tuple[float, float]:
    app = normalizers["applications"][application]
    row = app["roles"].get(role, app["default"])
    return float(row["center"]), float(row["scale"])


def _direction_diagnostics(
    nominal_rows: Sequence[Mapping[str, Any]],
    disrupted_rows: Sequence[Mapping[str, Any]],
    normalizers: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    combined = list(nominal_rows) + list(disrupted_rows)
    output: List[Dict[str, Any]] = []
    for application in sorted(APPLICATION_SIZES):
        app_rows = [row for row in combined if row["application"] == application]
        labels = np.asarray([row["disruption_active"] for row in app_rows], dtype=int)
        if labels.min() == labels.max():
            raise ValueError("direction diagnosis requires both classes")
        standardized = []
        change = []
        for row in app_rows:
            center, scale = _normalizer(
                normalizers, application, str(row["role"])
            )
            standardized.append(
                (float(row["distributed_entropy"]) - center) / scale
            )
            change.append(abs(float(row["entropy_change"])) / scale)
        z = np.asarray(standardized, dtype=float)
        scores = {
            "high": z,
            "low": -z,
            "absolute": np.abs(z),
            "change": np.asarray(change, dtype=float),
        }
        nominal_mask = labels == 0
        for direction, score in scores.items():
            threshold = float(np.quantile(score[nominal_mask], 0.95))
            predicted = score >= threshold
            output.append({
                "application": application,
                "direction": direction,
                "timepoints": int(len(labels)),
                "positive_prevalence": float(labels.mean()),
                "average_precision": float(average_precision_score(labels, score)),
                "roc_auc": float(roc_auc_score(labels, score)),
                "nominal_p95_threshold": threshold,
                "false_alarm_rate": float(predicted[nominal_mask].mean()),
                "detection_recall": float(predicted[labels == 1].mean()),
            })
    for row in output:
        peers = [
            value for value in output
            if value["direction"] == row["direction"]
        ]
        row["cross_application_mean_ap"] = float(np.mean([
            value["average_precision"] for value in peers
        ]))
        row["cross_application_min_ap"] = float(min(
            value["average_precision"] for value in peers
        ))
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(
    output_root: Path,
    nominal_seeds: Iterable[int] = range(5101, 5107),
    development_seeds: Iterable[int] = range(5201, 5204),
    horizon: int = 24,
) -> Dict[str, Any]:
    """Fit nominal statistics, then diagnose direction on development only."""

    nominal_seeds = tuple(int(value) for value in nominal_seeds)
    development_seeds = tuple(int(value) for value in development_seeds)
    output_root.mkdir(parents=True, exist_ok=True)
    nominal_rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    for application in APPLICATION_SIZES:
        for seed in nominal_seeds:
            rows, record = _episode_rows(
                application, int(seed), "nominal", "reliable", horizon=horizon
            )
            nominal_rows.extend(rows)
            run_records.append(record)
    normalizers = _fit_normalizers(nominal_rows)
    kpi_normalizers = _fit_normalizers(
        nominal_rows, field="local_kpi_composite"
    )

    development_rows: List[Dict[str, Any]] = []
    for application in APPLICATION_SIZES:
        for seed in development_seeds:
            for disruption, communication in DEVELOPMENT_REGIMES:
                rows, record = _episode_rows(
                    application,
                    int(seed),
                    disruption,
                    communication,
                    normalizers=normalizers,
                    horizon=horizon,
                )
                development_rows.extend(rows)
                run_records.append(record)
    diagnostics = _direction_diagnostics(
        nominal_rows, development_rows, normalizers
    )
    ranked = sorted(
        {
            row["direction"]: (
                row["cross_application_min_ap"],
                row["cross_application_mean_ap"],
            )
            for row in diagnostics
        }.items(),
        key=lambda item: (item[1][0], item[1][1], item[0]),
        reverse=True,
    )
    calibration = {
        "status": "development calibration; not a holdout result",
        "nominal_source": {
            "seeds": list(nominal_seeds),
            "applications": sorted(APPLICATION_SIZES),
            "disruption": "nominal",
            "communication": "reliable",
            "horizon": horizon,
        },
        "scale_floor": 0.02,
        "normalizers": normalizers,
        "kpi_normalizers": kpi_normalizers,
        "direction_diagnosis": {
            "development_seeds": list(development_seeds),
            "regimes": [list(value) for value in DEVELOPMENT_REGIMES],
            "ranking_rule": (
                "maximize minimum application AP, then mean application AP; "
                "final direction remains subject to preregistered validation"
            ),
            "development_leader": ranked[0][0],
            "ranking": [
                {
                    "direction": direction,
                    "minimum_application_ap": values[0],
                    "mean_application_ap": values[1],
                }
                for direction, values in ranked
            ],
        },
        "generated_at": _utc_now(),
    }
    calibration_path = output_root / "trigger_nominal_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    nominal_path = output_root / "nominal_distributed_entropy.csv"
    development_path = output_root / "development_distributed_entropy.csv"
    direction_path = output_root / "direction_diagnostics.csv"
    runs_path = output_root / "calibration_runs.csv"
    _write_csv(nominal_path, nominal_rows)
    _write_csv(development_path, development_rows)
    _write_csv(direction_path, diagnostics)
    _write_csv(runs_path, run_records)
    manifest = {
        "status": "complete",
        "analysis_environment": _analysis_environment(),
        "generated_at": _utc_now(),
        "nominal_episodes": len(APPLICATION_SIZES) * len(nominal_seeds),
        "development_episodes": (
            len(APPLICATION_SIZES)
            * len(development_seeds)
            * len(DEVELOPMENT_REGIMES)
        ),
        "monitor_only_trigger_parameters": _monitor_only_parameters(),
        "outputs": {
            str(path.name): _sha256(path)
            for path in (
                calibration_path,
                nominal_path,
                development_path,
                direction_path,
                runs_path,
            )
        },
        "maximum_absolute_conservation_error": float(max(
            abs(float(row["conservation_error"])) for row in run_records
        )),
    }
    manifest_path = output_root / "calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "development_leader": ranked[0][0]}
