"""Illustrative nonequilibrium mappings to two autonomous-agent applications.

These mappings are deliberately lower fidelity than the abstract model.  They
share the local belief/action microdynamics but use distinct external protocols
and physical bookkeeping: multi-commodity shortage/service for humanitarian
coordination and precedence-constrained component restoration for utilities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .information import shannon_entropy
from .model import Microstate, ModelParameters, MultiplexModel
from .simulate import directed_communication, make_model_layers


def economic_gini(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if values.sum() <= 0.0:
        return 0.0
    ordered = np.sort(values)
    n = ordered.size
    return float((2.0 * np.sum(np.arange(1, n + 1) * ordered) / (n * ordered.sum())) - (n + 1.0) / n)


def workload_conservation_residual(
    initial_workload: float,
    exogenous_injection: float,
    endogenous_cascade: float,
    remaining_workload: float,
    completed_work: float,
) -> float:
    """Independent balance reconstructed from event-category accumulators."""

    return float(
        initial_workload
        + exogenous_injection
        + endogenous_cascade
        - remaining_workload
        - completed_work
    )


def _information_entropy(spins: np.ndarray) -> float:
    positive = float(np.mean(spins > 0))
    return shannon_entropy(np.array([positive, 1.0 - positive])) if 0.0 < positive < 1.0 else 0.0


def _diagnostic_energy(
    communication: np.ndarray,
    dependency: np.ndarray,
    parameters: ModelParameters,
    state: Microstate,
    private_fields: np.ndarray,
    task_fields: np.ndarray,
) -> float:
    symmetric_comm = 0.5 * (communication + communication.T)
    symmetric_dep = 0.5 * (dependency + dependency.T)
    equilibrium_parameters = ModelParameters(
        belief_coupling=parameters.belief_coupling,
        action_coupling=parameters.action_coupling,
        belief_action_coupling=parameters.belief_action_coupling,
        temperature=parameters.temperature,
    )
    return MultiplexModel(
        symmetric_comm, symmetric_dep, equilibrium_parameters, private_fields, task_fields
    ).energy(state)


def run_application_mapping(
    application: str,
    n_agents: int,
    seed: int,
    horizon: int = 120,
    topology: str = "modular",
) -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray]]:
    """Run a driven humanitarian or cyber-utility trajectory."""

    if application not in ("humanitarian", "utility"):
        raise ValueError("application must be humanitarian or utility")
    rng = np.random.default_rng(seed)
    base_comm, dependency = make_model_layers(n_agents, topology, seed + 11, 1.0)
    communication = base_comm.copy()
    parameters = ModelParameters(
        belief_coupling=0.45,
        action_coupling=0.45,
        belief_action_coupling=0.60,
        temperature=2.1,
        memory_coupling=0.12,
        memory_rate=0.10,
    )
    beliefs = rng.choice([-1, 1], n_agents).astype(np.int8)
    actions = -np.ones(n_agents, dtype=np.int8)
    memory = beliefs.astype(float)
    workload = np.zeros(n_agents, dtype=float)
    state = Microstate(beliefs, actions, memory, workload)
    private_fields = rng.normal(0.0, 0.08, n_agents)
    task_fields = -0.35 * np.ones(n_agents)
    resources = np.full(n_agents, 1.0 if application == "humanitarian" else 0.0)
    restored = np.ones(n_agents, dtype=bool)
    cascade_depth = np.zeros(n_agents, dtype=int)
    rows: List[Dict[str, float]] = []
    snapshot: Dict[str, np.ndarray] = {}
    drive_start = horizon // 4
    drive_end = 3 * horizon // 5
    partition_end = drive_end + horizon // 8
    affected = np.arange(max(2, n_agents // 5))
    corrupted = np.arange(n_agents // 3, n_agents // 3 + max(2, n_agents // 8))
    protocol_work = 0.0
    entropy_flow_epoch = 0.0
    previous_private = private_fields.copy()
    previous_task = task_fields.copy()
    initial_workload = float(np.sum(workload))
    initial_resources = float(np.sum(resources))
    exogenous_injection = 0.0
    endogenous_cascade = 0.0
    completed_work = 0.0
    consumed_resources = 0.0

    for time_step in range(horizon):
        if time_step == drive_start:
            if application == "humanitarian":
                # Correlated demand shock across regions and commodity proxies.
                injection = rng.uniform(1.4, 2.2, affected.size)
                workload[affected] += injection
                exogenous_injection += float(np.sum(injection))
                task_fields[affected] += 1.25
                private_fields[affected] += rng.normal(0.9, 0.25, affected.size)
            else:
                # Abstract cyber-physical event: failures plus confidently
                # corrupted local telemetry; no offensive procedure is modeled.
                restored[affected] = False
                injection = rng.uniform(1.0, 1.8, affected.size)
                workload[affected] += injection
                exogenous_injection += float(np.sum(injection))
                task_fields[affected] += 1.1
                private_fields[affected] += 0.9
                private_fields[corrupted] *= -2.0
                communication = directed_communication(base_comm, 0.55, seed + 71)
        if drive_start <= time_step < drive_end and time_step % 9 == 0:
            if application == "humanitarian":
                targets = rng.choice(n_agents, max(1, n_agents // 12), replace=False)
                injection = rng.uniform(0.25, 0.60, targets.size)
                workload[targets] += injection
                exogenous_injection += float(np.sum(injection))
            else:
                failed = rng.choice(n_agents, max(1, n_agents // 16), replace=False)
                injection = rng.uniform(0.20, 0.55, failed.size)
                workload[failed] += injection
                exogenous_injection += float(np.sum(injection))
                restored[failed] = False
        if time_step == drive_start + 5:
            half = n_agents // 2
            communication[:half, half:] = 0.0
            communication[half:, :half] = 0.0
        if time_step == partition_end:
            communication = base_comm.copy()

        protocol_work += float(
            -np.dot(private_fields - previous_private, beliefs)
            -np.dot(task_fields - previous_task, actions)
        )
        previous_private[:] = private_fields
        previous_task[:] = task_fields
        comm_sum = communication.dot(beliefs)
        dep_sum = dependency.dot(actions)
        entropy_flow_epoch = 0.0
        changes = 0
        for _ in range(2 * n_agents):
            variable = int(rng.integers(2 * n_agents))
            if variable < n_agents:
                index = variable
                field = (
                    parameters.belief_coupling * comm_sum[index]
                    + parameters.belief_action_coupling * actions[index]
                    + private_fields[index]
                    + parameters.memory_coupling * memory[index]
                )
                values = beliefs
            else:
                index = variable - n_agents
                field = (
                    parameters.action_coupling * dep_sum[index]
                    + parameters.belief_action_coupling * beliefs[index]
                    + task_fields[index]
                    + workload[index]
                )
                values = actions
            probability_plus = 1.0 / (1.0 + np.exp(-np.clip(2.0 * field / parameters.temperature, -700, 700)))
            old = int(values[index])
            new = 1 if rng.random() < probability_plus else -1
            forward = probability_plus if new == 1 else 1.0 - probability_plus
            reverse = probability_plus if old == 1 else 1.0 - probability_plus
            entropy_flow_epoch += float(np.log(max(forward, 1e-300) / max(reverse, 1e-300)))
            if new != old:
                values[index] = new
                changes += 1
                if variable < n_agents:
                    comm_sum += communication[:, index] * (new - old)
                else:
                    dep_sum += dependency[:, index] * (new - old)
            if variable < n_agents:
                memory[index] = (1.0 - parameters.memory_rate) * memory[index] + parameters.memory_rate * new

        active = np.flatnonzero((actions > 0) & (workload > 0.0))
        if application == "humanitarian":
            # Shared vehicle/fuel capacity constrains how many local proposals
            # can become service in one epoch.
            capacity = max(1, n_agents // 7)
            selected = active[np.argsort(workload[active])[-capacity:]] if active.size else active
            delivered = np.minimum(np.minimum(workload[selected], 0.28), resources[selected])
            workload[selected] -= delivered
            resources[selected] = np.maximum(0.0, resources[selected] - delivered)
            completed_work += float(np.sum(delivered))
            consumed_resources += float(np.sum(delivered))
            # Unresolved shortages spill over through dependency edges.
            overloaded = np.flatnonzero(workload > 1.6)
            for source in overloaded:
                neighbors = np.flatnonzero(dependency[source])
                if neighbors.size:
                    target = int(neighbors[(time_step + source) % neighbors.size])
                    workload[target] += 0.025
                    endogenous_cascade += 0.025
                    cascade_depth[target] = max(cascade_depth[target], cascade_depth[source] + 1)
            service = float(np.sum(delivered))
            service_loss = float(np.sum(workload))
        else:
            # Shared crews implement at most N/10 accepted repair proposals.
            crew_capacity = max(1, n_agents // 10)
            candidates = active[~restored[active]] if active.size else active
            selected = candidates[np.argsort(workload[candidates])[-crew_capacity:]] if candidates.size else candidates
            repaired = np.minimum(workload[selected], 0.22)
            workload[selected] -= repaired
            completed_work += float(np.sum(repaired))
            restored[selected[workload[selected] <= 0.10]] = True
            # Failed components cause topology-dependent secondary outages.
            failed_nodes = np.flatnonzero(~restored)
            for source in failed_nodes:
                if workload[source] < 1.25:
                    continue
                neighbors = np.flatnonzero(dependency[source])
                for target in neighbors[:2]:
                    if rng.random() < 0.035:
                        restored[target] = False
                        workload[target] += 0.08
                        endogenous_cascade += 0.08
                        cascade_depth[target] = max(cascade_depth[target], cascade_depth[source] + 1)
            service = float(np.sum(repaired))
            service_loss = float(np.sum(~restored) + 0.5 * np.sum(workload))

        if time_step > partition_end:
            private_fields *= 0.985
            task_fields += 0.015 * (-0.35 - task_fields)
        state.workload[:] = workload
        diagnostic_energy = _diagnostic_energy(
            communication, dependency, parameters, state, private_fields, task_fields
        )
        rows.append(
            {
                "time": float(time_step),
                "application_code": 0.0 if application == "humanitarian" else 1.0,
                "drive_active": float(drive_start <= time_step < drive_end),
                "partition_active": float(drive_start + 5 <= time_step < partition_end),
                "belief_order": float(np.mean(beliefs)),
                "action_order": float(np.mean(actions)),
                "belief_action_consistency": float(np.mean(beliefs * actions)),
                "belief_entropy": _information_entropy(beliefs),
                "action_entropy": _information_entropy(actions),
                "entropy_flow_per_update": float(entropy_flow_epoch / (2 * n_agents)),
                "diagnostic_energy_per_agent": float(diagnostic_energy / n_agents),
                "protocol_work_per_agent": float(protocol_work / n_agents),
                "workload_density": float(np.mean(workload)),
                "service_loss": service_loss,
                "service_completed": service,
                "allocation_gini": economic_gini(workload),
                "cascade_depth": float(np.max(cascade_depth)),
                "activity": float(changes / (2 * n_agents)),
                "communication_edges": float(np.count_nonzero(communication)),
            }
        )
        if time_step == drive_start + 10:
            snapshot = {
                "beliefs": beliefs.copy(),
                "actions": actions.copy(),
                "workload": workload.copy(),
                "private_fields": private_fields.copy(),
                "communication": communication.copy(),
                "dependency": dependency.copy(),
                "time": np.array([time_step], dtype=int),
            }
    conservation = {
        "initial_workload": np.array([initial_workload]),
        "exogenous_injection": np.array([exogenous_injection]),
        "endogenous_cascade": np.array([endogenous_cascade]),
        "completed_work": np.array([completed_work]),
        "final_workload": np.array([np.sum(workload)]),
        "workload_residual": np.array(
            [
                workload_conservation_residual(
                    initial_workload,
                    exogenous_injection,
                    endogenous_cascade,
                    float(np.sum(workload)),
                    completed_work,
                )
            ]
        ),
        "initial_resources": np.array([initial_resources]),
        "consumed_resources": np.array([consumed_resources]),
        "final_resources": np.array([np.sum(resources)]),
        "resource_residual": np.array(
            [initial_resources - float(np.sum(resources)) - consumed_resources]
        ),
    }
    conservation.update(snapshot)
    return rows, conservation
