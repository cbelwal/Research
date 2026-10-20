from __future__ import annotations

import random
import sqlite3

import CConfig

from .models import AgentSpec
from .simulator import AbortRestartSimulator
from .utils import lcm_many


TARGET_PERIOD_SECONDS = 60
TARGET_PROCESSING_SECONDS = 4.5


def run_sensitivity_analysis(connection: sqlite3.Connection) -> None:
    rng = random.Random(CConfig.RANDOM_SEED + 1)
    rows: list[tuple] = []

    for hp_count in CConfig.SENSITIVITY_HP_COUNTS:
        for replication in range(CConfig.SENSITIVITY_REPLICATIONS):
            agents = _scenario_agents(rng, hp_count=hp_count)
            rows.append(
                _measure("higher_priority_count", str(hp_count), hp_count, replication, agents)
            )

    for label, periods in CConfig.SENSITIVITY_PERIOD_GROUPS.items():
        for replication in range(CConfig.SENSITIVITY_REPLICATIONS):
            agents = _scenario_agents(rng, hp_count=3, hp_periods=periods)
            rows.append(
                _measure(
                    "higher_priority_period",
                    label,
                    sum(periods) / len(periods),
                    replication,
                    agents,
                )
            )

    for offset_fraction in CConfig.SENSITIVITY_OFFSET_FRACTIONS:
        for replication in range(CConfig.SENSITIVITY_REPLICATIONS):
            agents = _scenario_agents(
                rng,
                hp_count=3,
                target_offset_fraction=offset_fraction,
            )
            rows.append(
                _measure(
                    "relative_offset",
                    f"{offset_fraction:.1f}",
                    offset_fraction,
                    replication,
                    agents,
                )
            )

    for commit_fraction in CConfig.SENSITIVITY_COMMIT_FRACTIONS:
        for replication in range(CConfig.SENSITIVITY_REPLICATIONS):
            agents = _scenario_agents(
                rng,
                hp_count=3,
                target_commit_fraction=commit_fraction,
            )
            rows.append(
                _measure(
                    "commit_fraction",
                    f"{commit_fraction:.2f}",
                    commit_fraction,
                    replication,
                    agents,
                )
            )

    connection.executemany(
        "INSERT INTO sensitivity_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()


def _scenario_agents(
    rng: random.Random,
    hp_count: int,
    hp_periods: tuple[int, ...] = (10, 12, 15),
    target_offset_fraction: float | None = None,
    target_commit_fraction: float = 0.20,
) -> list[AgentSpec]:
    scale = CConfig.TIME_SCALE
    total = round(TARGET_PROCESSING_SECONDS * scale)
    context = max(1, round(total * 0.10))
    commit = max(1, round(total * target_commit_fraction))
    execution = max(1, total - context - commit)
    target_period = TARGET_PERIOD_SECONDS * scale
    target_offset = (
        rng.randrange(target_period)
        if target_offset_fraction is None
        else round(target_period * target_offset_fraction)
    )
    agents = [
        AgentSpec(
            agent_id=1,
            priority=1,
            period=target_period,
            offset=target_offset,
            context_time=context,
            execution_time=execution,
            commit_time=commit,
        )
    ]

    selected_periods = [
        hp_periods[index % len(hp_periods)] for index in range(hp_count)
    ]
    selected_periods.sort(reverse=True)
    for index, period_seconds in enumerate(selected_periods, start=2):
        period = period_seconds * scale
        processing = max(3, round(period * rng.uniform(0.07, 0.14)))
        hp_context = max(1, round(processing * 0.15))
        hp_commit = max(1, round(processing * 0.15))
        hp_execution = max(1, processing - hp_context - hp_commit)
        agents.append(
            AgentSpec(
                agent_id=index,
                priority=index,
                period=period,
                offset=rng.randrange(period),
                context_time=hp_context,
                execution_time=hp_execution,
                commit_time=hp_commit,
            )
        )
    return agents


def _measure(
    sweep: str,
    label: str,
    value: float,
    replication: int,
    agents: list[AgentSpec],
) -> tuple:
    hyperperiod = lcm_many(agent.period for agent in agents)
    rmax = max(agent.offset for agent in agents)
    release_until = rmax + 3 * hyperperiod
    horizon = release_until + max(agent.period for agent in agents)
    result = AbortRestartSimulator(agents).run(release_until, horizon)
    target_jobs = [
        job
        for job in result.jobs
        if job.agent.agent_id == 1 and job.release_time >= rmax
    ]
    completed = [job for job in target_jobs if job.completion_time is not None]
    representative = max(
        completed,
        key=lambda job: (job.wasted_processing, job.response_time or 0),
        default=None,
    )
    if representative is None:
        return (
            sweep,
            label,
            value,
            replication,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    useful = representative.agent.processing_time
    waste_fraction = representative.wasted_processing / (
        representative.wasted_processing + useful
    )
    return (
        sweep,
        label,
        value,
        replication,
        int(result.schedulable),
        representative.response_time,
        representative.abort_count,
        representative.wasted_processing,
        waste_fraction,
        representative.wasted_context,
        representative.wasted_execution,
        representative.wasted_commit,
    )

