from __future__ import annotations

import math
import sqlite3
from collections import defaultdict

import CConfig

from .database import insert_jobs, insert_system
from .generator import GeneratedSystem
from .simulator import AbortRestartSimulator
from .utils import ceil_div, lcm_many, window_signature


def run_main_experiments(
    connection: sqlite3.Connection,
    systems: list[GeneratedSystem],
) -> None:
    for index, system in enumerate(systems, start=1):
        hyperperiod = lcm_many(agent.period for agent in system.agents)
        rmax = max(agent.offset for agent in system.agents)
        analysis_end = rmax + CConfig.PERIODICITY_WINDOWS * hyperperiod
        horizon = analysis_end + max(agent.relative_deadline for agent in system.agents)
        insert_system(connection, system, hyperperiod, rmax)

        result = AbortRestartSimulator(
            system.agents,
            stop_on_miss=CConfig.STOP_ON_DEADLINE_MISS,
        ).run(release_until=analysis_end, horizon=horizon)

        signatures = []
        periodicity_match = None
        if result.schedulable:
            signatures = [
                window_signature(result, rmax + window * hyperperiod, hyperperiod)
                for window in range(CConfig.PERIODICITY_WINDOWS)
            ]
            periodicity_match = int(len(set(signatures)) == 1)

        connection.execute(
            """
            INSERT INTO system_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                system.system_id,
                int(result.schedulable),
                periodicity_match,
                CConfig.PERIODICITY_WINDOWS,
                result.missed_job.agent.agent_id if result.missed_job else None,
                result.end_time if result.missed_job else None,
                len(result.jobs),
                sum(job.abort_count for job in result.jobs),
                sum(job.wasted_processing for job in result.jobs),
            ),
        )
        _insert_agent_results(connection, system, result, bool(periodicity_match))
        if CConfig.STORE_DETAILED_JOBS:
            insert_jobs(connection, system.system_id, result)
        if index % 25 == 0:
            connection.commit()
            print(f"Simulated {index}/{len(systems)} systems")
    connection.commit()


def _insert_agent_results(
    connection: sqlite3.Connection,
    system: GeneratedSystem,
    result,
    bound_is_valid: bool,
) -> None:
    jobs_by_agent = defaultdict(list)
    for job in result.jobs:
        jobs_by_agent[job.agent.agent_id].append(job)

    for agent in system.agents:
        agent_jobs = jobs_by_agent[agent.agent_id]
        completed = [job for job in agent_jobs if job.completion_time is not None]
        responses = [job.response_time for job in completed if job.response_time is not None]
        max_response = max(responses, default=None)
        higher = [item for item in system.agents if item.priority > agent.priority]
        retry_bound = None
        waste_bound = None
        retry_violations = None
        waste_violations = None

        if bound_is_valid and max_response is not None:
            retry_bound = sum(ceil_div(max_response, item.period) for item in higher)
            waste_bound = retry_bound * agent.processing_time
            retry_violations = sum(
                job.abort_count > retry_bound for job in completed
            )
            waste_violations = sum(
                job.wasted_processing > waste_bound for job in completed
            )

        waste_fractions = [
            job.wasted_processing
            / (job.wasted_processing + agent.processing_time)
            for job in completed
            if job.wasted_processing + agent.processing_time > 0
        ]
        connection.execute(
            """
            INSERT INTO agent_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                system.system_id,
                agent.agent_id,
                len(completed),
                sum(job.missed_deadline for job in agent_jobs),
                max_response,
                max((job.abort_count for job in agent_jobs), default=0),
                retry_bound,
                retry_violations,
                sum(job.wasted_processing for job in agent_jobs),
                max((job.wasted_processing for job in agent_jobs), default=0),
                waste_bound,
                waste_violations,
                (
                    sum(waste_fractions) / len(waste_fractions)
                    if waste_fractions
                    else None
                ),
                sum(job.wasted_commit for job in agent_jobs),
            ),
        )

