from __future__ import annotations

import sqlite3
from pathlib import Path

from .generator import GeneratedSystem
from .models import SimulationResult


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS systems (
    system_id INTEGER PRIMARY KEY,
    agent_count INTEGER NOT NULL,
    load_profile TEXT NOT NULL,
    hyperperiod_ticks INTEGER NOT NULL,
    rmax_ticks INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    system_id INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,
    priority INTEGER NOT NULL,
    period_ticks INTEGER NOT NULL,
    offset_ticks INTEGER NOT NULL,
    context_ticks INTEGER NOT NULL,
    execution_ticks INTEGER NOT NULL,
    commit_ticks INTEGER NOT NULL,
    processing_ticks INTEGER NOT NULL,
    PRIMARY KEY (system_id, agent_id),
    FOREIGN KEY (system_id) REFERENCES systems(system_id)
);

CREATE TABLE IF NOT EXISTS system_results (
    system_id INTEGER PRIMARY KEY,
    schedulable INTEGER NOT NULL,
    periodicity_match INTEGER,
    periodicity_windows INTEGER NOT NULL,
    missed_agent_id INTEGER,
    miss_time_ticks INTEGER,
    total_jobs INTEGER NOT NULL,
    total_aborts INTEGER NOT NULL,
    total_waste_ticks INTEGER NOT NULL,
    FOREIGN KEY (system_id) REFERENCES systems(system_id)
);

CREATE TABLE IF NOT EXISTS agent_results (
    system_id INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,
    completed_jobs INTEGER NOT NULL,
    missed_jobs INTEGER NOT NULL,
    max_response_ticks INTEGER,
    max_observed_aborts INTEGER NOT NULL,
    retry_bound INTEGER,
    retry_bound_violations INTEGER,
    total_waste_ticks INTEGER NOT NULL,
    max_waste_ticks INTEGER NOT NULL,
    waste_bound_ticks INTEGER,
    waste_bound_violations INTEGER,
    mean_waste_fraction REAL,
    commit_waste_ticks INTEGER NOT NULL,
    PRIMARY KEY (system_id, agent_id),
    FOREIGN KEY (system_id, agent_id) REFERENCES agents(system_id, agent_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    system_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    release_ticks INTEGER NOT NULL,
    deadline_ticks INTEGER NOT NULL,
    first_start_ticks INTEGER,
    completion_ticks INTEGER,
    response_ticks INTEGER,
    attempts INTEGER NOT NULL,
    aborts INTEGER NOT NULL,
    waste_ticks INTEGER NOT NULL,
    context_waste_ticks INTEGER NOT NULL,
    execution_waste_ticks INTEGER NOT NULL,
    commit_waste_ticks INTEGER NOT NULL,
    missed_deadline INTEGER NOT NULL,
    PRIMARY KEY (system_id, job_id)
);

CREATE TABLE IF NOT EXISTS sensitivity_runs (
    sweep TEXT NOT NULL,
    parameter_label TEXT NOT NULL,
    parameter_value REAL NOT NULL,
    replication INTEGER NOT NULL,
    schedulable INTEGER NOT NULL,
    response_ticks INTEGER,
    aborts INTEGER,
    waste_ticks INTEGER,
    waste_fraction REAL,
    context_waste_ticks INTEGER,
    execution_waste_ticks INTEGER,
    commit_waste_ticks INTEGER
);
"""


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def reset_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    return connect(path)


def insert_system(
    connection: sqlite3.Connection,
    system: GeneratedSystem,
    hyperperiod: int,
    rmax: int,
) -> None:
    connection.execute(
        "INSERT INTO systems VALUES (?, ?, ?, ?, ?)",
        (
            system.system_id,
            system.agent_count,
            system.load_profile,
            hyperperiod,
            rmax,
        ),
    )
    connection.executemany(
        """
        INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                system.system_id,
                agent.agent_id,
                agent.priority,
                agent.period,
                agent.offset,
                agent.context_time,
                agent.execution_time,
                agent.commit_time,
                agent.processing_time,
            )
            for agent in system.agents
        ],
    )


def insert_jobs(
    connection: sqlite3.Connection,
    system_id: int,
    result: SimulationResult,
) -> None:
    connection.executemany(
        """
        INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                system_id,
                job.job_id,
                job.agent.agent_id,
                job.sequence,
                job.release_time,
                job.deadline_time,
                job.first_start_time,
                job.completion_time,
                job.response_time,
                job.attempts,
                job.abort_count,
                job.wasted_processing,
                job.wasted_context,
                job.wasted_execution,
                job.wasted_commit,
                int(job.missed_deadline),
            )
            for job in result.jobs
        ],
    )

