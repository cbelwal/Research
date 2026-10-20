from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    priority: int
    period: int
    offset: int
    context_time: int
    execution_time: int
    commit_time: int
    deadline: Optional[int] = None

    @property
    def processing_time(self) -> int:
        return self.context_time + self.execution_time + self.commit_time

    @property
    def relative_deadline(self) -> int:
        return self.period if self.deadline is None else self.deadline


@dataclass
class JobRecord:
    job_id: int
    agent: AgentSpec
    sequence: int
    release_time: int
    deadline_time: int
    first_start_time: Optional[int] = None
    completion_time: Optional[int] = None
    abort_count: int = 0
    wasted_processing: int = 0
    wasted_context: int = 0
    wasted_execution: int = 0
    wasted_commit: int = 0
    attempts: int = 0
    missed_deadline: bool = False

    @property
    def response_time(self) -> Optional[int]:
        if self.completion_time is None:
            return None
        return self.completion_time - self.release_time


@dataclass
class RunningAttempt:
    job: JobRecord
    attempt_number: int
    start_time: int
    progress: int = 0
    context_consumed: int = 0
    execution_consumed: int = 0
    commit_consumed: int = 0


@dataclass(frozen=True)
class TraceSegment:
    start: int
    end: int
    agent_id: int
    priority: int
    job_id: int
    attempt_number: int
    phase: str


@dataclass(frozen=True)
class EventRecord:
    time: int
    event_type: str
    agent_id: Optional[int] = None
    job_id: Optional[int] = None
    attempt_number: Optional[int] = None
    phase: Optional[str] = None
    detail: str = ""


@dataclass
class SimulationResult:
    schedulable: bool
    end_time: int
    jobs: list[JobRecord] = field(default_factory=list)
    segments: list[TraceSegment] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    missed_job: Optional[JobRecord] = None

