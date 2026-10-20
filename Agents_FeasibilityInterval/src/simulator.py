from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import (
    AgentSpec,
    EventRecord,
    JobRecord,
    RunningAttempt,
    SimulationResult,
    TraceSegment,
)


class AbortRestartSimulator:
    """Discrete-event simulator implementing the paper's timestamp ordering."""

    def __init__(self, agents: Iterable[AgentSpec], stop_on_miss: bool = True):
        self.agents = sorted(agents, key=lambda item: item.priority)
        priorities = [agent.priority for agent in self.agents]
        if len(priorities) != len(set(priorities)):
            raise ValueError("Agent priorities must be unique.")
        self.stop_on_miss = stop_on_miss

    def run(self, release_until: int, horizon: int) -> SimulationResult:
        releases = self._build_release_schedule(release_until)
        release_times = sorted(releases)
        release_index = 0
        time = 0
        next_job_id = 1
        active: list[JobRecord] = []
        jobs: list[JobRecord] = []
        segments: list[TraceSegment] = []
        events: list[EventRecord] = []
        running: RunningAttempt | None = None
        schedulable = True
        missed_job: JobRecord | None = None

        while time <= horizon:
            completion_time = None
            if running is not None:
                remaining = running.job.agent.processing_time - running.progress
                completion_time = time + remaining

            next_release = (
                release_times[release_index]
                if release_index < len(release_times)
                else None
            )
            deadlines = [
                job.deadline_time
                for job in active
                if job.completion_time is None and job.deadline_time >= time
            ]
            next_deadline = min(deadlines) if deadlines else None
            candidates = [
                candidate
                for candidate in (completion_time, next_release, next_deadline, horizon)
                if candidate is not None and candidate >= time
            ]
            if not candidates:
                break
            event_time = min(candidates)

            if running is not None and event_time > time:
                self._consume(running, time, event_time, segments)
            time = event_time

            # 1. Successful completion and atomic publication.
            if (
                running is not None
                and running.progress == running.job.agent.processing_time
            ):
                job = running.job
                job.completion_time = time
                events.append(
                    EventRecord(
                        time,
                        "complete",
                        job.agent.agent_id,
                        job.job_id,
                        running.attempt_number,
                        "commit",
                    )
                )
                active.remove(job)
                running = None

            # 2. Deadline checks. A completion at this timestamp already won.
            due_jobs = [
                job
                for job in active
                if job.completion_time is None and job.deadline_time == time
            ]
            if due_jobs:
                due_jobs.sort(key=lambda job: job.agent.priority, reverse=True)
                missed_job = due_jobs[0]
                missed_job.missed_deadline = True
                events.append(
                    EventRecord(
                        time,
                        "deadline_miss",
                        missed_job.agent.agent_id,
                        missed_job.job_id,
                        detail="unfinished at relative deadline",
                    )
                )
                schedulable = False
                if self.stop_on_miss:
                    break

            # 3. Batch all releases at this timestamp.
            released_now: list[JobRecord] = []
            if next_release == time:
                for agent, sequence in releases[time]:
                    job = JobRecord(
                        job_id=next_job_id,
                        agent=agent,
                        sequence=sequence,
                        release_time=time,
                        deadline_time=time + agent.relative_deadline,
                    )
                    next_job_id += 1
                    jobs.append(job)
                    active.append(job)
                    released_now.append(job)
                    events.append(
                        EventRecord(
                            time,
                            "release",
                            agent.agent_id,
                            job.job_id,
                            detail=f"sequence={sequence}",
                        )
                    )
                release_index += 1

            ready = [
                job
                for job in active
                if job.completion_time is None
                and (running is None or job.job_id != running.job.job_id)
            ]
            highest_ready = max(
                ready,
                key=lambda job: job.agent.priority,
                default=None,
            )

            # A release batch can abort the current attempt at most once.
            if (
                running is not None
                and highest_ready is not None
                and highest_ready.agent.priority > running.job.agent.priority
            ):
                self._abort(running, time, events)
                running = None

            # 4. Dispatch the highest-priority ready job.
            if running is None:
                ready = [
                    job
                    for job in active
                    if job.completion_time is None
                ]
                if ready:
                    job = max(ready, key=lambda item: item.agent.priority)
                    job.attempts += 1
                    if job.first_start_time is None:
                        job.first_start_time = time
                    running = RunningAttempt(
                        job=job,
                        attempt_number=job.attempts,
                        start_time=time,
                    )
                    events.append(
                        EventRecord(
                            time,
                            "attempt_start",
                            job.agent.agent_id,
                            job.job_id,
                            running.attempt_number,
                            "context",
                        )
                    )

            if time == horizon:
                break

            if (
                running is None
                and release_index >= len(release_times)
                and not active
            ):
                break

        return SimulationResult(
            schedulable=schedulable,
            end_time=time,
            jobs=jobs,
            segments=segments,
            events=events,
            missed_job=missed_job,
        )

    def _build_release_schedule(
        self,
        release_until: int,
    ) -> dict[int, list[tuple[AgentSpec, int]]]:
        releases: dict[int, list[tuple[AgentSpec, int]]] = defaultdict(list)
        for agent in self.agents:
            sequence = 1
            release = agent.offset
            while release < release_until:
                releases[release].append((agent, sequence))
                sequence += 1
                release += agent.period
        for batch in releases.values():
            batch.sort(key=lambda item: item[0].priority, reverse=True)
        return releases

    @staticmethod
    def _phase_at(agent: AgentSpec, progress: int) -> str:
        if progress < agent.context_time:
            return "context"
        if progress < agent.context_time + agent.execution_time:
            return "execution"
        return "commit"

    def _consume(
        self,
        running: RunningAttempt,
        start: int,
        end: int,
        segments: list[TraceSegment],
    ) -> None:
        cursor_time = start
        target_progress = running.progress + (end - start)
        agent = running.job.agent
        boundaries = (
            ("context", agent.context_time),
            ("execution", agent.context_time + agent.execution_time),
            ("commit", agent.processing_time),
        )
        for phase, boundary in boundaries:
            if running.progress >= boundary:
                continue
            phase_end_progress = min(boundary, target_progress)
            duration = phase_end_progress - running.progress
            if duration <= 0:
                continue
            segments.append(
                TraceSegment(
                    cursor_time,
                    cursor_time + duration,
                    agent.agent_id,
                    agent.priority,
                    running.job.job_id,
                    running.attempt_number,
                    phase,
                )
            )
            if phase == "context":
                running.context_consumed += duration
            elif phase == "execution":
                running.execution_consumed += duration
            else:
                running.commit_consumed += duration
            running.progress += duration
            cursor_time += duration
            if running.progress == target_progress:
                break

    def _abort(
        self,
        running: RunningAttempt,
        time: int,
        events: list[EventRecord],
    ) -> None:
        job = running.job
        job.abort_count += 1
        job.wasted_processing += running.progress
        job.wasted_context += running.context_consumed
        job.wasted_execution += running.execution_consumed
        job.wasted_commit += running.commit_consumed
        events.append(
            EventRecord(
                time,
                "abort",
                job.agent.agent_id,
                job.job_id,
                running.attempt_number,
                self._phase_at(job.agent, running.progress),
                detail=f"consumed={running.progress}",
            )
        )

