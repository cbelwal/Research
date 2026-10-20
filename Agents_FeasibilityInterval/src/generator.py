from __future__ import annotations

import random
from dataclasses import dataclass

import CConfig

from .models import AgentSpec


@dataclass(frozen=True)
class GeneratedSystem:
    system_id: int
    agent_count: int
    load_profile: str
    agents: tuple[AgentSpec, ...]


def generate_systems() -> list[GeneratedSystem]:
    rng = random.Random(CConfig.RANDOM_SEED)
    systems: list[GeneratedSystem] = []
    system_id = 1
    profiles = tuple(CConfig.LOAD_PROFILES)

    for agent_count, system_count in CConfig.SYSTEMS_PER_AGENT_COUNT.items():
        for local_index in range(system_count):
            profile = profiles[local_index % len(profiles)]
            agents = _generate_agents(rng, agent_count, profile)
            systems.append(
                GeneratedSystem(
                    system_id=system_id,
                    agent_count=agent_count,
                    load_profile=profile,
                    agents=tuple(agents),
                )
            )
            system_id += 1
    return systems


def _generate_agents(
    rng: random.Random,
    count: int,
    profile: str,
) -> list[AgentSpec]:
    period_ticks = [
        rng.choice(CConfig.PERIOD_SECONDS) * CConfig.TIME_SCALE
        for _ in range(count)
    ]
    period_ticks.sort(reverse=True)
    ratio_low, ratio_high = CConfig.LOAD_PROFILES[profile]
    agents: list[AgentSpec] = []

    for index, period in enumerate(period_ticks):
        processing = max(3, round(period * rng.uniform(ratio_low, ratio_high)))
        context_fraction = rng.uniform(*CConfig.CONTEXT_FRACTION_RANGE)
        commit_fraction = rng.uniform(*CConfig.COMMIT_FRACTION_RANGE)
        context = max(1, round(processing * context_fraction))
        commit = max(1, round(processing * commit_fraction))
        execution = max(1, processing - context - commit)
        processing = context + execution + commit
        offset = rng.randrange(0, period)
        priority = index + 1
        agents.append(
            AgentSpec(
                agent_id=priority,
                priority=priority,
                period=period,
                offset=offset,
                context_time=context,
                execution_time=execution,
                commit_time=commit,
            )
        )
    return agents

