from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

from .models import EventRecord, SimulationResult


def lcm_many(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = math.lcm(result, value)
    return result


def ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def window_signature(
    result: SimulationResult,
    start: int,
    length: int,
) -> str:
    occupancy: list[tuple[int, str] | None] = [None] * length
    for segment in result.segments:
        left = max(segment.start, start)
        right = min(segment.end, start + length)
        for tick in range(left, right):
            occupancy[tick - start] = (segment.agent_id, segment.phase)

    normalized_events = []
    included_types = {"attempt_start", "abort", "complete"}
    for event in result.events:
        if start <= event.time < start + length and event.event_type in included_types:
            normalized_events.append(
                (
                    event.time - start,
                    event.event_type,
                    event.agent_id,
                    event.phase,
                )
            )

    payload = json.dumps(
        {"occupancy": occupancy, "events": normalized_events},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

