"""Editable inputs for the schedule examples and paper figures."""

from CConfig import TIME_SCALE


def ticks(seconds: float) -> int:
    return round(seconds * TIME_SCALE)


# Analogues of the two-task offset examples in FeasibilityInterval.pdf.
TWO_AGENT_BASE = {
    "periods": (9, 6),
    "processing": (3, 3),
}
TWO_AGENT_RELEASE_CASES = (
    {"name": "low_at_0_high_at_1", "offsets": (0, 1)},
    {"name": "low_at_2_high_at_1", "offsets": (2, 1)},
    {"name": "synchronous", "offsets": (0, 0)},
)

# Analogue of the three-task hyperperiod illustration.
THREE_AGENT_EXAMPLE = {
    "periods": (18, 12, 9),
    "processing": (2, 4, 3),
    "offsets": (0, 0, 0),
    "hyperperiods": 4,
}

# Corrected worked example from agent_feasibility.pdf.
AGENT_WORKED_EXAMPLE = {
    "periods": (40, 20, 10),
    "offsets": (6, 0, 0),
    "context": (0.5, 0.3, 0.2),
    "execution": (3.0, 2.0, 1.0),
    "commit": (1.0, 0.5, 0.1),
    "hyperperiods": 1,
}

