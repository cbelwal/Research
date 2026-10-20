"""Central configuration for the agent feasibility experiments."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "ExperimentalResults"
DATABASE_PATH = ROOT_DIR / "agent_experiments.db"
TRACE_DATA_DIR = ROOT_DIR / "ChronosTraces"
TRACE_RESULTS_DIR = OUTPUT_DIR / "ChronosTrace"

# Official Azure LLM Inference Trace 2024 release assets used by Chronos.
CHRONOS_TRACE_URLS = {
    "code": (
        "https://github.com/Azure/AzurePublicDataset/releases/download/"
        "dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv"
    ),
    "conversation": (
        "https://github.com/Azure/AzurePublicDataset/releases/download/"
        "dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv"
    ),
}

# Trace replay mapping. The Azure trace contains arrivals and token counts, but
# not agent identities, priorities, or phase timings. Conversation requests are
# mapped to high priority and code-completion requests to low priority.
# Chronos's A100/7B weights (alpha, beta, gamma) determine relative request
# costs, then all costs are normalized to the target utilization because our
# single-resource whole-request model does not implement Chronos's GPU batching.
TRACE_TIME_SCALE = 1_000_000  # integer microseconds
TRACE_CHRONOS_ALPHA_MS_PER_INPUT_TOKEN = 0.080
TRACE_CHRONOS_BETA_MS_PER_OUTPUT_TOKEN = 0.250
TRACE_CHRONOS_GAMMA_MS = 7.0
TRACE_TARGET_UTILIZATION = 0.70
TRACE_LOAD_FACTORS = (0.50, 1.00, 1.25, 1.50)
TRACE_TTFT_DEADLINE_MS = 2_000.0
TRACE_REPLAY_LIMIT = 50_000
TRACE_REPLAY_SEED = 42

RANDOM_SEED = 20261012
TIME_SCALE = 10  # ticks per second

# These groups give exactly 3,000 agent records while covering systems with
# 3 through 8 agents:
# 100*3 + 100*4 + 100*5 + 100*6 + 80*7 + 80*8 = 3,000.
SYSTEMS_PER_AGENT_COUNT = {
    3: 100,
    4: 100,
    5: 100,
    6: 100,
    7: 80,
    8: 80,
}
TOTAL_AGENTS = sum(size * count for size, count in SYSTEMS_PER_AGENT_COUNT.items())

PERIOD_SECONDS = (10, 12, 15, 20, 24, 30, 40, 60)
LOAD_PROFILES = {
    "low": (0.03, 0.07),
    "medium": (0.07, 0.13),
    "high": (0.12, 0.22),
}
CONTEXT_FRACTION_RANGE = (0.10, 0.25)
COMMIT_FRACTION_RANGE = (0.10, 0.25)

PERIODICITY_WINDOWS = 5
STORE_DETAILED_JOBS = True
STOP_ON_DEADLINE_MISS = True

SENSITIVITY_REPLICATIONS = 100
SENSITIVITY_HP_COUNTS = tuple(range(1, 7))
SENSITIVITY_PERIOD_GROUPS = {
    "very_frequent": (6, 8, 10),
    "frequent": (8, 10, 12),
    "baseline": (10, 12, 15),
    "infrequent": (12, 15, 20),
    "very_infrequent": (15, 20, 30),
}
SENSITIVITY_OFFSET_FRACTIONS = tuple(i / 10 for i in range(10))
SENSITIVITY_COMMIT_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)

PLOT_DPI = 300
PLOT_FORMATS = ("png", "pdf")
