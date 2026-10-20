AGENT FEASIBILITY INTERVAL EXPERIMENTS
======================================

Purpose
-------
This project implements the experimental study described in Section 7 of
agent_feasibility.pdf. It uses the revised paper's exact abort-and-restart
semantics:

1. Context assembly, decoding/tool execution, and commit are preemptible.
2. A strictly higher-priority release aborts the current lower-priority
   attempt, which later restarts from context assembly.
3. All processing consumed before an abort is waste, including partial
   commit processing.
4. At a timestamp, completion is finalized first, deadlines are checked
   second, releases are batched third, and dispatch occurs last.
5. A release exactly at an attempt's completion time does not abort it.
6. The simulator stops at the first deadline miss in the baseline run.

Experiment size
---------------
CConfig.py is the single metadata/configuration file. The default study
contains exactly 3,000 agent records across systems with 3 through 8 agents:

    100 systems * 3 agents =   300 agents
    100 systems * 4 agents =   400 agents
    100 systems * 5 agents =   500 agents
    100 systems * 6 agents =   600 agents
     80 systems * 7 agents =   560 agents
     80 systems * 8 agents =   640 agents
                              ------------
                               3,000 agents

The periods are selected from a bounded set whose least common multiple
remains practical for exact hyperperiod simulation. Every schedulable system
is simulated for five complete hyperperiod windows after Rmax, with additional
time allowed for jobs released near the final analysis boundary to complete or
miss their deadlines.

Project layout
--------------
CConfig.py
    Main experiment metadata: population size, random seed, time resolution,
    parameter ranges, sensitivity settings, database path, and output formats.

ExamplePlotConfig.py
    Editable inputs for the two-agent and three-agent schedule illustrations
    inspired by FeasibilityInterval.pdf, plus the corrected worked example from
    agent_feasibility.pdf.

run_all.py
    Runs the complete reproducible pipeline.

src/models.py
    Typed data structures for agents, jobs, attempts, events, and trace
    segments.

src/simulator.py
    Event-driven fixed-priority abort-and-restart simulator.

src/generator.py
    Seeded generation of the 3,000-agent experiment population.

src/database.py
    SQLite schema and persistence helpers.

src/experiments.py
    Hyperperiod validation, deadline analysis, retry-bound checks, and
    wasted-processing-bound checks.

src/sensitivity.py
    Sensitivity sweeps over:
      - number of higher-priority agents;
      - higher-priority periods;
      - relative release offset;
      - commit fraction of the target agent's processing time.

src/plotting.py
    Publication plots and CSV tables.

src/example_plots.py
    Schedule figures controlled by ExamplePlotConfig.py.

src/chronos_trace.py
    Streaming characterization and trace-driven abort/restart replay for the
    Azure LLM Inference Trace 2024 used by Chronos.

run_chronos_trace.py
    Runs the Chronos/Azure trace study after the two official trace files have
    been downloaded into ChronosTraces/.

tests/test_simulator.py
    Regression tests for timestamp ordering, commit-phase waste, and the
    classical-response-time counterexample.

agent_experiments.db
    Generated SQLite database containing system metadata, all 3,000 agent
    specifications, job-level measurements, aggregate results, and sensitivity
    runs.

ExperimentalResults/
    Generated CSV tables and both PNG and PDF versions of every plot.

ExperimentalResults/ChronosTrace/
    Full-trace characterization, 50,000-request real-arrival replay results,
    policy-comparison figures, and a dedicated interpretation README.

Which files to use
------------------
Use the following entry points and configuration files:

| Goal | Command | Required inputs | Configuration | Main outputs |
|---|---|---|---|---|
| Run unit tests | `py -m unittest discover -s tests -v` | Source code | None | Test report |
| Run the complete synthetic study | `py run_all.py` | None; data is generated | `CConfig.py` and `ExamplePlotConfig.py` | `agent_experiments.db` and `ExperimentalResults/` |
| Run the Chronos/Azure extension | `py run_chronos_trace.py` | Two Azure CSV files in `ChronosTraces/` | Trace parameters in `CConfig.py` | `ExperimentalResults/ChronosTrace/` and `chronos_trace_*` database tables |
| Regenerate only schedule examples | See the command under "Running individual stages" | None | `ExamplePlotConfig.py` | Example PNG/PDF files |

Recommended execution order:

1. Install dependencies.
2. Run the tests.
3. Run `py run_all.py`.
4. Download the Chronos trace files.
5. Run `py run_chronos_trace.py`.

This order matters because `run_all.py` replaces `agent_experiments.db`.
Running it after the Chronos extension removes the `chronos_trace_*` tables
until `run_chronos_trace.py` is run again.

Configuration parameters
------------------------
All experiment metadata is in `CConfig.py`. Parameters are ordinary Python
constants and should be edited before running the relevant entry point.

### Synthetic experiment parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `RANDOM_SEED` | `20261012` | Seed for deterministic synthetic generation |
| `TIME_SCALE` | `10` | Integer simulation ticks per second; default resolution is 0.1 seconds |
| `SYSTEMS_PER_AGENT_COUNT` | `{3:100, 4:100, 5:100, 6:100, 7:80, 8:80}` | Number of generated systems at each system size |
| `TOTAL_AGENTS` | Computed as `3000` | Validation total derived from `SYSTEMS_PER_AGENT_COUNT` |
| `PERIOD_SECONDS` | `(10,12,15,20,24,30,40,60)` | Allowed periodic release intervals |
| `LOAD_PROFILES["low"]` | `(0.03,0.07)` | Processing-time fraction of period for low-load agents |
| `LOAD_PROFILES["medium"]` | `(0.07,0.13)` | Processing-time fraction for medium-load agents |
| `LOAD_PROFILES["high"]` | `(0.12,0.22)` | Processing-time fraction for high-load agents |
| `CONTEXT_FRACTION_RANGE` | `(0.10,0.25)` | Random fraction of processing assigned to context assembly |
| `COMMIT_FRACTION_RANGE` | `(0.10,0.25)` | Random fraction of processing assigned to commit |
| `PERIODICITY_WINDOWS` | `5` | Consecutive hyperperiod windows compared for exact repetition |
| `STORE_DETAILED_JOBS` | `True` | Store every generated job in the SQLite `jobs` table |
| `STOP_ON_DEADLINE_MISS` | `True` | Stop a baseline system run on its first missed deadline |

The remaining processing fraction after context and commit is assigned to
decoding/tool execution. Deadlines equal periods. Priorities are unique, and a
larger integer means higher priority.

Changing `SYSTEMS_PER_AGENT_COUNT` changes the population. Keep
`TOTAL_AGENTS` computed from that dictionary rather than hard-coding a separate
value. Adding very large or relatively prime periods can make the exact
hyperperiod simulation expensive because:

\[
L = \operatorname{lcm}(T_1,\ldots,T_n).
\]

### Sensitivity-analysis parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `SENSITIVITY_REPLICATIONS` | `100` | Replications per parameter level |
| `SENSITIVITY_HP_COUNTS` | `1` through `6` | Higher-priority-agent counts |
| `SENSITIVITY_PERIOD_GROUPS` | Five named tuples | Very-frequent through very-infrequent higher-priority periods |
| `SENSITIVITY_OFFSET_FRACTIONS` | `0.0` through `0.9` | Target-agent release offset as a fraction of its period |
| `SENSITIVITY_COMMIT_FRACTIONS` | `0.05` through `0.50` | Fraction of target processing assigned to commit |

`src/sensitivity.py` defines the fixed target workload used by these sweeps:

- target period: 60 seconds;
- target processing time: 4.5 seconds;
- default higher-priority period group: 10, 12, and 15 seconds;
- 100 deterministic replications per configured level.

### Plot parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `PLOT_DPI` | `300` | PNG resolution |
| `PLOT_FORMATS` | `("png","pdf")` | Raster and vector output formats |

### Chronos/Azure trace parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `TRACE_DATA_DIR` | `ChronosTraces/` | Directory containing the two raw Azure files |
| `TRACE_RESULTS_DIR` | `ExperimentalResults/ChronosTrace/` | Trace-analysis output directory |
| `TRACE_CHRONOS_ALPHA_MS_PER_INPUT_TOKEN` | `0.080` | Chronos A100/7B relative prefill cost |
| `TRACE_CHRONOS_BETA_MS_PER_OUTPUT_TOKEN` | `0.250` | Chronos A100/7B relative decode cost |
| `TRACE_CHRONOS_GAMMA_MS` | `7.0` | Chronos constant request overhead used as commit cost |
| `TRACE_TARGET_UTILIZATION` | `0.70` | Week-mean utilization used to normalize the non-batched replay model |
| `TRACE_LOAD_FACTORS` | `(0.50,1.00,1.25,1.50)` | Arrival-rate multipliers; values above 1 compress inter-arrival times |
| `TRACE_TTFT_DEADLINE_MS` | `2000` | Relative replay deadline |
| `TRACE_REPLAY_LIMIT` | `50000` | Number of requests in the contiguous mixed-service replay |
| `TRACE_REPLAY_SEED` | `42` | Seed selecting the deterministic replay window and quantile reservoirs |

The trace contains only timestamps and input/output token counts. The replay
adds the following assumptions:

- conversation requests receive priority 2;
- code-completion requests receive priority 1;
- context cost is based on input tokens and Chronos alpha;
- execution cost is based on output tokens and Chronos beta;
- Chronos gamma is represented as commit cost;
- all costs are normalized because this experiment models one non-batched
  resource rather than Chronos's continuously batched GPU.

These assumptions make the trace suitable for testing abort/restart accounting
and policy trade-offs. They do not turn the nonperiodic trace into a validation
of the LCM feasibility theorem.

Example-figure parameters
-------------------------
`ExamplePlotConfig.py` controls only explanatory figures and does not change
the 3,000-agent population.

### `TWO_AGENT_BASE`

- `periods`: periods for low- and high-priority agents;
- `processing`: processing times for those agents.

`TWO_AGENT_RELEASE_CASES` supplies named offset pairs for each two-agent plot.

### `THREE_AGENT_EXAMPLE`

- `periods`: periods for three agents in increasing-priority order;
- `processing`: their processing times;
- `offsets`: first-release offsets;
- `hyperperiods`: number of repeated hyperperiods shown.

### `AGENT_WORKED_EXAMPLE`

- `periods` and `offsets`: corrected paper example;
- `context`, `execution`, and `commit`: per-agent phase durations;
- `hyperperiods`: number of hyperperiods rendered.

Installation
------------
Use Python 3.11 or newer.

From PowerShell in this directory:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt

Running tests
-------------
    py -m unittest discover -s tests -v

Running the complete study
--------------------------
    py run_all.py

The run is deterministic under RANDOM_SEED in CConfig.py. The script replaces
`agent_experiments.db`, generates the synthetic workload, runs the main and
sensitivity experiments, exports CSV tables, and regenerates the publication
plots in `ExperimentalResults/`.

Inputs:

- `CConfig.py`;
- `ExamplePlotConfig.py`;
- no external dataset.

Use these results:

- `ExperimentalResults/agent_results.csv` for all 3,000 agents;
- `ExperimentalResults/system_results.csv` for the 560 systems;
- `ExperimentalResults/paper_table_main.csv` for the paper's compact main
  table;
- `ExperimentalResults/paper_table_sensitivity.csv` for the compact
  sensitivity table;
- `agent_experiments.db` for complete job-level data.

Running the Chronos/Azure production-trace extension
----------------------------------------------------
Download the official CC-BY Azure LLM Inference Trace 2024 release assets:

    mkdir ChronosTraces
    curl.exe -L -o ChronosTraces\AzureLLMInferenceTrace_code_1week.csv https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv
    curl.exe -L -o ChronosTraces\AzureLLMInferenceTrace_conv_1week.csv https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv

Then run:

    py run_chronos_trace.py

The raw trace directory is git-ignored. Results are written to
ExperimentalResults\ChronosTrace and to four chronos_trace_* tables in
agent_experiments.db.

Required input filenames:

- `ChronosTraces\AzureLLMInferenceTrace_code_1week.csv`;
- `ChronosTraces\AzureLLMInferenceTrace_conv_1week.csv`.

Use these results:

- `trace_characterization.csv` for complete 44.1-million-request statistics;
- `trace_replay_requests.csv` for the exact 50,000 replay inputs;
- `trace_replay_summary.csv` for compact policy/load results;
- `trace_replay_results.csv` for every request outcome;
- `trace_window_load.csv` for realized RPS and utilization;
- `ExperimentalResults\ChronosTrace\README.md` for assumptions and
  interpretation.

Running individual stages
-------------------------
The intended publication workflow is run_all.py. Individual functions can
also be called from Python:

    from src.generator import generate_systems
    from src.database import reset_database
    from src.experiments import run_main_experiments
    from src.sensitivity import run_sensitivity_analysis
    from src.plotting import generate_outputs
    from src.example_plots import generate_example_plots

Regenerate only explanatory schedule figures:

    py -c "from src.example_plots import generate_example_plots; import CConfig; generate_example_plots(CConfig.OUTPUT_DIR)"

Regenerate CSV tables and main plots from the existing database without
rerunning simulations:

    py -c "import CConfig; from src.database import connect; from src.plotting import generate_outputs; c=connect(CConfig.DATABASE_PATH); generate_outputs(c,CConfig.OUTPUT_DIR); c.close()"

Database contents
-----------------
The SQLite database `agent_experiments.db` is the authoritative, normalized
copy of both the generated synthetic workload and its simulation results. It
contains the following tables:

systems
    One row per generated multi-agent system.

agents
    All 3,000 agent definitions, including phase costs, period, offset, and
    priority.

system_results
    Schedulability, periodicity result, job count, abort count, and waste.

agent_results
    Per-agent maxima and totals, retry and waste bounds, violations, and mean
    observed waste fraction.

jobs
    Job-level release, completion, response time, attempts, aborts, phase-level
    waste, and deadline outcome.

sensitivity_runs
    Raw results for every sensitivity replication.

chronos_trace_characterization
    Full-trace statistics for the Azure code-completion and conversation
    services.

chronos_trace_requests
    The 50,000 timestamp-preserving production requests selected for replay.

chronos_trace_replay_results
    Per-request outcomes for every Chronos-trace policy and load factor.

chronos_trace_replay_summary
    Aggregated Chronos-trace policy results.

Shareable data files
--------------------
The generated data and results are deliberately exported to ordinary CSV files
so they can be shared without requiring Python or SQLite.

Synthetic 3,000-agent dataset:

ExperimentalResults/agent_results.csv
    The easiest standalone synthetic-agent file to share. It has one row for
    each of the 3,000 generated agents and includes system/agent identifiers,
    priority, period, release offset, context time, execution time, commit
    time, total processing time, system size, load profile, observed response
    and abort measurements, wasted processing, and analytical bounds. It
    combines generated inputs and experimental outcomes in one table.

ExperimentalResults/system_results.csv
    One row for each of the 560 generated systems. It includes system size,
    load profile, hyperperiod, maximum offset, schedulability, periodicity,
    job count, abort count, and total wasted processing.

ExperimentalResults/sensitivity_results.csv
    All 2,700 raw sensitivity-analysis replications.

ExperimentalResults/paper_table_main.csv
    Compact aggregate table for systems containing 3 through 8 agents.

ExperimentalResults/paper_table_sensitivity.csv
    Compact aggregate sensitivity table.

Complete synthetic data:

agent_experiments.db
    Contains all generated definitions and outcomes, including all 49,108
    synthetic jobs. Job-level synthetic data is stored in the `jobs` table and
    is not duplicated in a standalone CSV because it is substantially more
    detailed than the paper tables. Use this database when every release,
    response time, attempt count, phase-specific waste value, and deadline
    result is required.

Chronos/Azure trace-derived data:

ExperimentalResults/ChronosTrace/trace_characterization.csv
    Complete-trace statistics for all 44,107,694 valid Azure requests.

ExperimentalResults/ChronosTrace/trace_replay_requests.csv
    The exact 50,000-request mixed-service input used by the replay, including
    arrival times, token counts, assigned priorities, mapped phase durations,
    processing time, and deadline.

ExperimentalResults/ChronosTrace/trace_replay_results.csv
    All 600,000 per-request outcomes across policies and load factors
    (approximately 57 MB).

ExperimentalResults/ChronosTrace/trace_replay_summary.csv
    Compact policy/load/service summary and the easiest Chronos result file to
    share.

ExperimentalResults/ChronosTrace/trace_window_load.csv
    Realized request rate, processing time, and utilization of the selected
    replay window.

ExperimentalResults/ChronosTrace/trace_files_manifest.csv
    Official source URLs, licenses, byte sizes, and SHA-256 hashes for the two
    raw Azure files.

ExperimentalResults/ChronosTrace/README.md
    Complete mapping assumptions, methodology, results, limitations, and
    interpretation guidance for the production-trace extension.

The original Azure CSV files total approximately 1.83 GB and are intentionally
excluded from version control. `ChronosTraces/README.md` provides official
download URLs, checksums, licensing information, and download commands.

Generated paper artifacts
-------------------------
ExperimentalResults/paper_table_main.csv
    Compact main results table grouped by systems with 3 through 8 agents.

ExperimentalResults/system_results.csv
    System-level data suitable for supplementary material.

ExperimentalResults/agent_results.csv
    Agent-level bound and waste measurements.

ExperimentalResults/sensitivity_results.csv
    Raw sensitivity data.

ExperimentalResults/paper_table_sensitivity.csv
    Aggregated sensitivity results ready for use as a paper table.

ExperimentalResults/agent_worked_example_trace.csv
    Phase-level intervals and abort/completion outcomes for the corrected
    three-agent worked example.

ExperimentalResults/periodicity_validation.[png|pdf]
    Counts of periodic schedules, deadline misses, and any mismatches.

ExperimentalResults/observed_vs_bounds.[png|pdf]
    Observed retry/waste maxima against their conditional analytical bounds.

ExperimentalResults/waste_fraction_distribution.[png|pdf]
    Distribution of observed wasted-processing fractions by system size.

ExperimentalResults/sensitivity_analysis.[png|pdf]
    Four-panel sensitivity figure.

ExperimentalResults/two_agent_offset_examples.[png|pdf]
    Two-agent processing patterns for three offset configurations.

ExperimentalResults/three_agent_hyperperiod_example.[png|pdf]
    Repeated three-agent processing pattern across hyperperiods.

ExperimentalResults/agent_worked_example.[png|pdf]
    Corrected paper example showing the A1 commit-phase abort at t=10.

Interpretation cautions
-----------------------
The retry and waste bounds are checked only for systems whose complete
five-window trace is schedulable and periodic. For those deterministic
fixed-offset systems, the maximum observed response over the repeating trace
is used as the response bound input. Results from deadline-missing systems are
reported separately and are not presented as completed-job bounds.

The sensitivity experiment reports observed trace outcomes. It does not claim
that a minimum-inter-arrival periodic stream certifies arbitrary sporadic
arrivals. That limitation follows the revised paper.

Customizing example figures
---------------------------
Edit ExamplePlotConfig.py. The periods, processing times, offsets, number of
hyperperiods, and three processing phases are defined directly in that file.
Then rerun:

    py -c "from src.example_plots import generate_example_plots; import CConfig; generate_example_plots(CConfig.OUTPUT_DIR)"
