# Experimental results for abort-and-restart multi-agent orchestration

## Purpose of this directory

This directory is a self-contained handoff of the experimental study supporting
`Reference/agent_feasibility.pdf`. It is intended both for human review and as
input to another language model editing the paper.

The study evaluates three separate questions:

1. **Feasibility-interval periodicity:** For schedulable periodic systems, does
   the complete execution pattern repeat across consecutive hyperperiods after
   the maximum initial release offset?
2. **Conditional retry and waste bounds:** Given a valid response-time bound
   obtained from the exact repeating abort-and-restart trace, do observed aborts
   and wasted processing remain within the bounds from Lemma 4 and Corollary 2?
3. **Sensitivity:** How do schedulability, aborts, and wasted processing change
   with the number and frequency of higher-priority agents, relative offsets,
   and the division of work among context, execution, and commit phases?

This is a simulation study. It does not contain production LLM traces or
measurements from a deployed serving system.

## Execution model implemented

The simulator follows the revised paper's model exactly:

- One resource executes at most one agent attempt at a time.
- Agents are periodic and have unique fixed priorities.
- Every job has three processing phases:
  `context`, `execution`, and `commit`.
- All three phases are preemptible.
- A release from a strictly higher-priority agent immediately aborts the
  running lower-priority attempt.
- An aborted job remains ready but restarts from the beginning of context
  assembly when dispatched again.
- No processing progress survives an abort.
- All consumed processing on an aborted attempt is waste, including partial
  commit processing.
- Externally visible effects are assumed to be staged and published atomically
  only after successful completion.
- At each timestamp, event ordering is:
  1. finalize successful completion;
  2. check deadlines;
  3. batch all releases;
  4. dispatch the highest-priority ready job.
- An attempt occupies a half-open interval `[start, finish)`. Therefore, a
  release exactly at its completion timestamp does not abort it.
- Relative deadline equals period.
- The baseline simulation stops on the first deadline miss. It does not replace,
  merge, or silently discard an unfinished job.

All stored times are integer **ticks**. The default resolution is:

```text
TIME_SCALE = 10 ticks per second
```

Thus, divide a `*_ticks` column by 10 to obtain seconds.

## Experiment population

The main experiment contains exactly **3,000 agent definitions** distributed
over **560 independently generated systems**:

| Agents per system | Systems | Agent records | Schedulable systems |
|---:|---:|---:|---:|
| 3 | 100 | 300 | 92 |
| 4 | 100 | 400 | 77 |
| 5 | 100 | 500 | 62 |
| 6 | 100 | 600 | 45 |
| 7 | 80 | 560 | 32 |
| 8 | 80 | 640 | 27 |
| **Total** | **560** | **3,000** | **335** |

The random seed is fixed in `CConfig.py`, making the study reproducible.

### Generated parameters

Periods are selected from:

```text
10, 12, 15, 20, 24, 30, 40, and 60 seconds
```

The finite period set keeps exact least-common-multiple hyperperiod simulation
tractable. Each system belongs to a low, medium, or high nominal-load profile.
Processing time is generated as a profile-dependent fraction of period and is
then divided into:

- context assembly;
- decoding/tool execution;
- commit processing.

Each agent receives a release offset sampled in `[0, period)`. Priorities are
unique within a system, with a larger number representing higher priority.

## Main experiment procedure

For each generated system:

1. Compute:

   \[
   L = \operatorname{lcm}(T_1,\ldots,T_n)
   \]

   and:

   \[
   R_{\max} = \max_i \phi_i.
   \]

2. Release jobs through five complete hyperperiod windows beginning at
   \(R_{\max}\).
3. Allow additional time after the last release window so boundary jobs can
   complete or miss their deadlines.
4. Stop immediately if a job misses its deadline.
5. For a schedulable system, construct a hash for each of the five windows.
   The hash includes:
   - the executing agent at every tick;
   - the active phase at every tick;
   - normalized attempt starts;
   - abort events;
   - completion events.
6. Mark the schedule periodic only if all five complete hashes are identical.
7. For every agent in a periodic schedulable system, use the maximum observed
   response time in the repeating trace as a valid fixed-offset response-time
   bound.
8. Evaluate:

   \[
   N_k^{bound}
   =
   \sum_{j>k}
   \left\lceil \frac{R_k}{T_j} \right\rceil
   \]

   and:

   \[
   W_k^{bound}=N_k^{bound}P_k.
   \]

9. Compare every completed job's observed abort count and wasted processing
   against these bounds.

The bound evaluation is deliberately not applied as a completion proof for a
deadline-missing system. This follows the paper: Lemma 4 is conditional on an
independently valid abort-aware response-time bound.

## Main numerical results

Across the full study:

| Metric | Result |
|---|---:|
| Systems | 560 |
| Agent definitions | 3,000 |
| Jobs released before stopping or completing the study | 49,108 |
| Schedulable systems | 335 |
| Systems stopped on a deadline miss | 225 |
| Observed aborts | 12,693 |
| Observed wasted processing | 168,095 ticks |
| Periodicity mismatches among schedulable systems | **0** |
| Retry-bound violations | **0** |
| Wasted-processing-bound violations | **0** |
| Agents with nonzero commit-phase waste | 293 |

These results empirically support the following limited claims for the sampled
periodic fixed-offset systems:

- every schedulable trace repeated over the five checked hyperperiods;
- observed abort counts respected the conditional retry-count bound;
- observed wasted processing, including partial commit, respected the
  conditional full-processing-time waste bound;
- commit-phase waste is not merely theoretical: it occurred for 293 agents.

The study does **not** prove the theorem for all possible periods, processing
times, or continuous release offsets. It is empirical validation over the
declared discrete parameter space.

## Sensitivity analysis

The sensitivity study contains **2,700 runs**. Each parameter point has 100
seeded replications. The target low-priority agent has a 60-second period and a
4.5-second total processing requirement unless the swept parameter modifies its
phase split.

### Number of higher-priority agents

| Higher-priority agents | Schedulable runs | Mean aborts | Mean observed waste |
|---:|---:|---:|---:|
| 1 | 100% | 0.46 | 13.9% |
| 2 | 100% | 0.93 | 23.3% |
| 3 | 100% | 1.86 | 32.8% |
| 4 | 88% | 5.20 | 52.9% |
| 5 | 36% | 6.19 | 56.2% |
| 6 | 11% | 7.18 | 56.7% |

This is the strongest sensitivity result: adding higher-priority agents sharply
increases discarded work and eventually causes frequent deadline failure.

### Frequency of higher-priority agents

| Period group | Mean period | Schedulable runs | Mean aborts | Mean observed waste |
|---|---:|---:|---:|---:|
| Very frequent | 8.0 s | 74% | 9.50 | 76.1% |
| Frequent | 10.0 s | 100% | 5.00 | 63.1% |
| Baseline | 12.33 s | 100% | 2.19 | 39.4% |
| Infrequent | 15.67 s | 100% | 1.27 | 26.1% |
| Very infrequent | 21.67 s | 100% | 0.90 | 21.4% |

More frequent high-priority releases lead to more aborts and substantially
higher wasted processing. The very-frequent condition also produces deadline
misses in 26% of runs.

### Relative release offset

All sampled offset conditions were schedulable, but mean waste ranged from
29.2% to 39.4%. The maximum mean value occurred at an offset fraction of 0.3.
This demonstrates that phase alignment affects waste even when schedulability
does not change.

### Commit fraction

All commit-fraction conditions were schedulable. Mean total waste remained
between 32.8% and 39.1%, but the composition of waste shifted toward commit as
the commit fraction grew:

- at commit fraction 0.05, mean commit waste was 0.04 ticks;
- at commit fraction 0.50, mean commit waste was 8.55 ticks.

Total waste is not expected to increase monotonically with commit fraction
because changing phase boundaries also changes where releases intersect an
attempt. The important result is that commit-phase work is observably discarded
and must be included in the bound.

## Files in this directory

### Paper-ready tables

#### `paper_table_main.csv`

Compact main-study table grouped by system size. Recommended as the primary
experimental table in the paper.

Columns:

- `agent_count`: agents per system;
- `systems`: number of generated systems;
- `agents`: total agent records;
- `schedulable_systems`: systems completing without a deadline miss;
- `periodicity_failures`: schedulable systems whose five hashes differed;
- `jobs`: jobs created during simulation;
- `observed_aborts`: total aborted attempts;
- `wasted_ticks`: total processing consumed by aborted attempts;
- `retry_bound_violations`: completed jobs exceeding the retry bound;
- `waste_bound_violations`: completed jobs exceeding the waste bound.

#### `paper_table_sensitivity.csv`

Aggregated sensitivity results, one row per parameter value.

Columns:

- `sweep`: sensitivity dimension;
- `parameter_label`: human-readable level;
- `parameter_value`: numeric level used for plotting;
- `replications`: number of runs;
- `schedulable_fraction`: fraction completing without a deadline miss;
- `mean_response_ticks`: mean target-agent response time among completed runs;
- `mean_aborts`: mean target-agent abort count;
- `mean_waste_ticks`: mean target-agent wasted processing;
- `mean_waste_fraction`: mean
  \(W^{obs}/(W^{obs}+P)\);
- phase-specific mean waste columns.

### Detailed CSV data

#### `system_results.csv`

One row per generated system. It combines workload metadata and aggregate
outcomes.

Important columns:

- `system_id`;
- `agent_count`;
- `load_profile`;
- `hyperperiod_ticks`;
- `rmax_ticks`;
- `schedulable`;
- `periodicity_match`;
- `total_jobs`;
- `total_aborts`;
- `total_waste_ticks`.

For deadline-missing systems, `periodicity_match` is empty because a complete
repeating schedulable trace was not established.

#### `agent_results.csv`

One row per one of the 3,000 agents.

It includes:

- complete agent configuration;
- number of completed and missed jobs;
- maximum observed response time;
- maximum observed abort count;
- conditional retry bound and violations;
- total and maximum observed waste;
- conditional waste bound and violations;
- mean observed waste fraction;
- total commit-phase waste.

An empty bound field means the required valid periodic response-time input was
not established, usually because the system missed a deadline.

#### `sensitivity_results.csv`

Raw row-level sensitivity data for all 2,700 replications. Use this file for
alternative statistical analysis or new plots.

#### `agent_worked_example_trace.csv`

Phase-level trace for the corrected three-agent worked example in the paper.
It shows:

- A3 completing on `[0, 1.3)`;
- A2 completing on `[1.3, 4.1)`;
- A1 running context and execution from `6.0` to `9.5`;
- A1 consuming commit processing on `[9.5, 10.0)`;
- A1 being aborted at `10.0`;
- A1 restarting at `11.3` and completing at `15.8`.

This file is suitable for reproducing the worked-example table in the paper.

### Main-result figures

Every figure is supplied as:

- **PDF** for vector inclusion in LaTeX or publication workflows;
- **PNG** for rapid inspection and document tools that do not ingest PDF.

#### `periodicity_validation.[pdf|png]`

Stacked/grouped counts by system size:

- schedulable systems with matching hyperperiod traces;
- systems that encountered a deadline miss;
- any periodicity mismatch.

The generated study has zero mismatches.

#### `observed_vs_bounds.[pdf|png]`

Two panels:

1. maximum observed aborts versus retry-count bound;
2. maximum observed wasted processing versus waste bound.

The dashed diagonal is equality. Valid observations should lie on or below it.
No violations were observed.

#### `waste_fraction_distribution.[pdf|png]`

Boxplots of each agent's mean observed waste fraction, grouped by system size.
Use it to show how discarded processing changes as orchestration complexity
grows.

#### `sensitivity_analysis.[pdf|png]`

Four panels covering:

1. number of higher-priority agents;
2. mean higher-priority period;
3. relative release offset;
4. target-agent commit fraction.

Each panel overlays observed waste percentage and the percentage of runs that
remain schedulable.

### Explanatory schedule figures

#### `two_agent_offset_examples.[pdf|png]`

Three two-agent schedules analogous to the offset cases in
`FeasibilityInterval.pdf`:

- low-priority release at 0, high-priority release at 1;
- low-priority release at 2, high-priority release at 1;
- synchronous release.

The dashed vertical line separates consecutive hyperperiods.

#### `three_agent_hyperperiod_example.[pdf|png]`

Three-agent processing pattern across four hyperperiods, analogous to the
multi-task schedule illustration in the original P-FRP paper.

#### `agent_worked_example.[pdf|png]`

Corrected example from `agent_feasibility.pdf`. Colors distinguish context,
execution, and commit. The red line marks the abort at 10 seconds while A1 is
in commit.

## SQLite database

The full database is one directory above this folder:

```text
../agent_experiments.db
```

It contains:

- `systems`: one row per system;
- `agents`: all 3,000 generated agent definitions;
- `system_results`: system-level outcomes;
- `agent_results`: per-agent bounds and observations;
- `jobs`: all 49,108 job records;
- `sensitivity_runs`: all 2,700 sensitivity runs.

The database is the authoritative source if a new table, figure, or statistical
test is needed.

## Chronos production-trace extension

`ChronosTrace/` contains a separate experiment using the complete 44.1-million
request Azure LLM Inference Trace 2024 cited by Chronos. That directory has its
own detailed `README.md`, full-trace characterization, a timestamp-preserving
50,000-request mixed-service replay, policy comparisons, and raw CSV results.

This extension must not be conflated with the periodic synthetic experiment:

- the Azure arrivals are nonperiodic and therefore do not test the LCM
  feasibility theorem;
- the trace does not provide agent identities, priorities, deadlines, or phase
  execution times, so the replay declares those mapping assumptions explicitly;
- it does test abort/restart accounting and the latency-versus-waste trade-off
  against real production arrival and token-length distributions.

Example query:

```sql
SELECT
    s.agent_count,
    COUNT(*) AS systems,
    SUM(r.schedulable) AS schedulable_systems
FROM systems AS s
JOIN system_results AS r USING (system_id)
GROUP BY s.agent_count
ORDER BY s.agent_count;
```

## Reproduction

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m unittest discover -s tests -v
py run_all.py
```

`run_all.py` replaces `agent_experiments.db` and regenerates this
`ExperimentalResults` directory.

Experiment metadata is in `CConfig.py`. Inputs for explanatory schedule plots
are in `ExamplePlotConfig.py`.

## Recommended use in the paper

The experimental section should distinguish four types of evidence:

1. **Periodicity validation:** zero schedule-signature mismatches over 335
   schedulable sampled systems and five checked hyperperiods each.
2. **Conditional bound validation:** zero retry-count and waste-bound
   violations where a valid periodic abort-aware response-time bound was
   available.
3. **Commit-waste evidence:** 293 agents incurred nonzero discarded commit
   processing.
4. **Sensitivity evidence:** the number and frequency of higher-priority agents
   are the dominant drivers of waste and loss of schedulability; offsets and
   phase composition alter the amount and location of discarded work.

Avoid wording that says the simulation proves global schedulability or proves
the theorem. It validates the implementation and provides empirical evidence
for the sampled parameter space.

## Important limitations

- Fixed periodic arrivals only; arbitrary sporadic arrivals are not certified.
- Discrete time grid of 0.1 seconds.
- Single resource and unique static priorities.
- Fixed bounded processing costs.
- No cancellation/rollback overhead.
- No retained progress after preemption.
- No asynchronous tool work continuing after cancellation.
- No irreversible unstaged side effects.
- No production LLM workload or billing trace.
- The retry and waste bounds are conditional on a valid abort-aware response
  bound; the classical resume-based recurrence is not used for this purpose.
