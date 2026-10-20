from __future__ import annotations

import csv
import heapq
import json
import math
import random
import sqlite3
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import CConfig


@dataclass(frozen=True)
class TraceRequest:
    request_id: int
    service: str
    arrival_us: int
    input_tokens: int
    output_tokens: int
    priority: int
    context_us: int
    execution_us: int
    commit_us: int
    deadline_us: int

    @property
    def processing_us(self) -> int:
        return self.context_us + self.execution_us + self.commit_us


@dataclass
class ReplayJob:
    request: TraceRequest
    progress_us: int = 0
    attempts: int = 0
    preemptions: int = 0
    aborts: int = 0
    wasted_us: int = 0
    completion_us: int | None = None


@dataclass
class ReplayResult:
    policy: str
    jobs: list[ReplayJob] = field(default_factory=list)


def parse_timestamp_us(value: str) -> int:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return int(round(parsed.timestamp() * 1_000_000))


def iter_trace_rows(path: Path, service: str) -> Iterator[tuple[int, int, int, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        timestamp_column = _find_column(normalized, ("timestamp",))
        context_column = _find_column(normalized, ("contexttokens", "inputtokens"))
        generated_column = _find_column(
            normalized,
            ("generatedtokens", "outputtokens"),
        )
        for row in reader:
            try:
                input_tokens = int(float(row[context_column]))
                output_tokens = int(float(row[generated_column]))
                if input_tokens <= 0 or output_tokens <= 0:
                    continue
                yield (
                    parse_timestamp_us(row[timestamp_column]),
                    input_tokens,
                    output_tokens,
                    service,
                )
            except (KeyError, TypeError, ValueError):
                continue


def characterize_trace(
    path: Path,
    service: str,
    reservoir_size: int = 200_000,
) -> dict:
    rng = random.Random(CConfig.TRACE_REPLAY_SEED + (0 if service == "code" else 1))
    sample_inputs: list[int] = []
    sample_outputs: list[int] = []
    count = 0
    first_timestamp = None
    last_timestamp = None
    previous_timestamp = None
    interarrival_count = 0
    interarrival_mean = 0.0
    interarrival_m2 = 0.0
    input_total = 0
    output_total = 0

    for timestamp, input_tokens, output_tokens, _ in iter_trace_rows(path, service):
        count += 1
        first_timestamp = timestamp if first_timestamp is None else first_timestamp
        last_timestamp = timestamp
        input_total += input_tokens
        output_total += output_tokens
        if previous_timestamp is not None:
            delta = timestamp - previous_timestamp
            interarrival_count += 1
            difference = delta - interarrival_mean
            interarrival_mean += difference / interarrival_count
            interarrival_m2 += difference * (delta - interarrival_mean)
        previous_timestamp = timestamp

        if len(sample_inputs) < reservoir_size:
            sample_inputs.append(input_tokens)
            sample_outputs.append(output_tokens)
        else:
            replacement = rng.randrange(count)
            if replacement < reservoir_size:
                sample_inputs[replacement] = input_tokens
                sample_outputs[replacement] = output_tokens

    if not count or first_timestamp is None or last_timestamp is None:
        raise ValueError(f"No valid rows found in {path}.")
    duration_seconds = max((last_timestamp - first_timestamp) / 1_000_000, 0.0)
    interarrival_variance = (
        interarrival_m2 / (interarrival_count - 1)
        if interarrival_count > 1
        else 0.0
    )
    interarrival_cv = (
        math.sqrt(interarrival_variance) / interarrival_mean
        if interarrival_mean > 0
        else 0.0
    )
    input_series = pd.Series(sample_inputs)
    output_series = pd.Series(sample_outputs)
    return {
        "service": service,
        "rows": count,
        "first_timestamp_us": first_timestamp,
        "last_timestamp_us": last_timestamp,
        "duration_seconds": duration_seconds,
        "mean_rps": count / duration_seconds if duration_seconds > 0 else None,
        "mean_input_tokens": input_total / count,
        "mean_output_tokens": output_total / count,
        "median_input_tokens_approx": float(input_series.median()),
        "p99_input_tokens_approx": float(input_series.quantile(0.99)),
        "median_output_tokens_approx": float(output_series.median()),
        "p99_output_tokens_approx": float(output_series.quantile(0.99)),
        "interarrival_cv": interarrival_cv,
        "quantile_reservoir_size": len(sample_inputs),
    }


def load_contiguous_combined_window(
    code_path: Path,
    conversation_path: Path,
    service_stats: list[dict],
    limit: int,
    timing_scale: float,
) -> list[TraceRequest]:
    shared_start = max(item["first_timestamp_us"] for item in service_stats)
    shared_end = min(item["last_timestamp_us"] for item in service_stats)
    rng = random.Random(CConfig.TRACE_REPLAY_SEED)
    window_start = shared_start + int((shared_end - shared_start) * rng.uniform(0.1, 0.8))

    iterators = {
        "code": iter_trace_rows(code_path, "code"),
        "conversation": iter_trace_rows(conversation_path, "conversation"),
    }
    current = {
        service: _advance_to(iterator, window_start)
        for service, iterator in iterators.items()
    }
    requests: list[TraceRequest] = []
    base_timestamp = None

    while len(requests) < limit and any(value is not None for value in current.values()):
        service = min(
            (name for name, value in current.items() if value is not None),
            key=lambda name: current[name][0],
        )
        timestamp, input_tokens, output_tokens, _ = current[service]
        if base_timestamp is None:
            base_timestamp = timestamp
        arrival_us = timestamp - base_timestamp
        requests.append(
            _map_request(
                len(requests) + 1,
                service,
                arrival_us,
                input_tokens,
                output_tokens,
                timing_scale,
            )
        )
        current[service] = next(iterators[service], None)
    return requests


def replay_requests(requests: list[TraceRequest], policy: str) -> ReplayResult:
    if policy not in {"abort_restart", "resume", "nonpreemptive"}:
        raise ValueError(f"Unsupported policy: {policy}")
    jobs = [ReplayJob(request=request) for request in requests]
    ready: list[tuple[int, int, ReplayJob]] = []
    running: ReplayJob | None = None
    current_time = requests[0].arrival_us if requests else 0
    index = 0
    sequence = 0

    def dispatch() -> ReplayJob | None:
        nonlocal sequence
        if not ready:
            return None
        _, _, selected = heapq.heappop(ready)
        selected.attempts += 1
        sequence += 1
        return selected

    while index < len(jobs) or running is not None or ready:
        next_release = (
            jobs[index].request.arrival_us if index < len(jobs) else math.inf
        )
        if running is None:
            if not ready and next_release < math.inf:
                current_time = max(current_time, next_release)
                while (
                    index < len(jobs)
                    and jobs[index].request.arrival_us == current_time
                ):
                    job = jobs[index]
                    heapq.heappush(
                        ready,
                        (-job.request.priority, job.request.request_id, job),
                    )
                    index += 1
            running = dispatch()
            continue

        remaining = running.request.processing_us - running.progress_us
        completion_time = current_time + remaining
        if completion_time <= next_release:
            current_time = completion_time
            running.progress_us = running.request.processing_us
            running.completion_us = current_time
            running = dispatch()
            continue

        consumed = next_release - current_time
        running.progress_us += consumed
        current_time = next_release
        released: list[ReplayJob] = []
        while index < len(jobs) and jobs[index].request.arrival_us == current_time:
            released.append(jobs[index])
            index += 1
        for job in released:
            heapq.heappush(
                ready,
                (-job.request.priority, job.request.request_id, job),
            )

        highest_release_priority = max(job.request.priority for job in released)
        if (
            policy != "nonpreemptive"
            and highest_release_priority > running.request.priority
        ):
            running.preemptions += 1
            if policy == "abort_restart":
                running.aborts += 1
                running.wasted_us += running.progress_us
                running.progress_us = 0
            heapq.heappush(
                ready,
                (
                    -running.request.priority,
                    running.request.request_id,
                    running,
                ),
            )
            running = dispatch()

    return ReplayResult(policy=policy, jobs=jobs)


def analyze_chronos_trace(
    code_path: Path,
    conversation_path: Path,
    output_dir: Path,
    database_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    characterization_path = output_dir / "trace_characterization.json"
    if characterization_path.exists():
        print("Reusing complete-trace characterization...")
        service_stats = json.loads(characterization_path.read_text(encoding="utf-8"))
    else:
        print("Characterizing complete code trace...")
        code_stats = characterize_trace(code_path, "code")
        print("Characterizing complete conversation trace...")
        conversation_stats = characterize_trace(conversation_path, "conversation")
        service_stats = [code_stats, conversation_stats]
        characterization_path.write_text(
            json.dumps(service_stats, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(service_stats).to_csv(
            output_dir / "trace_characterization.csv",
            index=False,
        )

    timing_scale = _timing_normalization_scale(service_stats)
    request_path = output_dir / "trace_replay_requests.csv"
    if request_path.exists():
        print(f"Reusing contiguous {CConfig.TRACE_REPLAY_LIMIT:,}-request window...")
        requests = _requests_from_frame(pd.read_csv(request_path), timing_scale)
    else:
        print(f"Loading contiguous {CConfig.TRACE_REPLAY_LIMIT:,}-request window...")
        requests = load_contiguous_combined_window(
            code_path,
            conversation_path,
            service_stats,
            CConfig.TRACE_REPLAY_LIMIT,
            timing_scale,
        )
    request_frame = pd.DataFrame(
        [
            {
                "request_id": item.request_id,
                "service": item.service,
                "arrival_us": item.arrival_us,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "priority": item.priority,
                "context_us": item.context_us,
                "execution_us": item.execution_us,
                "commit_us": item.commit_us,
                "processing_us": item.processing_us,
                "deadline_us": item.deadline_us,
            }
            for item in requests
        ]
    )
    request_frame.to_csv(output_dir / "trace_replay_requests.csv", index=False)
    window_stats = _window_load_stats(request_frame)
    window_stats.to_csv(output_dir / "trace_window_load.csv", index=False)

    result_frames = []
    for load_factor in CConfig.TRACE_LOAD_FACTORS:
        load_requests = [
            TraceRequest(
                request_id=item.request_id,
                service=item.service,
                arrival_us=round(item.arrival_us / load_factor),
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                priority=item.priority,
                context_us=item.context_us,
                execution_us=item.execution_us,
                commit_us=item.commit_us,
                deadline_us=item.deadline_us,
            )
            for item in requests
        ]
        for policy in ("abort_restart", "resume", "nonpreemptive"):
            print(f"Replaying policy={policy}, load={load_factor:.2f}x...")
            frame = _result_frame(replay_requests(load_requests, policy))
            frame.insert(1, "load_factor", load_factor)
            result_frames.append(frame)
    details = pd.concat(result_frames, ignore_index=True)
    details.to_csv(output_dir / "trace_replay_results.csv", index=False)
    summary = _summarize_results(details)
    summary.to_csv(output_dir / "trace_replay_summary.csv", index=False)
    _write_database(database_path, service_stats, request_frame, details, summary)
    _plot_results(details, summary, output_dir)
    _write_readme(output_dir, service_stats, window_stats, summary, timing_scale)


def _map_request(
    request_id: int,
    service: str,
    arrival_us: int,
    input_tokens: int,
    output_tokens: int,
    timing_scale: float,
) -> TraceRequest:
    priority = 2 if service == "conversation" else 1
    return TraceRequest(
        request_id=request_id,
        service=service,
        arrival_us=arrival_us,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        priority=priority,
        context_us=max(
            1,
            round(
                input_tokens
                * CConfig.TRACE_CHRONOS_ALPHA_MS_PER_INPUT_TOKEN
                * 1_000
                * timing_scale
            ),
        ),
        execution_us=max(
            1,
            round(
                output_tokens
                * CConfig.TRACE_CHRONOS_BETA_MS_PER_OUTPUT_TOKEN
                * 1_000
                * timing_scale
            ),
        ),
        commit_us=max(
            1,
            round(CConfig.TRACE_CHRONOS_GAMMA_MS * 1_000 * timing_scale),
        ),
        deadline_us=round(CConfig.TRACE_TTFT_DEADLINE_MS * 1_000),
    )


def _result_frame(result: ReplayResult) -> pd.DataFrame:
    high_arrivals = sorted(
        job.request.arrival_us
        for job in result.jobs
        if job.request.priority == 2
    )
    rows = []
    for job in result.jobs:
        completion = job.completion_us
        response = completion - job.request.arrival_us if completion is not None else None
        higher_releases = 0
        if completion is not None and job.request.priority == 1:
            higher_releases = (
                bisect_left(high_arrivals, completion)
                - bisect_left(high_arrivals, job.request.arrival_us)
            )
        rows.append(
            {
                "policy": result.policy,
                "request_id": job.request.request_id,
                "service": job.request.service,
                "priority": job.request.priority,
                "arrival_us": job.request.arrival_us,
                "completion_us": completion,
                "response_us": response,
                "deadline_us": job.request.deadline_us,
                "deadline_miss": int(
                    response is None or response > job.request.deadline_us
                ),
                "processing_us": job.request.processing_us,
                "attempts": job.attempts,
                "preemptions": job.preemptions,
                "aborts": job.aborts,
                "wasted_us": job.wasted_us,
                "waste_fraction": (
                    job.wasted_us / (job.wasted_us + job.request.processing_us)
                    if job.wasted_us + job.request.processing_us > 0
                    else 0.0
                ),
                "higher_priority_releases_in_response": higher_releases,
                "release_count_bound_violation": int(
                    job.aborts > higher_releases
                ),
                "per_attempt_waste_bound_violation": int(
                    job.wasted_us > job.aborts * job.request.processing_us
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_results(details: pd.DataFrame) -> pd.DataFrame:
    return (
        details.groupby(["load_factor", "policy", "service"], as_index=False)
        .agg(
            requests=("request_id", "count"),
            deadline_miss_rate=("deadline_miss", "mean"),
            mean_response_ms=("response_us", lambda values: values.mean() / 1_000),
            p99_response_ms=(
                "response_us",
                lambda values: values.quantile(0.99) / 1_000,
            ),
            total_preemptions=("preemptions", "sum"),
            total_aborts=("aborts", "sum"),
            total_waste_seconds=("wasted_us", lambda values: values.sum() / 1_000_000),
            mean_waste_fraction=("waste_fraction", "mean"),
            release_count_bound_violations=("release_count_bound_violation", "sum"),
            per_attempt_waste_bound_violations=(
                "per_attempt_waste_bound_violation",
                "sum",
            ),
        )
    )


def _plot_results(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plot_data = details.copy()
    plot_data["response_ms"] = plot_data["response_us"] / 1_000
    plot_data["waste_percent"] = 100 * plot_data["waste_fraction"]
    plot_data["policy_service"] = (
        plot_data["policy"] + " / " + plot_data["service"]
    )
    nominal = plot_data[plot_data["load_factor"] == 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    sample = nominal.groupby(["policy", "service"], group_keys=False).sample(
        n=min(5_000, len(nominal) // 6),
        random_state=CConfig.TRACE_REPLAY_SEED,
        replace=False,
    )
    sns.ecdfplot(
        data=sample,
        x="response_ms",
        hue="policy_service",
        ax=axes[0],
    )
    axes[0].set(xlabel="Response time (ms)", ylabel="Empirical CDF")
    sns.barplot(
        data=summary[summary["load_factor"] == 1.0],
        x="policy",
        y="total_waste_seconds",
        hue="service",
        ax=axes[1],
    )
    axes[1].set(xlabel="Policy", ylabel="Total wasted processing (seconds)")
    axes[1].tick_params(axis="x", rotation=20)
    _save_plot(fig, output_dir / "trace_policy_comparison")

    abort_data = plot_data[
        (plot_data["policy"] == "abort_restart")
        & (plot_data["service"] == "code")
        & (plot_data["load_factor"] == 1.0)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    sns.histplot(abort_data, x="aborts", discrete=True, ax=axes[0])
    axes[0].set(xlabel="Aborts per code request", ylabel="Requests")
    sns.histplot(
        abort_data,
        x="waste_percent",
        bins=30,
        ax=axes[1],
    )
    axes[1].set(xlabel="Observed waste fraction (%)", ylabel="Requests")
    _save_plot(fig, output_dir / "trace_abort_waste_distribution")


def _write_database(
    path: Path,
    service_stats: list[dict],
    requests: pd.DataFrame,
    details: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    connection = sqlite3.connect(path)
    try:
        pd.DataFrame(service_stats).to_sql(
            "chronos_trace_characterization",
            connection,
            if_exists="replace",
            index=False,
        )
        requests.to_sql(
            "chronos_trace_requests",
            connection,
            if_exists="replace",
            index=False,
        )
        details.to_sql(
            "chronos_trace_replay_results",
            connection,
            if_exists="replace",
            index=False,
        )
        summary.to_sql(
            "chronos_trace_replay_summary",
            connection,
            if_exists="replace",
            index=False,
        )
    finally:
        connection.close()


def _write_readme(
    output_dir: Path,
    service_stats: list[dict],
    window_stats: pd.DataFrame,
    summary: pd.DataFrame,
    timing_scale: float,
) -> None:
    stats_table = pd.DataFrame(service_stats).to_markdown(index=False)
    window_table = window_stats.to_markdown(index=False)
    summary_table = summary.to_markdown(index=False)
    half_abort_code = _summary_row(summary, 0.5, "abort_restart", "code")
    half_resume_code = _summary_row(summary, 0.5, "resume", "code")
    half_nonpreemptive_code = _summary_row(summary, 0.5, "nonpreemptive", "code")
    half_abort_conversation = _summary_row(
        summary,
        0.5,
        "abort_restart",
        "conversation",
    )
    half_nonpreemptive_conversation = _summary_row(
        summary,
        0.5,
        "nonpreemptive",
        "conversation",
    )
    nominal_abort_code = _summary_row(summary, 1.0, "abort_restart", "code")
    content = f"""# Chronos/Azure trace analysis

This directory analyzes the official Azure LLM Inference Trace 2024 used by
Chronos. The raw CC-BY trace files are stored in the git-ignored `ChronosTraces`
directory and are not copied here.

Sources:

- Chronos paper: https://doi.org/10.3389/fcomp.2026.1873627
- Chronos artifact: https://github.com/am-research/rtss-ttft-tbt
- Azure dataset documentation:
  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
- Dataset paper: Stojkovic et al., "DynamoLLM", HPCA 2025.
- License: CC-BY, as declared by Azure/AzurePublicDataset.

## Full-trace characterization

{stats_table}

Approximate quantiles use a deterministic 200,000-row reservoir per service.
Counts, duration, means, RPS, and inter-arrival CV are computed over every valid
row in each complete trace.

## Replay mapping

The raw trace supplies only timestamp, context-token count, and generated-token
count. It does not supply agent identity, priority, deadline, context/execute/
commit phase timing, or an abort policy. Therefore the replay adds explicit,
configurable assumptions:

- conversation requests: priority 2;
- code-completion requests: priority 1;
- relative costs use Chronos's A100/7B parameters:
  alpha={CConfig.TRACE_CHRONOS_ALPHA_MS_PER_INPUT_TOKEN} ms/input token,
  beta={CConfig.TRACE_CHRONOS_BETA_MS_PER_OUTPUT_TOKEN} ms/output token,
  gamma={CConfig.TRACE_CHRONOS_GAMMA_MS} ms;
- all three costs are multiplied by {timing_scale:.6f} to give the combined
  nominal trace a target utilization of {CConfig.TRACE_TARGET_UTILIZATION:.2f}
  in this non-batched single-resource model;
- relative deadline: {CConfig.TRACE_TTFT_DEADLINE_MS} ms;
- workload: a contiguous, timestamp-preserving combined window of
  {CConfig.TRACE_REPLAY_LIMIT:,} requests selected deterministically.
- load factors: {", ".join(str(value) for value in CConfig.TRACE_LOAD_FACTORS)};
  factors above 1 compress inter-arrival times.

These coefficients are hypotheses for stress-testing the paper's logic, not
measurements provided by Azure or Chronos.

## Realized replay-window load

{window_table}

The normalization uses week-wide mean rates, but the replay deliberately keeps
a contiguous real timestamp window. This selected window is substantially
burstier than the week-wide mean, so its realized utilization can exceed the
nominal target. `load_factor` multiplies the realized arrival rate and
utilization by compressing or expanding inter-arrival times.

## Policy results

{summary_table}

## Key findings

1. The complete files contain {sum(item["rows"] for item in service_stats):,}
   valid requests. The measured counts, mean RPS, medians, and P99 token lengths
   closely reproduce Chronos's trace-characterization table.
2. At load factor 0.5, the selected burst window has approximately
   {window_stats.iloc[-1]["realized_utilization"] * 0.5:.3f} realized
   utilization and every request meets the 2-second deadline under all three
   policies.
3. At that schedulable load, abort/restart causes
   {int(half_abort_code["total_aborts"]):,} code-request aborts and
   {half_abort_code["total_waste_seconds"]:.2f} seconds of discarded
   processing. Mean code response rises to
   {half_abort_code["mean_response_ms"]:.2f} ms, compared with
   {half_resume_code["mean_response_ms"]:.2f} ms under resume and
   {half_nonpreemptive_code["mean_response_ms"]:.2f} ms under non-preemptive
   execution.
4. The same immediate preemption reduces mean conversation response to
   {half_abort_conversation["mean_response_ms"]:.2f} ms versus
   {half_nonpreemptive_conversation["mean_response_ms"]:.2f} ms under
   non-preemptive execution. This is the paper's latency-versus-waste trade-off
   on a real arrival/token trace.
5. At load factor 1.0 the selected burst has realized utilization
   {window_stats.iloc[-1]["realized_utilization"]:.3f}, above capacity.
   Conversation requests remain responsive because they have strict priority,
   while {100 * nominal_abort_code["deadline_miss_rate"]:.2f}% of code requests
   miss under abort/restart. This demonstrates that priority and abort bounds do
   not replace admission control when offered load exceeds capacity.
6. No request violates the trace-specific release-count consistency check or
   the per-attempt waste bound in any replay.

`abort_restart` implements the paper's model. `resume` preempts but preserves
progress. `nonpreemptive` lets a running request finish. The trace can test
observed abort counts, waste, and policy trade-offs under real arrival/token
distributions. It cannot validate the periodic LCM feasibility theorem because
the production arrivals are nonperiodic and requests are not recurring agents.

The release-count check counts actual high-priority arrivals during each
low-priority request's observed response interval. It is a trace-specific
consistency check, not the periodic formula from Lemma 4. The per-attempt waste
check verifies `observed waste <= aborts × processing time`.

## Difference from the Chronos experiment

Chronos randomly sub-samples 50,000 requests from one service (primarily the
conversation trace), models chunked prefill plus continuously batched decode,
and studies TTFT/TBT admission control. This extension instead merges a
contiguous timestamp-preserving window from both services, assigns conversation
strictly higher priority than code, and models whole agent attempts on one
resource. It is designed to stress abort/restart accounting, not to reproduce
Chronos's scheduler results.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _save_plot(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    for extension in CConfig.PLOT_FORMATS:
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=CConfig.PLOT_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


def _advance_to(
    iterator: Iterator[tuple[int, int, int, str]],
    minimum_timestamp: int,
) -> tuple[int, int, int, str] | None:
    for row in iterator:
        if row[0] >= minimum_timestamp:
            return row
    return None


def _find_column(normalized: dict[str, str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for normalized_name, original_name in normalized.items():
        if any(candidate in normalized_name for candidate in candidates):
            return original_name
    raise ValueError(f"Missing required column; available columns: {normalized}")


def _timing_normalization_scale(service_stats: list[dict]) -> float:
    raw_utilization = 0.0
    for item in service_stats:
        mean_processing_ms = (
            item["mean_input_tokens"]
            * CConfig.TRACE_CHRONOS_ALPHA_MS_PER_INPUT_TOKEN
            + item["mean_output_tokens"]
            * CConfig.TRACE_CHRONOS_BETA_MS_PER_OUTPUT_TOKEN
            + CConfig.TRACE_CHRONOS_GAMMA_MS
        )
        raw_utilization += item["mean_rps"] * mean_processing_ms / 1_000
    return CConfig.TRACE_TARGET_UTILIZATION / raw_utilization


def _requests_from_frame(frame: pd.DataFrame, timing_scale: float) -> list[TraceRequest]:
    return [
        _map_request(
            int(row.request_id),
            str(row.service),
            int(row.arrival_us),
            int(row.input_tokens),
            int(row.output_tokens),
            timing_scale,
        )
        for row in frame.itertuples(index=False)
    ]


def _window_load_stats(frame: pd.DataFrame) -> pd.DataFrame:
    duration_seconds = (
        frame["arrival_us"].max() - frame["arrival_us"].min()
    ) / 1_000_000
    rows = []
    for service, group in frame.groupby("service"):
        rows.append(
            {
                "service": service,
                "requests": len(group),
                "duration_seconds": duration_seconds,
                "realized_rps": len(group) / duration_seconds,
                "mean_processing_ms": group["processing_us"].mean() / 1_000,
                "realized_utilization": (
                    group["processing_us"].sum() / 1_000_000 / duration_seconds
                ),
            }
        )
    rows.append(
        {
            "service": "combined",
            "requests": len(frame),
            "duration_seconds": duration_seconds,
            "realized_rps": len(frame) / duration_seconds,
            "mean_processing_ms": frame["processing_us"].mean() / 1_000,
            "realized_utilization": (
                frame["processing_us"].sum() / 1_000_000 / duration_seconds
            ),
        }
    )
    return pd.DataFrame(rows)


def _summary_row(
    summary: pd.DataFrame,
    load_factor: float,
    policy: str,
    service: str,
) -> pd.Series:
    rows = summary[
        (summary["load_factor"] == load_factor)
        & (summary["policy"] == policy)
        & (summary["service"] == service)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one summary row for {load_factor}, {policy}, {service}."
        )
    return rows.iloc[0]
