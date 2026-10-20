from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

import CConfig
import ExamplePlotConfig

from .models import AgentSpec
from .simulator import AbortRestartSimulator
from .utils import lcm_many


PHASE_COLORS = {
    "context": "#4C78A8",
    "execution": "#59A14F",
    "commit": "#F28E2B",
}


def generate_example_plots(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _two_agent_examples(output_dir)
    _three_agent_example(output_dir)
    _worked_agent_example(output_dir)


def _two_agent_examples(output_dir: Path) -> None:
    base = ExamplePlotConfig.TWO_AGENT_BASE
    periods = base["periods"]
    processing = base["processing"]
    scale = CConfig.TIME_SCALE
    hyperperiod = lcm_many(period * scale for period in periods)

    fig, axes = plt.subplots(
        len(ExamplePlotConfig.TWO_AGENT_RELEASE_CASES),
        1,
        figsize=(10, 6.5),
        sharex=True,
    )
    for ax, case in zip(axes, ExamplePlotConfig.TWO_AGENT_RELEASE_CASES):
        agents = [
            AgentSpec(
                agent_id=index + 1,
                priority=index + 1,
                period=periods[index] * scale,
                offset=case["offsets"][index] * scale,
                context_time=0,
                execution_time=processing[index] * scale,
                commit_time=0,
            )
            for index in range(2)
        ]
        start = max(agent.offset for agent in agents)
        end = start + 2 * hyperperiod
        result = AbortRestartSimulator(agents).run(end, end)
        _draw_schedule(ax, result, start, end)
        ax.axvline((start + hyperperiod) / scale, linestyle="--", color="black")
        ax.set_title(case["name"].replace("_", " "))
    axes[-1].set_xlabel("Time (seconds)")
    _save(fig, output_dir, "two_agent_offset_examples")


def _three_agent_example(output_dir: Path) -> None:
    config = ExamplePlotConfig.THREE_AGENT_EXAMPLE
    scale = CConfig.TIME_SCALE
    agents = [
        AgentSpec(
            agent_id=index + 1,
            priority=index + 1,
            period=config["periods"][index] * scale,
            offset=config["offsets"][index] * scale,
            context_time=0,
            execution_time=config["processing"][index] * scale,
            commit_time=0,
        )
        for index in range(3)
    ]
    hyperperiod = lcm_many(agent.period for agent in agents)
    end = config["hyperperiods"] * hyperperiod
    result = AbortRestartSimulator(agents).run(end, end)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    _draw_schedule(ax, result, 0, end)
    for window in range(1, config["hyperperiods"]):
        ax.axvline(window * hyperperiod / scale, linestyle="--", color="black")
    ax.set_title("Three-agent processing pattern across hyperperiods")
    ax.set_xlabel("Time (seconds)")
    _save(fig, output_dir, "three_agent_hyperperiod_example")


def _worked_agent_example(output_dir: Path) -> None:
    config = ExamplePlotConfig.AGENT_WORKED_EXAMPLE
    scale = CConfig.TIME_SCALE
    agents = [
        AgentSpec(
            agent_id=index + 1,
            priority=index + 1,
            period=round(config["periods"][index] * scale),
            offset=round(config["offsets"][index] * scale),
            context_time=round(config["context"][index] * scale),
            execution_time=round(config["execution"][index] * scale),
            commit_time=round(config["commit"][index] * scale),
        )
        for index in range(3)
    ]
    end = max(config["periods"]) * config["hyperperiods"] * scale
    result = AbortRestartSimulator(agents).run(end, end)
    events_by_time = {}
    for event in result.events:
        if event.event_type in {"abort", "complete"}:
            events_by_time.setdefault(event.time, []).append(event.event_type)
    with (output_dir / "agent_worked_example_trace.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_seconds",
                "end_seconds",
                "agent",
                "job_id",
                "attempt",
                "phase",
                "outcome_at_end",
            ]
        )
        for segment in result.segments:
            writer.writerow(
                [
                    segment.start / scale,
                    segment.end / scale,
                    f"A{segment.agent_id}",
                    segment.job_id,
                    segment.attempt_number,
                    segment.phase,
                    "|".join(events_by_time.get(segment.end, [])),
                ]
            )
    fig, ax = plt.subplots(figsize=(10, 3.4))
    _draw_schedule(ax, result, 0, end, show_phase=True)
    for event in result.events:
        if event.event_type == "abort":
            ax.axvline(event.time / scale, color="red", linewidth=1.2)
            ax.text(
                event.time / scale,
                3.35,
                "abort",
                color="red",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_title("Corrected three-agent trace with commit-phase abort")
    ax.set_xlabel("Time (seconds)")
    _save(fig, output_dir, "agent_worked_example")


def _draw_schedule(
    ax,
    result,
    start: int,
    end: int,
    show_phase: bool = False,
) -> None:
    scale = CConfig.TIME_SCALE
    agents = sorted({segment.agent_id for segment in result.segments})
    for segment in result.segments:
        left = max(segment.start, start)
        right = min(segment.end, end)
        if right <= left:
            continue
        color = (
            PHASE_COLORS[segment.phase]
            if show_phase
            else plt.cm.Set2((segment.agent_id - 1) % 8)
        )
        ax.broken_barh(
            [((left / scale), (right - left) / scale)],
            (segment.agent_id - 0.38, 0.76),
            facecolors=color,
            edgecolors="black",
            linewidth=0.25,
        )
    ax.set_xlim(start / scale, end / scale)
    ax.set_yticks(agents)
    ax.set_yticklabels([f"A{agent}" for agent in agents])
    ax.set_ylabel("Agent")
    ax.grid(axis="x", alpha=0.25)


def _save(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for extension in CConfig.PLOT_FORMATS:
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=CConfig.PLOT_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)
