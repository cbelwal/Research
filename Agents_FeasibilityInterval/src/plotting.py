from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import CConfig


def generate_outputs(connection: sqlite3.Connection, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    systems = pd.read_sql_query(
        """
        SELECT s.*, r.schedulable, r.periodicity_match, r.total_jobs,
               r.total_aborts, r.total_waste_ticks
        FROM systems s JOIN system_results r USING (system_id)
        """,
        connection,
    )
    agents = pd.read_sql_query(
        """
        SELECT a.*, s.agent_count, s.load_profile,
               r.completed_jobs, r.missed_jobs, r.max_response_ticks,
               r.max_observed_aborts, r.retry_bound, r.retry_bound_violations,
               r.total_waste_ticks, r.max_waste_ticks, r.waste_bound_ticks,
               r.waste_bound_violations, r.mean_waste_fraction,
               r.commit_waste_ticks
        FROM agents a
        JOIN systems s USING (system_id)
        JOIN agent_results r USING (system_id, agent_id)
        """,
        connection,
    )
    sensitivity = pd.read_sql_query("SELECT * FROM sensitivity_runs", connection)

    systems.to_csv(output_dir / "system_results.csv", index=False)
    agents.to_csv(output_dir / "agent_results.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_results.csv", index=False)
    sensitivity_summary = (
        sensitivity.groupby(
            ["sweep", "parameter_label", "parameter_value"],
            as_index=False,
        )
        .agg(
            replications=("replication", "count"),
            schedulable_fraction=("schedulable", "mean"),
            mean_response_ticks=("response_ticks", "mean"),
            mean_aborts=("aborts", "mean"),
            mean_waste_ticks=("waste_ticks", "mean"),
            mean_waste_fraction=("waste_fraction", "mean"),
            mean_context_waste_ticks=("context_waste_ticks", "mean"),
            mean_execution_waste_ticks=("execution_waste_ticks", "mean"),
            mean_commit_waste_ticks=("commit_waste_ticks", "mean"),
        )
        .sort_values(["sweep", "parameter_value"])
    )
    sensitivity_summary.to_csv(
        output_dir / "paper_table_sensitivity.csv",
        index=False,
    )

    paper_table = (
        systems.groupby("agent_count")
        .agg(
            systems=("system_id", "count"),
            agents=("agent_count", "sum"),
            schedulable_systems=("schedulable", "sum"),
            periodicity_failures=(
                "periodicity_match",
                lambda values: int((values.dropna() == 0).sum()),
            ),
            jobs=("total_jobs", "sum"),
            observed_aborts=("total_aborts", "sum"),
            wasted_ticks=("total_waste_ticks", "sum"),
        )
        .reset_index()
    )
    violations = agents.groupby("agent_count", as_index=False).agg(
        retry_bound_violations=("retry_bound_violations", "sum"),
        waste_bound_violations=("waste_bound_violations", "sum"),
    )
    paper_table = paper_table.merge(violations, on="agent_count", how="left")
    paper_table.to_csv(output_dir / "paper_table_main.csv", index=False)

    _plot_periodicity(systems, output_dir)
    _plot_bounds(agents, output_dir)
    _plot_waste_distribution(agents, output_dir)
    _plot_sensitivity(sensitivity, output_dir)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for extension in CConfig.PLOT_FORMATS:
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=CConfig.PLOT_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_periodicity(systems: pd.DataFrame, output_dir: Path) -> None:
    counts = (
        systems.assign(
            outcome=systems.apply(
                lambda row: (
                    "Deadline miss"
                    if row["schedulable"] == 0
                    else (
                        "Periodic"
                        if row["periodicity_match"] == 1
                        else "Mismatch"
                    )
                ),
                axis=1,
            )
        )
        .groupby(["agent_count", "outcome"])
        .size()
        .reset_index(name="systems")
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.barplot(data=counts, x="agent_count", y="systems", hue="outcome", ax=ax)
    ax.set(xlabel="Agents per system", ylabel="Number of systems")
    ax.legend(title="")
    _save(fig, output_dir, "periodicity_validation")


def _plot_bounds(agents: pd.DataFrame, output_dir: Path) -> None:
    valid = agents.dropna(subset=["retry_bound", "waste_bound_ticks"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    sns.scatterplot(
        data=valid,
        x="retry_bound",
        y="max_observed_aborts",
        hue="agent_count",
        alpha=0.65,
        ax=axes[0],
        palette="viridis",
        legend=False,
    )
    max_retry = max(valid["retry_bound"].max(), 1)
    axes[0].plot([0, max_retry], [0, max_retry], "--", color="black", linewidth=1)
    axes[0].set(xlabel="Retry-count bound", ylabel="Maximum observed aborts")

    sns.scatterplot(
        data=valid,
        x="waste_bound_ticks",
        y="max_waste_ticks",
        hue="agent_count",
        alpha=0.65,
        ax=axes[1],
        palette="viridis",
    )
    max_waste = max(valid["waste_bound_ticks"].max(), 1)
    axes[1].plot([0, max_waste], [0, max_waste], "--", color="black", linewidth=1)
    axes[1].set(xlabel="Wasted-processing bound (ticks)", ylabel="Maximum observed waste (ticks)")
    axes[1].legend(title="Agents")
    _save(fig, output_dir, "observed_vs_bounds")


def _plot_waste_distribution(agents: pd.DataFrame, output_dir: Path) -> None:
    valid = agents.dropna(subset=["mean_waste_fraction"]).copy()
    valid["mean_waste_percent"] = 100 * valid["mean_waste_fraction"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.boxplot(
        data=valid,
        x="agent_count",
        y="mean_waste_percent",
        hue="agent_count",
        legend=False,
        ax=ax,
        palette="viridis",
    )
    ax.set(xlabel="Agents per system", ylabel="Mean observed waste fraction (%)")
    _save(fig, output_dir, "waste_fraction_distribution")


def _plot_sensitivity(data: pd.DataFrame, output_dir: Path) -> None:
    data = data.copy()
    data["waste_percent"] = 100 * data["waste_fraction"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    sweeps = (
        ("higher_priority_count", "Higher-priority agents", axes[0, 0]),
        ("higher_priority_period", "Mean higher-priority period (s)", axes[0, 1]),
        ("relative_offset", "Target release offset / period", axes[1, 0]),
        ("commit_fraction", "Commit fraction of target processing", axes[1, 1]),
    )
    for sweep, xlabel, ax in sweeps:
        subset = data[data["sweep"] == sweep]
        summary = (
            subset.groupby(["parameter_value", "parameter_label"], as_index=False)
            .agg(
                waste_percent=("waste_percent", "mean"),
                aborts=("aborts", "mean"),
                schedulable=("schedulable", "mean"),
            )
            .sort_values("parameter_value")
        )
        ax.plot(
            summary["parameter_value"],
            summary["waste_percent"],
            marker="o",
            label="Waste (%)",
        )
        secondary = ax.twinx()
        secondary.plot(
            summary["parameter_value"],
            100 * summary["schedulable"],
            marker="s",
            color="tab:orange",
            label="Schedulable (%)",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Observed waste fraction (%)")
        secondary.set_ylabel("Schedulable runs (%)")
        ax.set_ylim(bottom=0)
        secondary.set_ylim(0, 105)
    _save(fig, output_dir, "sensitivity_analysis")
