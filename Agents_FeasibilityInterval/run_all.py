from __future__ import annotations

import CConfig

from src.database import reset_database
from src.example_plots import generate_example_plots
from src.experiments import run_main_experiments
from src.generator import generate_systems
from src.plotting import generate_outputs
from src.sensitivity import run_sensitivity_analysis


def main() -> None:
    CConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    systems = generate_systems()
    generated_agents = sum(system.agent_count for system in systems)
    if generated_agents != CConfig.TOTAL_AGENTS:
        raise RuntimeError(
            f"Configured {CConfig.TOTAL_AGENTS} agents but generated {generated_agents}."
        )

    connection = reset_database(CConfig.DATABASE_PATH)
    try:
        print(f"Running main experiments for {generated_agents} agents...")
        run_main_experiments(connection, systems)
        print("Running sensitivity analysis...")
        run_sensitivity_analysis(connection)
        print("Generating CSV files and publication plots...")
        generate_outputs(connection, CConfig.OUTPUT_DIR)
        generate_example_plots(CConfig.OUTPUT_DIR)
    finally:
        connection.close()
    print(f"Database: {CConfig.DATABASE_PATH}")
    print(f"Experimental results: {CConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
