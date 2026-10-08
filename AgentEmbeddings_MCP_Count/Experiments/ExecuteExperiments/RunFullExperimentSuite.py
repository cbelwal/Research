"""Run the complete 50,000-agent experiment and artifact pipeline."""

import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def run(script, *arguments):
    command = [
        sys.executable,
        os.path.join(PROJECT_ROOT, script),
        *arguments,
    ]
    print(f"\nRunning: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    dimensions = ("8", "24")
    run("Experiments/DataGeneration/PlotSyntheticData.py")
    run(
        "Experiments/ExecuteExperiments/RunExperiments_Algorithms.py",
        "--dimensions",
        *dimensions,
        "--algorithms",
        "2",
        "3",
    )
    run(
        "Experiments/ExecuteExperiments/RunExperiments_Baselines.py",
        "--dimensions",
        *dimensions,
    )
    run(
        "Experiments/ExecuteExperiments/PlotExperimentalData.py",
        "--dimensions",
        *dimensions,
        "--algorithms",
        "2",
        "3",
    )
    run(
        "Experiments/ExecuteExperiments/PlotAlgorithmComparison.py",
        "--dimensions",
        *dimensions,
    )
    run(
        "Experiments/ExecuteExperiments/GenerateAppendixTables.py",
        "--dimensions",
        *dimensions,
    )


if __name__ == "__main__":
    main()
