from __future__ import annotations

import CConfig

from src.chronos_trace import analyze_chronos_trace


def main() -> None:
    code_path = CConfig.TRACE_DATA_DIR / "AzureLLMInferenceTrace_code_1week.csv"
    conversation_path = (
        CConfig.TRACE_DATA_DIR / "AzureLLMInferenceTrace_conv_1week.csv"
    )
    missing = [
        str(path)
        for path in (code_path, conversation_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Download the official trace files before analysis: "
            + ", ".join(missing)
        )
    analyze_chronos_trace(
        code_path,
        conversation_path,
        CConfig.TRACE_RESULTS_DIR,
        CConfig.DATABASE_PATH,
    )
    print(f"Chronos trace results: {CConfig.TRACE_RESULTS_DIR}")


if __name__ == "__main__":
    main()

