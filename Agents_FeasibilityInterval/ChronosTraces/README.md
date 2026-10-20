# Chronos trace files

This directory is reserved for the **Azure LLM Inference Trace 2024** used by
the Chronos paper:

> Marref, Tarmissi, and Chaibi, "Formal schedulability analysis for LLM
> inference: TTFT and TBT deadline guarantees via response-time theory,"
> *Frontiers in Computer Science*, 2026.
>
> https://doi.org/10.3389/fcomp.2026.1873627

The raw CSV files are **not uploaded to this repository because of their
size**:

| File | Approximate size |
|---|---:|
| `AzureLLMInferenceTrace_code_1week.csv` | 692 MB |
| `AzureLLMInferenceTrace_conv_1week.csv` | 1.14 GB |

Download the official files from the Azure Public Dataset GitHub release:

- Code-completion trace:
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv
- Conversational trace:
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv

The dataset documentation referenced by the Chronos work is available at:

https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md

The Chronos artifact repository is:

https://github.com/am-research/rtss-ttft-tbt

## Download with PowerShell

Run these commands from the project root:

```powershell
curl.exe -L -o ChronosTraces\AzureLLMInferenceTrace_code_1week.csv `
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv

curl.exe -L -o ChronosTraces\AzureLLMInferenceTrace_conv_1week.csv `
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
```

Expected file metadata:

| File | Bytes | SHA-256 |
|---|---:|---|
| `AzureLLMInferenceTrace_code_1week.csv` | 691,989,454 | `71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f` |
| `AzureLLMInferenceTrace_conv_1week.csv` | 1,135,195,393 | `a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941` |

The traces are distributed under the **CC-BY attribution license** declared by
`Azure/AzurePublicDataset`. Cite the Azure dataset and its associated
DynamoLLM paper when publishing results derived from these files.

After downloading both files, run:

```powershell
py run_chronos_trace.py
```

Generated analyses are written to:

```text
ExperimentalResults\ChronosTrace
```
