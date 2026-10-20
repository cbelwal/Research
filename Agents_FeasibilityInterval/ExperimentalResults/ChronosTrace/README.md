# Chronos/Azure trace analysis

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

| service      |     rows |   first_timestamp_us |   last_timestamp_us |   duration_seconds |   mean_rps |   mean_input_tokens |   mean_output_tokens |   median_input_tokens_approx |   p99_input_tokens_approx |   median_output_tokens_approx |   p99_output_tokens_approx |   interarrival_cv |   quantile_reservoir_size |
|:-------------|---------:|---------------------:|--------------------:|-------------------:|-----------:|--------------------:|---------------------:|-----------------------------:|--------------------------:|------------------------------:|---------------------------:|------------------:|--------------------------:|
| code         | 16803695 |     1715299200009930 |    1715903999929501 |             604800 |    27.7839 |             2511.28 |              22.6897 |                         1930 |                   7685    |                             8 |                        271 |           1.9627  |                    200000 |
| conversation | 27303999 |     1715472000001163 |    1716076799995460 |             604800 |    45.1455 |             1631.58 |             105.509  |                          931 |                   6653.01 |                            41 |                        697 |           1.69245 |                    200000 |

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
  alpha=0.08 ms/input token,
  beta=0.25 ms/output token,
  gamma=7.0 ms;
- all three costs are multiplied by 0.052500 to give the combined
  nominal trace a target utilization of 0.70
  in this non-batched single-resource model;
- relative deadline: 2000.0 ms;
- workload: a contiguous, timestamp-preserving combined window of
  50,000 requests selected deterministically.
- load factors: 0.5, 1.0, 1.25, 1.5;
  factors above 1 compress inter-arrival times.

These coefficients are hypotheses for stress-testing the paper's logic, not
measurements provided by Azure or Chronos.

## Realized replay-window load

| service      |   requests |   duration_seconds |   realized_rps |   mean_processing_ms |   realized_utilization |
|:-------------|-----------:|-------------------:|---------------:|---------------------:|-----------------------:|
| code         |      28785 |            364.766 |        78.9135 |             11.1015  |               0.876057 |
| conversation |      21215 |            364.766 |        58.1605 |              8.20062 |               0.476952 |
| combined     |      50000 |            364.766 |       137.074  |              9.87065 |               1.35301  |

The normalization uses week-wide mean rates, but the replay deliberately keeps
a contiguous real timestamp window. This selected window is substantially
burstier than the week-wide mean, so its realized utilization can exceed the
nominal target. `load_factor` multiplies the realized arrival rate and
utilization by compressing or expanding inter-arrival times.

## Policy results

|   load_factor | policy        | service      |   requests |   deadline_miss_rate |   mean_response_ms |   p99_response_ms |   total_preemptions |   total_aborts |   total_waste_seconds |   mean_waste_fraction |   release_count_bound_violations |   per_attempt_waste_bound_violations |
|--------------:|:--------------|:-------------|-----------:|---------------------:|-------------------:|------------------:|--------------------:|---------------:|----------------------:|----------------------:|---------------------------------:|-------------------------------------:|
|          0.5  | abort_restart | code         |      28785 |             0        |            89.5435 |          463.81   |                4518 |           4518 |               35.7302 |             0.0392817 |                                0 |                                    0 |
|          0.5  | abort_restart | conversation |      21215 |             0        |            20.3243 |           76.1359 |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          0.5  | nonpreemptive | code         |      28785 |             0        |            65.2203 |          277.587  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          0.5  | nonpreemptive | conversation |      21215 |             0        |            25.2145 |           84.9962 |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          0.5  | resume        | code         |      28785 |             0        |            68.4589 |          288.496  |                4101 |              0 |                0      |             0         |                                0 |                                    0 |
|          0.5  | resume        | conversation |      21215 |             0        |            20.3243 |           76.1359 |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1    | abort_restart | code         |      28785 |             0.990933 |        137627      |       193068      |                4877 |           4877 |               40.9186 |             0.0382177 |                                0 |                                    0 |
|          1    | abort_restart | conversation |      21215 |             0        |            27.1469 |           99.0143 |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1    | nonpreemptive | code         |      28785 |             0.977662 |         94297.9    |       139777      |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1    | nonpreemptive | conversation |      21215 |             0        |            36.0301 |          107.014  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1    | resume        | code         |      28785 |             0.977731 |         94303.9    |       139780      |                4877 |              0 |                0      |             0         |                                0 |                                    0 |
|          1    | resume        | conversation |      21215 |             0        |            27.1469 |           99.0143 |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.25 | abort_restart | code         |      28785 |             0.995796 |        188651      |       233967      |                3861 |           3861 |               31.6329 |             0.0283571 |                                0 |                                    0 |
|          1.25 | abort_restart | conversation |      21215 |             0        |            32.7926 |          120.775  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.25 | nonpreemptive | code         |      28785 |             0.992392 |        151768      |       202334      |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.25 | nonpreemptive | conversation |      21215 |             0        |            42.2389 |          133.341  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.25 | resume        | code         |      28785 |             0.992427 |        151774      |       202334      |                3861 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.25 | resume        | conversation |      21215 |             0        |            32.7926 |          120.775  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.5  | abort_restart | code         |      28785 |             0.997638 |        219985      |       273989      |                2828 |           2828 |               24.4287 |             0.0185165 |                                0 |                                    0 |
|          1.5  | abort_restart | conversation |      21215 |             0        |            53.0634 |          348.733  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.5  | nonpreemptive | code         |      28785 |             0.995345 |        189912      |       249560      |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.5  | nonpreemptive | conversation |      21215 |             0        |            62.0497 |          356.551  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.5  | resume        | code         |      28785 |             0.99538  |        189918      |       249560      |                2828 |              0 |                0      |             0         |                                0 |                                    0 |
|          1.5  | resume        | conversation |      21215 |             0        |            53.0634 |          348.733  |                   0 |              0 |                0      |             0         |                                0 |                                    0 |

## Key findings

1. The complete files contain 44,107,694
   valid requests. The measured counts, mean RPS, medians, and P99 token lengths
   closely reproduce Chronos's trace-characterization table.
2. At load factor 0.5, the selected burst window has approximately
   0.677 realized
   utilization and every request meets the 2-second deadline under all three
   policies.
3. At that schedulable load, abort/restart causes
   4,518 code-request aborts and
   35.73 seconds of discarded
   processing. Mean code response rises to
   89.54 ms, compared with
   68.46 ms under resume and
   65.22 ms under non-preemptive
   execution.
4. The same immediate preemption reduces mean conversation response to
   20.32 ms versus
   25.21 ms under
   non-preemptive execution. This is the paper's latency-versus-waste trade-off
   on a real arrival/token trace.
5. At load factor 1.0 the selected burst has realized utilization
   1.353, above capacity.
   Conversation requests remain responsive because they have strict priority,
   while 99.09% of code requests
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
