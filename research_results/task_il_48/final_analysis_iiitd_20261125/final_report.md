# FEDQCL Task-IL campaign: final execution and results report

## Campaign reconciliation

| Ledger category | Expected | Verified terminal | Remaining |
|---|---:|---:|---:|
| Training runs | 48 | 48 | 0 |
| Private dataset generation | 19 | 19 | 0 |
| Dataset validation | 19 | 19 | 0 |
| **All executable ledger jobs** | **86** | **86** | **0** |

All 48 training entries reached 80-or-configured-round terminal states, exited successfully, and passed the repository production validator plus independent terminal artifact verification. The verification receipt records 48 expected, 48 verified, and no failures. All 38 dataset dependencies are marked verified complete in the authoritative execution ledger. E05-05 and E10-02 remain separately executed as the specification requires; they are the same seed/configuration and are not independent replications.

- Training implementation revision: `38192b43b1f084333db415de388a240b8f3f35d1` (Task-IL runs use this revision).
- Dataset: CIFAR-100; evaluation split: test; default seed: 42.
- Method: conventional Task-IL with shared ResNet-18 backbone, task-specific heads, private local replay, SGD and the documented direct current-shard-weighted model aggregation.
- No selective-sync controller or unrelated optimization was included.
- Communication bytes are the implementation’s **logical payload plus protocol-envelope accounting**, not measured physical network traffic.

## Execution attempts and retries

The attempt ledger contains 90 records for 86 logical ledger jobs: 86 current attempts passed, while four earlier attempts are retained as superseded history. Three Task-IL attempts were intentionally stopped with SIGTERM only after a verified atomic checkpoint to migrate E08-01/02/03 for throughput; their resumed attempts finished and passed verification. One duplicate E06-03 attempt was superseded after the V100 attempt was selected; the V100 run finished and passed. There are no live attempts, unresolved failures, or incomplete logical jobs. The authoritative attempt ledger preserves the old records and their diagnostic evidence.

## Family-level results

Means below are descriptive arithmetic averages across entries in each family (one default seed). Families vary curricula, participation, synchronization, task widths, or exposure as specified; cross-family differences are not causal comparisons. Accuracy and forgetting are fractions in the source CSV; this table shows percentages. Communication is decimal GB per run. Optimizer-step, sample, and replay counts are mean per run. See `analysis/experiment_results.csv` for every individual run and the full metric fields.

| Family | Study | n | Mean final accuracy | Mean current-task accuracy | Mean previous-task accuracy | Mean forgetting | Mean current-task loss | Mean logical GB | Mean optimizer steps | Mean processed samples | Mean replay samples |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 | Participation and joint/disjoint curricula | 8 | 80.28% | 88.92% | 80.07% | 10.10% | 0.2587 | 47.39 | 35000 | 450000 | 427700 |
| E02 | Synchronization frequency | 3 | 81.12% | 75.40% | 81.75% | 6.16% | 0.4584 | 19.69 | 35000 | 450000 | 409500 |
| E03 | Client population | 6 | 79.07% | 84.53% | 79.42% | 7.60% | 0.3286 | 13.70 | 35000 | 450000 | 409500 |
| E04 | Simultaneous task sharing | 3 | 83.42% | 84.54% | 83.30% | 6.83% | 0.3754 | 27.18 | 35067 | 450000 | 428393 |
| E05 | Task width with partial participation | 8 | 68.25% | 67.31% | 68.52% | 6.89% | 0.8358 | 7.30 | 34775 | 450000 | 350350 |
| E06 | Cross-client task recurrence | 3 | 82.71% | 84.48% | 82.54% | 7.64% | 0.3546 | 27.18 | 35067 | 450000 | 439183 |
| E07 | Communication outage duration | 6 | 83.37% | 88.82% | 82.77% | 6.97% | 0.2690 | 21.70 | 35000 | 450000 | 409500 |
| E08 | Consecutive-task overlap | 3 | 52.50% | 52.71% | 52.49% | 10.00% | 1.4382 | 36.66 | 34967 | 450000 | 414700 |
| E09 | Class recurrence delay | 5 | 78.12% | 73.90% | 78.29% | 12.28% | 1.0688 | 23.13 | 35000 | 450000 | 414050 |
| E10 | Task width | 3 | 70.49% | 74.99% | 69.54% | 4.23% | 0.7466 | 9.81 | 34723 | 450000 | 315987 |

## Individual-run results and diagnostic outputs

`analysis/experiment_results.csv` contains 48 verified per-run summaries, including final/global and current/previous-task accuracy, loss, forgetting, backward transfer, logical communication bytes, uploads/downloads, optimizer steps, processed samples, replay samples and exact configuration fields. Detailed time-series and class/task/client diagnostics are in:

- `analysis/round_metrics.csv` — 3,112 rows.
- `analysis/task_metrics.csv` — 83,279 rows.
- `analysis/class_metrics.csv` — 205,193 rows.
- `analysis/client_diagnostics.csv` — 5,165 rows.
- `analysis/diagnostic_class_metrics.csv` — 10,330 rows.
- `analysis/baseline_contrasts.csv` — 44 matched/descriptive contrasts.
- `analysis/recurrence_gains.csv` — 220 recurrence records.
- `analysis/overlap_retention.csv` — 1,125 overlap records.
- `analysis/incomplete_jobs.csv` — empty (0 incomplete entries).
- `analysis/plots/` — family plots for accuracy, forgetting, loss, communication, sample/step budgets, and intervention diagnostics.

## Validation and provenance

- Final production verification: `verification/per_run_verification.summary.json` and `verification/per_run_verification.jsonl`; 48/48 verified at the recorded training revision.
- Analysis receipt: `analysis/receipt.json`; completed 48, expected 48, incomplete 0.
- Analysis input: `analysis_job_ledger.jsonl` (48 training entries), built from verified staged outputs with path references resolved to iiitd and all 48 production validators passing.
- Input mapping/provenance: `analysis_input_manifest.json`.
- Transferred bundle integrity: `final_analysis_manifest.sha256`; 299 listed files were compared by SHA-256 between iiitd and this local copy, with zero mismatches. Manifest SHA-256: `5ee22ceab8424a7452e5a11e0256a8f7fdb8d3dc01564dec66380903cf4945da`.
- Final analysis destination on iiitd: `/home/anurag/Projects/FedQCL_Task/research_results/task_il_48/analysis/`.
- Verified result collection on iiitd: `/home/anurag/Projects/FedQCL_Task/research_results/task_il_48/central_stage/` plus the four native iiitd E08/E09 result directories. Original result directories on Mtech and V100 were preserved.

## Interpretation limits

This campaign uses one seed, so it establishes verified execution and descriptive outcomes for the specified matrix, not statistical significance or robust ranking across random seeds. E10-02 duplicates E05-05 by design and must not be counted as an independent replication. Several families intentionally change exposure or curriculum; consult the exact schedule and per-run fields before making causal claims. Logical communication estimates do not substitute for network-level byte measurements.
