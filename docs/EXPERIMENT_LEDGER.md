# Selective synchronization execution ledger

This document is the operational entry point. The original run.sh and legacy hidden-client
launchers remain available. Research runs use system/main.py --sync-policy; do not mix legacy
hidden-count arguments with research policies.

## Setup

Use a fresh Python environment with the existing requirements.txt; Python 3.10+ is required
by the research code. Install torch/torchvision matched to the GPU. The generator uses scipy,
numpy and torchvision. The runner does not install packages or execute training implicitly
when building the ledger.

From the repository root:

    python -m unittest discover -s tests -v
    python scripts/build_job_ledger.py
    bash run_all_sync.sh --dry-run
    bash run_all_sync.sh --stage pilot --resume

The final command executes the documented 54 pilot training runs and their generation/validation
dependencies. With no arguments run_all_sync.sh defaults to pilot, not the entire campaign.
To execute all currently enumerated jobs explicitly:

    bash run_all_sync.sh --resume
    python scripts/aggregate_results.py

## Enumerated matrix

| Stage | Training configurations |
|---|---:|
| pilot: six policies x three regimes x three paired seeds | 54 |
| ablation: extra four regimes x six policies x three seeds | 72 |
| budget: five non-full policies x three regimes x three seeds x k=2,4,5 | 135 |
| threshold: three methods x three regimes x three seeds x four thresholds | 108 |
| reference/transport ablations: 14 alternatives x three regimes x three seeds | 126 |
| mechanism: six policies x three regimes x three seeds x eight alternatives | 432 |
| counterfactual: six policies x three regimes x three seeds | 54 |
| Total nominal training jobs | 981 |

21 dataset-generation and 21 validation jobs are additional. The builder is authoritative:
it deduplicates exact configuration keys. It writes every command, its dependencies, expected
artifacts and deterministic job ID to experiments/job_ledger.jsonl. Paths are resolved for
the current checkout; do not copy a generated ledger between machines.

Pilot regimes: near, separated, changing; paired seeds 42,43,44. Other regimes:
clustered, disjoint Dirichlet (not the original resampling generator), equal-histogram
conditional-domain stress and within-task drift. The current post-training variant uses
the same histogram after training, and retains unsynchronized local updates. It is not a
post-training gradient/update-statistic implementation.

## Scripts

| Script | Function |
|---|---|
| scripts/generate_streams.py | Disjoint CIFAR-100 training/validation/test allocation and windows |
| scripts/validate_streams.py | Verify IDs, counts and inventory |
| scripts/build_job_ledger.py | Enumerate canonical staged jobs |
| scripts/run_campaign.py | Dependency-ordered execution, logs, validation and retries |
| run_all_sync.sh | Single entry point for building and executing ledger |
| scripts/aggregate_results.py | Complete-run summaries and paired pilot contrasts |
| system/main.py | Actual training CLI |
| tests/test_sync_policies.py | Pure protocol tests |

Execute a stage independently with --stage budget, --stage threshold, --stage ablation,
--stage mechanism or --stage counterfactual. Use --job-id to isolate a job.
Use --dry-run before launching expensive runs. --resume skips valid completed jobs and
retries failed jobs without overwriting failed output. --device cpu is available for small
diagnostics, not recommended for a complete campaign. The current runner is serial;
run multiple independent ledgers only with separate result roots to avoid collisions.

## Accounting and output

Research outputs include configuration, provenance, round/decision records, logical
communication events, report arrivals, frozen-server metrics, final summary and server
checkpoint. Optional --sync-checkpoint saves a complete round checkpoint, and --sync-resume
restores it. Logical tensor bytes plus canonical JSON message envelopes are counted;
the common clock and availability are declared free control. These are not physical
network bytes and the current histogram encoding is JSON rather than the proposed uint32 wire format.

## Remaining research gates

The following are not executable jobs yet and must not be counted as completed:
rich feature/gradient/Jacobian/Bayesian/spectral signals, centralized cached coverage,
update-matching and impact prediction, adaptive hard byte caps, matched-byte threshold
calibration, combined methods, sparse peer networks and imperfect peer reconciliation,
multi-round counterfactual continuation, external baseline reproductions, held-out
confirmatory seeds and larger cohorts. The current ledger is a concrete histogram
campaign, not proof that the conditional research extensions work.

The same-checkpoint branch runner reuses independently trained candidates for one-round
aggregates; explicit branch equivalence testing and multi-round continuation remain
necessary before drawing causal conclusions. The pilot is not a publication-level
confirmation. Confirmatory tolerances and sample sizes must be frozen using development data.

No GPU experiment results are included in these commits.
