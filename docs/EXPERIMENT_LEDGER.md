# Selective synchronization execution ledger

## Setup and execution

Python 3.10+ is required. Install the repository requirements with a PyTorch/torchvision
pair suitable for the target GPU, plus scipy if not already installed. The original
run.sh remains a legacy entry point.

From the repository root:

```bash
python -m unittest discover -s tests -v
FEDQCL_TRAINING_TEST=1 python -m unittest discover -s tests -p test_sync_training.py -v
python scripts/build_job_ledger.py
bash run_all_sync.sh --dry-run
bash run_all_sync.sh --stage pilot --resume
```

The first two commands are correctness checks. The pilot has 54 neural-training jobs
plus its dataset-generation and validation dependencies. Do not start the whole
campaign until the CPU smoke, full-data single run and pilot have passed.

```bash
bash run_all_sync.sh --all
python scripts/aggregate_results.py
python scripts/match_budgets.py
python scripts/plot_results.py
```

The first command executes all currently enumerated jobs, not all scientifically
conceivable variants. It is extremely expensive. Without arguments the shell
entry point defaults to the pilot. Select stages using --stage, or an individual
job using --job-id. Use --device cpu or --gpu-id as needed. Serial by default; for one host with multiple GPUs use --workers 4 --gpu-ids 0,1,2,3. A per-ledger lock prevents concurrent ownership. Do not run different ledgers with overlapping artifact paths or device slots.

## Current nominal matrix

| Stage | Training jobs |
|---|---:|
| pilot | 54 |
| ablation (additional regimes and protocol variants) | 198 |
| budget | 135 |
| threshold | 108 |
| mechanism | 432 |
| counterfactual | 54 |
| signals | 108 |
| advanced (cached/update matching and combined) | 72 |
| impact_data (exhaustive development subsets) | 9 |
| adaptive (held-out seeds, two adaptive modes and baselines) | 63 |
| scale (10, 20, 50 clients) | 135 |
| extended_signals | 432 |
| rate_matched | 135 |
| network | 27 |
| continuation | 18 |
| robustness | 54 |
| math_signals | 297 |
| weighted | 117 |
| dynamic_threshold | 27 |
| combined_threshold | 144 |
| replay_profiles | 81 |
| correctness | 270 |
| hard_bytes | 180 |
| encoding | 108 |
| privacy | 54 |
| distributed | 45 |
| comparators | 36 |
| continuation_extended | 18 |
| data_extended | 108 |
| dataset_transfer | 18 |
| impact_data_strict | 75 |
| confirmatory | 165 |
| byte_matched | 108 |
| Total training | 3885 |

Other jobs: 168 generation, 168 validation, 2 impact fits, 243 schedule extractions, 67 analyses and 4 loopback benchmarks. Total: 4,537 jobs, verified by protocol CI at 14a3bbd9. This default ledger excludes optional TinyImageNet transfer (set TINYIMAGENET_ROOT to the extracted official dataset). The builder is authoritative and checks unique IDs and topological dependency order. Use --estimate-only for workload estimates.

Job IDs and training output paths bind the exact source commit. Dataset directories
bind the generator content hash. Rebuild the ledger after checking out a new source
commit. Previously completed results from a different source commit are not reused
as if they were produced by the current implementation. A generated ledger is
machine-specific because commands contain absolute paths.

## Stage details and causal information

- Pilot: five clients, five CIFAR-100 tasks, seeds 42/43/44, three clients after
  full bootstrap; near, separated and changing client relationships.
- Development: seeds 42/43/44 use the validation split for round evaluation.
- Adaptive: fit on development partition seed 42; calibrate on 43 and 44. Evaluate
  on held-out partition seeds 45/46/47 and their test split. These are preliminary
  held-out checks, not a sufficiently powered confirmatory statistical study.
- Strict: fit seeds 60–64, calibration 65–84, held-out 85–89. The 95% joint seed-group maximum-residual conformal bound assumes exchangeability; it is not a guarantee of transfer to arbitrary OOD populations. The held-out matrix is predeclared, but sample size must be justified from development-only variability before confirmatory execution. scripts/plan_confirmation.py provides a paired normal approximation.
- Impact fit: ridge current/past harm predictors and group-level upper residual
  calibration. The exploratory confidence setting is 0.5 because two calibration
  seeds cannot support a finite 95% distribution-free bound. Increase independent
  calibration groups before making stronger claims.
- Scaling: greedy selector at 10/20/50 clients; not exact combinatorial enumeration.
- Rate matched: replay only the selected counts of source threshold/adaptive runs.
  This is an offline count-matched random baseline, not a deployable scheduler;
  it does not imply exact equal total bytes.
- Network: sparse ring with central edge-distance arbiter and direct peer report
  loss followed by charged reliable relay reconciliation.
- Continuation: horizons 3 and 12 rounds, full participation after the initial
  alternative subset; actual/full and best/worst validation subsets. Explicit
  independent branch checks for actual/full aggregates.
- Robustness: fixed upload, rejection, download and unavailable-client schedules;
  optional charged replacement after failed upload in relay peer mode.

## Scripts

| Script | Function |
|---|---|
| scripts/generate_streams.py | Disjoint controlled CIFAR-100 shards, exact integer allocation, approximate target similarities |
| scripts/validate_streams.py | IDs, inventory, class counts and windows |
| scripts/build_job_ledger.py | Canonical jobs, source commit and dependencies |
| scripts/run_campaign.py | Dependency-aware parallel execution, GPU slots, artifact hashes, retries |
| scripts/plan_confirmation.py | Development-only paired power approximation |
| scripts/compare_matched_bytes.py | Achieved byte-matching gap and tolerance |
| scripts/benchmark_transport.py | TCP loopback application-frame benchmark |
| scripts/fit_impact.py | Development-only fit and group residual calibration |
| scripts/analyze_counterfactuals.py | Regret, opportunity, interactions and rank diagnostics |
| scripts/extract_count_schedule.py | Offline matched participant counts |
| scripts/aggregate_results.py | Completed-run tables and paired pilot contrasts |
| scripts/match_budgets.py | Development byte matching with explicit unmatched status |
| scripts/plot_results.py | Observed byte/accuracy/forgetting frontiers |
| run_all_sync.sh | Central execution entry |
| system/main.py | Training CLI |

## Accounting and checkpoints

Binary report envelopes default to uint32 counts or float64 real vectors plus canonical metadata. JSON, FP16 and INT8 are separate ablations; compressed multichannel reports are not enabled. Model tensor bytes remain analytic. Control traffic can be charged explicitly. Hard reliable pre-training byte caps support adaptive selection and campaign reserves. The reserve is a model-transfer lower bound, not a global feasibility guarantee. Byte-matched baselines use source per-round caps; attained totals are checked separately. Infeasible configurations can fail by design, and dependent aggregate jobs remain blocked until their prerequisites succeed.

The campaign saves a checkpoint at each committed round. The runner may resume it
with --sync-resume, or archive a failed output before retrying. The checkpoint
signature includes source commit, dataset manifest and external policy input hashes.
Use a new output directory for a different configuration.

## Verification and remaining gates

The protocol workflow at 14a3bbd9 discovered 50 tests, with 48 passed and two opt-in neural tests skipped in the pure job, and generated all 4,537 default jobs. The earlier complete workflow at 2c7a7378 passed 46 pure tests and both opt-in neural tests. Consult the current GitHub Actions workflow for the expanded extractor smoke. No full CIFAR/GPU research campaign has been executed. See IMPLEMENTATION_TRACEABILITY.md for limitations and external reproduction requirements.
