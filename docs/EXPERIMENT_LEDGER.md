# Active campaign — merged Tier-A only

Exactly **9 training trajectories + 3 supporting jobs = 12 jobs**.
This is a literal Tier-A-only interpretation of the audit. The Tier-S
replay-aware gradient family is included only as the parent of Tier-A
current/past gradient interference; standalone label-distribution and
model-update experiments are excluded. No additional baselines or Task-IL.

| Family | M1 last successful sync | M2 pooled historical server report | M3 direct peer selection |
|---|---|---|---|
| Class-conditional feature prototypes + activation moments | A01-M1 | A01-M2 | A01-M3 |
| Current/replay gradient similarity + interference diagnostic | A02-M1 | A02-M2 | A02-M3 |
| Functional predictive similarity | A03-M1 | A03-M2 | A03-M3 |

Replay loss and within-task deterioration on fixed private memory probes
are logged across every run; not standalone selection signals. Interference
is logged as signed current/past shared-backbone gradient-sketch cosine;
the selector compares the two gradient channels separately using normalized
Euclidean discrepancy (direction and magnitude). All reports preserve
explicit missing-channel support; M2 pools per-block sufficient statistics.
Feature mean and second moment of normalized 32-dimensional projected
activations use one frozen learned encoder, the model already distributed
to all clients at bootstrap. No additional encoder download, no future data.
Missing class or task coverage is penalized; past/current channel groups
receive equal weight when both available. The anchor is checkpointed.
Feature reports start after bootstrap. The gradient probe uses native local
model coordinates (stale-model confounding remains), not a common refreshed model.
Functional similarity compares aligned probabilities with normalized JS on
four public CIFAR-100 training-source examples per class, reserved BEFORE
private allocation and disjoint from training/validation/test. Every
experiment uses the same partition, including the reservation (44,600
private train examples, 5,000 validation, 400 public probes, 10,000 test).
Only encountered task probes are used; probes are preprovisioned from the
public dataset, not sent from private client datasets. Public data provisioning
is outside simulated per-round communication accounting.

Fixed learning backbone: FedAvg + vanilla ER, ResNet-18, SGD .005,
private memory 500, batch 13, replay batch 13; five clients, k=3 (all at
round 0), five tasks of 20 classes, 10 rounds/task, 5 local epochs/round,
clustered CIFAR-100, seed 42, validation evaluation. Pre-training
partial participation remains unchanged. M1 uses last successful sync rather
than previous round per audit recommendation. M2 uses latest compatible
accepted other-client reports, not raw examples. M3 uses exact peer
representative selection. Single seed, no independent baseline; descriptive
not confirmatory. No claims of formal privacy or physical network execution.

```bash
python scripts/build_job_ledger.py
bash run_all_sync.sh --dry-run
bash run_all_sync.sh --estimate-only
bash run_all_sync.sh --all
```

Portable ledger: configs/sync/core_signal_ledger.jsonl.
Resolved ledger: experiments/job_ledger.jsonl.
Results: research_results/signal-matrix-<commit>/matrix.csv and matrix.json.
Only 9 training jobs are active; extended/Task-IL are explicit separate
historical campaigns. Old results cannot be reused as the new design.

---

# Historical catalogs (not active)

# Active research campaign: exactly 27 training configurations

The only default campaign is the prespecified 9 signal × 3 mechanism matrix in
[configs/sync/core_signal_ledger.jsonl](../configs/sync/core_signal_ledger.jsonl).
It has **30 executable jobs**: one shared dataset generation, one dataset
validation, 27 training trajectories and one dependency-gated final analysis.
Do not run Task-IL, full/random baselines, seed sweeps, other signals or the
extended catalog for this goal. Those separate historical studies remain opt-in.

| Signal | Extractor / distance | M1 | M2 | M3 |
|---|---|---|---|---|
| Replay | Prior-task exemplar class counts / TV | S01-M1 | S01-M2 | S01-M3 |
| Gradients | Mean gradient CountSketch / cosine | S02-M1 | S02-M2 | S02-M3 |
| Activations | Fixed common anchor feature means and variances / bounded L2 | S03-M1 | S03-M2 | S03-M3 |
| Jensen–Shannon | Current label histogram / normalized JS | S04-M1 | S04-M2 | S04-M3 |
| Mutual information | Plug-in MI between current labels and predictions / bounded L2 | S05-M1 | S05-M2 | S05-M3 |
| Bayesian | Diagonal empirical-Fisher/Laplace head statistics / bounded L2 | S06-M1 | S06-M2 | S06-M3 |
| CKA | Aligned public synthetic probe centered Gram / 1 − linear CKA | S07-M1 | S07-M2 | S07-M3 |
| Gradient variance | Between-minibatch gradient-sketch variance / bounded L2 | S08-M1 | S08-M2 | S08-M3 |
| Fisher | Empirical diagonal task-head Fisher sketch / bounded L2 | S09-M1 | S09-M2 | S09-M3 |

M1 compares against its most recent valid local observation, not only its
last synchronization. M2 receives a server-computed sample-count-weighted
aggregate of latest compatible accepted reports from other clients, piggybacked
on downloads; age is conservatively the oldest contributing round. M3
exchanges reports directly and performs exact representative selection.
All use fixed k=3, with five clients at bootstrap and deadline priority.

Fixed settings: one shared clustered non-IID CIFAR-100 partition, seed 42
(partition/training), 5 clients, 5 tasks of 20 classes, 10 rounds/task,
5 local epochs/round, ResNet-18, plain sample-weighted full-state FedAvg,
SGD learning rate .005, current batch 13, private 500-example task-balanced
uniform replay, one pooled replay batch of 13 and mean CE + mean replay CE.
Selection is pre-training. Model and report information remain as in the
existing simulator; no new privacy guarantee is claimed. Validation evaluation.

Replay reports are unavailable before the first memory capture and mostly
unchanged within a task; allow cross-task comparisons in common replay-label
coordinates. Other reports remain task-specific. CKA uses 32 aligned public
synthetic probes, not unpaired private examples; M2 compares to a pooled
normalized centered Gram reference, not the actual global model's CKA.
Bayesian inference is head-only approximate; Fisher is head-only empirical;
gradient variance is between minibatches of projected gradients. The MI
estimate is finite-sample and scalar. Gradients are compared in shared native
parameter coordinates, not claimed to be invariant to learned model differences.

Single seed, no full/random baselines: this is a descriptive comparison of
signals and mechanisms, not a powered confirmatory test or absolute savings
claim versus full participation. All 27 are required, no automatic expansions.

```bash
python scripts/build_job_ledger.py
python scripts/run_campaign.py --dry-run
python scripts/run_campaign.py --estimate-only
bash run_all_sync.sh --all
```

The no-argument launcher also executes the complete 30-job default ledger.
Machine-specific resolved ledger: experiments/job_ledger.jsonl.
Output: research_results/signal-matrix-<commit>/matrix.csv and matrix.json.
Old checkpoints/results are commit-bound and cannot be blindly reused.
The separate old Task-IL scripts and explicit extended profile are NOT part
of the active goal. A multi-host coordinator must still prevent duplicate jobs.

---

## Historical expanded-catalog reference (not default)

# Mandatory research backbone: FedAvg + vanilla ER

All current research training configurations, including the separate 48-entry
Task-IL suite, use backbone `fedavg_uniform_er_v1`. No FEDQCL queue, drift
constraint, historical reference model, V-scaled loss or FedBuff-style stale
delta is used. Original legacy launchers are separate and are not research jobs.

- Standard SGD, ResNet-18 with task-specific heads, current shard sample-weighted
  full-state FedAvg. Selected stale models are averaged directly, not corrected.
- 500 private stored examples per client, task-balanced quotas, random uniform
  sampling without replacement. One pooled replay minibatch of at most 13
  examples total per optimizer step; mean current CE plus mean replay CE.
- Capture at observed task completion regardless of synchronization; a windowed
  task admits only examples from windows actually revealed to that client.
- All research configurations use the same memory strategy, loss, aggregation
  and current-data sampler. Synchronization policies and original schedule
  differences remain the experimental variables.
- Removed redundant ER, local capture, full-state aggregation ablations,
  balanced sampler, alternative memory strategies, FEDQCL correction ablations,
  queue-dependent report and multiplicity-weighted mass comparator. Remaining
  experiments are retained. The generated ledger is authoritative; the old
  4,537-job count is superseded. Current protocol CI generated 3,934 jobs (3,282 training, 652 supporting) and the separate 86-job Task-IL DAG (48 training, 38 supporting). The two ledgers contain 4,020 entries total, 3,330 training entries. E10-02 is an intentional duplicate of E05-05; do not treat it as independent evidence.

```bash
python scripts/build_job_ledger.py
python scripts/run_campaign.py --dry-run
bash run_all_sync.sh --stage pilot --resume
bash run_all_sync.sh --all --workers 4 --gpu-ids 0,1,2,3
bash run_all_task_il.sh --dry-run
bash run_all_task_il.sh
```

The generated ledgers are commit-bound. Previous FEDQCL or FedBuff checkpoints
and results are not compatible with this backbone and cannot be reused as
completed research runs. See TASK_IL_EXPERIMENT_SPEC.md for the portable 48-run
matrix and IMPLEMENTATION_TRACEABILITY.md for remaining scientific limitations.
Full experiment execution is not implied by ledger generation or CI smoke tests.

## Historical pre-conversion ledger reference (superseded counts and learning variants)

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
