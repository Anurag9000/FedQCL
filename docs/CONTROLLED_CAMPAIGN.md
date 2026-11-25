# Comprehensive controlled-stream campaign

The default synchronization ledger is now the prespecified comprehensive study.
The earlier 24-run single-stream campaign remains available as an explicit
pilot (SYNC_SUITE=pilot). The unrelated historical extended and Task-IL
ledgers remain unchanged.

## Design

24 independent settings S01–S24, three matched partition/training seeds
42,43,44. Every setting/seed has 24 signal/mechanism selectors at k=1,2,3,4,
plus random with identical age safeguard and oldest-first at each k.
One full participation reference per setting/seed (no redundant k=5 selector
runs). All use FedAvg + vanilla ER, 5 clients, 5 tasks, 50 rounds, same model,
local optimizer, memory, and ten prespecified counterfactual checkpoints.
Full bootstrap in round zero.

The catalog is configs/sync/campaign.json; each setting is a separate immutable
JSON. S01–S14 are the originally proposed conditions. S15 recurring,
S16 convergence, S17 divergence, S18 similar past/different current,
S19 gradual reassignment, S20–S24 balanced Dirichlet draws at alpha
100,10,1,.1,.01. S01/S03/S13/S18 form the four historical/current combinations.
Dirichlet rows are projected to fixed class and client margins; realized
heterogeneity, not nominal alpha, is the independent descriptive measure.
S12 changes client weights while keeping group sample mass approximately
matched. S02 targets the same mean pairwise TV as S04 but has a diffuse
pairwise distance pattern. Task classes remain disjoint. Within-class
brightness domains are a controlled covariate proxy, not semantic drift.

## Ledger

- 72 independent datasets; 72 generation and 72 validation jobs.
- 6,912 signal/mechanism trajectories; 576 matched-budget baseline runs;
  72 full references. 7,560 training trajectories.
- One consolidated analysis; 7,705 total jobs.
- 75,600 counterfactual checkpoints, 378,000 extra local candidate trainings.
- Exact subset evaluations for each actual budget and full reference.
- Budget 5 has no selective decision, so it is represented by full once.

## Execution

```bash
python scripts/build_job_ledger.py
python scripts/run_campaign.py --estimate-only
bash run_all_sync.sh --stage data --setting S14 --seed 42 --resume --device cpu
python scripts/preflight_sync.py --setting S14 --seed 42 --output preflight_summary.json
python scripts/profile_sync.py --experiment A08-M3 --setting S14 --seed 42 --budget 3 --stop-after-round 13 --output sync_profiles/S14
bash run_all_sync.sh --all --setting S14 --seed 42 --budget 3
bash run_all_sync.sh --all
```

Filters select disjoint groups with dependency closure and do not mark
consolidated analysis complete. Unfiltered postprocess requires every
trajectory. The local ledger runner uses exclusive GPU slots and a local
ledger lock; this is not a multi-host transactional distributed scheduler.
Coordinate disjoint ownership and identical source commit/stream before
parallel multi-host execution. Do not publish restricted allocation records
or private checkpoints. This large study requires substantial storage and
measured GPU/CPU/I/O capacity; no runtime guarantee is implied.

## Analysis

Each setting/seed has one verified truth hash; no cross-setting pooling of
raw distributions as if they were identical. The analyzer generates per-run
metrics, paired differences vs random/oldest/full, per-setting/budget/selector
three-seed descriptive means and standard deviations, realized distribution
statistics, detailed JSONL audit tables and effectiveness heatmaps. It rejects
incomplete matrices. Baselines do not calculate or transmit admission
signals. They retain researcher-only distribution diagnostics and
counterfactual validation.

This is a bounded comprehensive factorial over the named distribution
settings, selectors, nontrivial budgets and seeds, not every conceivable
FL hyperparameter combination. Fleet size, memory size, label semantics,
model-refresh policy, availability/failures and communication-only (train all,
select upload) are held fixed; their separate factorials are not claimed.
Counterfactuals are immediate alternatives, not complete trajectories.
Three paired seeds are descriptive evidence, not a guarantee of statistical
significance. Independently verified realized profiles and actual replay
histories govern interpretation.
