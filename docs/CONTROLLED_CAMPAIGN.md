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

The integer allocator first minimizes label allocation error subject to exact
row and class margins, then realizes within-class domain allocations while
preserving each allocated label count and the domain inventory. This is an
explicit lexicographic allocation objective, not an unconstrained joint fit.

## Statistical threshold and logical bandwidth extension

Four prespecified thresholds .05, .10, .20, .35 are crossed with all 24 selectors,
24 settings and three paired seeds. Threshold selection has no fixed cap; k=5
is only a CLI placeholder. Due and unknown clients are admitted, M3 elects
representatives until all valid peers are covered. Empty rounds preserve the
server model; distribution coverage is undefined when no client participates.
Counterfactuals enumerate subsets of the actual selected size, plus full.
This adds 6,912 trajectories: 14,472 training and 14,617 total jobs. The
fixed-count comparisons still use matched random/oldest/full. Threshold runs
are compared with full and displayed on realized byte/accuracy curves; there
is no claim of exact matched-count/byte random control for each threshold run.

All primary runs charge initial dense model distribution and ACKs, common
control, admission/election, statistics, model upload/download, download ACKs,
and optional refresh. Each binary statistic is charged according to its
actual encoded length. Model body size is exact dense native state tensor
bytes under an agreed schema, not physical network or torch.save file bytes.
Peer traffic counts once globally and at both endpoints. Per-event, round
(including -1 initialization), client-round and whole-run accounting is
reconciled by mandatory completion checks. No raw samples/replay items are
transmitted; local optimization and offline counterfactuals have no FL traffic.
Physical link throughput, packet overhead and hard-byte-cap variants are not
measured by this audit. Use --selection-mode threshold --threshold 0.2 to
filter threshold trajectories; --budget 3 selects only fixed-budget runs.

Audited communication events also preserve base64 of the exact serialized envelope (not model tensor contents), and its SHA-256; the verifier decodes and checks each envelope, tensor schema, message roster, category and direction. The peer roster communicates last-sync ages, and selected peers receive an explicit admission response. Treat derived statistical envelopes as restricted research output, not public releases.

Offline causal decision replay verifies every primary roster against historical local references or the bank actually delivered and current peer reports. Floating threshold ties now defer at equality; exact rational TV remains exact. A participation_schedule.json artifact supports subsequent random matched-count runs without sharing private statistics. The fixed four-skip safeguard is explicit; no adaptive age sweep is implied. Run identity hashes omit the absolute dataset path so that different checkout roots do not create duplicate run IDs.
