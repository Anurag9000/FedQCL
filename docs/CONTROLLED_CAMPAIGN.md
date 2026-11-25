# Active campaign: 101 primary + 22 targeted ten-client trajectories, seed 42

The 101-run five-client research matrix remains intact. A targeted ten-client extension directly tests absolute versus fractional omission. Three validation-only learning-rate prefixes (20 rounds each, no held-out evaluation) precede screening. The complete default ledger has 179 jobs: 50 generation/validation jobs (25 streams), three calibration prefixes + one calibration selection, 123 production trajectories, one shortlist, and one consolidated analysis. Materialized phases: 54 jobs before calibration, 82 after calibration, 179 after shortlist. Only seed 42 is used.

## Calibration

Three common-backbone full-participation prefixes test LR .005, .01, .02 for the first two tasks. Select maximal equal-weight mean validation current-task and past-task accuracy over the prefix, then lower forgetting, then smaller LR. Freeze that LR for every statistical method and baseline, including the ten-client extension. Keep SGD, 5 local epochs, batch 13, replay batch 13, unit replay coefficient, full model state FedAvg. The actual best LR is unknown until execution. No held-out test selection. The prefix calibration does not establish global optimality over all five tasks.

After the 27 five-client screening runs, select one finalist per M1/M2/M3 using the existing validation rank-sum rule. For each finalist, retain nominal thresholds .05/.10/.20/.35 if the four fixed-state predicted mean participation counts differ by at least .25 between adjacent levels; otherwise select four increasing grid points from 0 to 1 targeting mean counts 4,3,2,1 with greedy separation where possible. Store all grid points, predicted means, actual thresholds and nominal slot mapping. No guarantee of four distinct participation schedules; actual adaptive trajectories are required. The ten-client adaptive run reuses the exact threshold from the five-client nominal .20 slot (no retuning by fleet size).

## Training suite

| Phase | Five clients | Ten clients |
|---|---:|---:|
| Screening all 24 selectors + full/random/oldest | 27 | 0 |
| Independent distribution anchors S01/S03/S04/S10/S13 | 26 | 0 |
| Fixed count finalists + random, k=1/2/4 (five), 6/8/9 (ten) | 12 | 12 |
| Full ten-client baseline | 0 | 1 |
| Threshold finalists | 12 | 3 |
| Matched-count random | 12 | 3 |
| Matched cumulative-byte random | 12 | 3 |
| Total | 101 | 22 |

Five clients: 500 replay examples/client. Ten clients: 250/client to hold total capacity at 2,500. The S14N10 dataset is a disjoint integer-balanced split of every S14 client/window/class/brightness cell into two children. Allocation RNG is task-specific and independent of shard shuffle. Consolidated analysis verifies exact nested private sample IDs and windows. Total private data remains 44,600, validation 5,000, public 400, test 10,000. Workload per client approximately halves. Thus same total data and replay capacity, not same local data volume or optimizer steps; measure actual steps, samples and communication. Conditional brightness is a limited domain shift, not arbitrary concept drift.

Compare 5/5 vs 10/10, 4/5 vs 8/10 (same 20% omitted), 4/5 vs 9/10 (same one omitted), 3/5 vs 6/10 (same 40% omitted), 3/5 vs 8/10 (same two omitted), and 1/5 vs 6/10 (same four omitted). Compare losses relative to each fleet's own full baseline, actual bytes and normalized savings, and same-threshold adaptive participation. Ten-client neural evidence is targeted, not all budgets, all mechanisms/signals or all distributions. Ten clients generate 90 directed peer reports vs 20 for five. More clients do not guarantee less omission harm; weights, novelty, redundancy and historical state matter.

All 24 original datasets plus paired ten-client dataset are validated. All 32 five-client subsets at ten checkpoints per trajectory; ten-client neural counterfactuals at rounds 13/33 evaluate all actual-size subsets plus full (not all 1,024) to limit cost. Dataset-only subset analysis enumerates all 1,024 for the paired stream. Counterfactuals are one-round branches, not alternate 50-round trajectories.

## Synchronization

All clients bootstrap at round 0. Thereafter fixed k is exact, with due-client priority; k=1/2/3/4 for five, 6/8/9 for ten. Full is a separate baseline. In threshold mode no fixed count/cap: M1/M2 choose all due, missing-evidence or above-threshold clients; zero through N is possible when safeguards allow. M3 uses representative coverage and, with a nonempty roster, always seeds at least one representative (so 1..N). All clients receive round controls, selected clients alone train/upload/download, M3 peers exchange reports even if not selected; M2 piggybacks aggregate references. Max skip 4: due if r-last_sync>4; fixed cap can violate due deadlines, threshold serves all due. Exact per-event application bytes, not physical network bandwidth. Empty round means unchanged server model, not no communication.

## Execute

```bash
bash run_all_sync.sh --estimate-only
bash run_all_sync.sh --all
# Or explicit phases:
bash run_all_sync.sh --stage data --resume --device cpu
bash run_all_sync.sh --stage calibration --resume
python scripts/build_job_ledger.py
bash run_all_sync.sh --stage screening --stage shortlist --resume
python scripts/build_job_ledger.py --require-shortlist
bash run_all_sync.sh --stage generalization --stage participation --stage adaptive --stage scale --stage matched --stage postprocess --resume
```

Unfiltered execution automatically expands each phase under the ledger lock. Filtered execution does not expand. All results and receipts carry commit identity. Additional outputs: calibration.json, shortlist.json, scale_comparisons.csv, decision_sensitivity.jsonl, all_subset_metrics.jsonl, distribution_subset_metrics.jsonl. Threshold ID labels are nominal slots, actual thresholds in resolved config and output. Single-seed results exploratory; no cross-seed CIs. Secondary client counts 20/50, resource heterogeneity, compression, and physical networks remain optional future studies. `SYNC_SUITE=exhaustive` preserves prior factorial.

## Historical plan (superseded as default)

# Active compact controlled campaign (101 training runs)

The default `core` uses seed 42 for both partition and training. Five clients, five tasks, 50 rounds, sample-weighted complete FedAvg, uniform experience replay, pre-training admission, all existing statistical and communication instrumentation. No test metrics determine selection.

| Phase | Design | Trajectories |
|---|---|---:|
| Screening | S14, all 24 selectors at k=3, random/oldest at k=3, full k=5 | 27 |
| Generalization | S01/S03/S10/S13: three finalists + full/random each; S04 additionally oldest | 26 |
| Fixed sensitivity | S14, finalists and random at k=1/2/4 (reuse k=3/full) | 12 |
| Adaptive | S14, one finalist/mechanism at thresholds .05/.10/.20/.35 | 12 |
| Matched count | One random comparator per adaptive source | 12 |
| Matched bytes | One cumulative-ceiling random comparator per adaptive source | 12 |
| Total | | 101 |

All 24 distribution settings generated and independently validated, but only S01/S03/S04/S10/S13/S14 receive neural trajectories. All 32 distribution subsets per round for all 24 streams. All 32 candidate aggregation subsets at each of ten checkpoints in each run (32,320 evaluations; 5,050 isolated candidate local trainings). No counterfactual is an alternative long-term trajectory. Fixed-state sensitivity uses causal observed scores or peer reports and actual due/last-sync state; model tensor byte estimates are not total hypothetical network bytes. Actual transaction bandwidth remains exact logical payload + envelopes, not physical network throughput.

## Staging and execution

```bash
python scripts/build_job_ledger.py
bash run_all_sync.sh --estimate-only
bash run_all_sync.sh --stage data --resume --device cpu
bash run_all_sync.sh --stage screening --stage shortlist --resume
python scripts/build_job_ledger.py --require-shortlist
bash run_all_sync.sh --all
```

Or simply `bash run_all_sync.sh --all` to run screening and automatically expand under the runner lock. Before screening, the actual ledger has 76 jobs (48 data + 27 training + 1 shortlist); after a verified shortlist it has 151 (48 data + 101 training + 1 shortlist + 1 final analysis). The estimate prints both planned and materialized counts. Filtered runs never trigger implicit expansion. Rebuild after screening before filtering generalization/adaptive/participation. Resume retains screening job IDs and paths at the same commit. Changed source commits intentionally require new ledgers/results.

Finalists: within each M1/M2/M3 group rank the eight S14 screening configurations by validation final mean accuracy (higher), forgetting (lower), exact logical total bytes (lower), and mean joint distribution coverage regret (lower); equal-weight competition rank sum, selector ID tie-break. Record all inputs, ranks, validation/audit hashes and commit in shortlist.json. No held-out metrics used. Selection and evaluation on S14 are exploratory; six training streams and single seed do not imply generalization to all settings or uncertainty intervals. Matched controls constrain cumulative bytes (not strict equality or online caps); matched-count does not imply equal sample computation.

Final analysis keeps all prior tables, adds shortlist.json, distribution_subset_metrics.jsonl, all_subset_metrics.jsonl, decision_sensitivity.jsonl. Single-seed SD/CI are null, absent baselines are not fabricated, missing heatmap cells are masked. Ground-truth settings includes dataset-only settings, explicitly labeled. Final held-out test results are reported but not used to shortlist. Secondary sensitivity and physical networking remain outside this primary campaign.

## Optional historical suites

`SYNC_SUITE=exhaustive bash run_all_sync.sh --estimate-only` retains the former 35,208-run / 35,353-job matrix (three seeds); `SYNC_SUITE=pilot` and `SYNC_SUITE=extended` retain prior separate suites. Do not run them by default.

## Historical exhaustive specification (superseded as default)

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

## Matched controls and independent final evaluation

For each threshold trajectory a random comparator reproduces its exact participant
count in every round (including zero). Each statistical trajectory, fixed or
threshold, also has a random comparator constrained by the source's cumulative
application-byte curve. The offline plan uses only sample-count message metadata,
model schema, source byte totals and the seeded random/age scheduler. It budgets
whole-model exchanges through the actual serializer, including controls, and
verifies actual per-round rosters and bytes. It does not pad or claim exact
byte equality: unused budget is reported; infeasible common traffic fails closed.
All comparator dependencies point to the source trajectory and validated stream.
The 6,912 count controls and 13,824 byte controls increase the default to
35,208 training trajectories and 35,353 jobs. Final held-out test metrics are
recorded only after optimization and all validation decisions finish. Validation
remains the basis for model/threshold selection; test metrics must not be used
to tune the suite. Count matching does not imply identical processed samples on
unequal-workload streams. The new comparison is logical bytes, not throughput.

The consolidated analyzer additionally records per-round validation accuracy against
actual cumulative and task-local bytes, first attainment of prespecified 50/70/90%
current-task validation accuracy (unreached targets stay null), and paired
comparisons at common within-task byte checkpoints, without interpolation or
extrapolation across tasks. Three-objective Pareto membership is based only on
validation final accuracy, forgetting and total bytes, separately per setting.
Paired seed-level t intervals are exploratory (n=3; no multiplicity-adjusted
significance claim). Held-out test metrics are reported separately.
