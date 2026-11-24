# Task-IL analysis campaign: 48 runs

This opt-in campaign is separate from the existing selective-synchronization
and FEDQCL queue-regulation studies. It uses CIFAR-100, the existing ResNet-18,
ordinary private task-balanced experience replay, SGD, and direct sample-weighted
FedAvg model aggregation. No shared raw-data
archive, intelligent selection, class-incremental method, queue weighting,
or extra terminal flush is involved.

## Run

From the repository root with the existing requirements (and scipy) installed:

    bash run_all_task_il.sh --dry-run
    bash run_all_task_il.sh --estimate-only
    bash run_all_task_il.sh --stage experiment_01
    bash run_all_task_il.sh --gpu-ids 0,1 --workers 2
    bash run_all_task_il.sh --device cpu
    TASK_IL_SEEDS="42 43 44" TASK_IL_SPLIT=validation bash run_all_task_il.sh --dry-run

The no-argument launcher executes all ten studies in dependency/priority order,
resuming validated outputs. The committed portable ledger is
configs/task_il/job_ledger.jsonl (48 parameter entries). The independently
CI-generated default dependency graph is checked in as
configs/task_il/execution_dag.jsonl (86 jobs: 48 train, 19 generate, 19 validate).
The builder verifies the default graph and resolves it into
experiments/task_il_48/job_ledger.jsonl, with commands, source commit, schedule
hashes, data generation and validation dependencies, and artifact paths. The
resolved ledger is machine- and checkout-specific and intentionally not tracked.
The runner's ordinary --stage, --job-id, --retry-failed, --gpu-id and --workers
options remain available. Results and private data are ignored by Git. The
analysis script emits CSVs and figures only for validated completed runs.

## Experimental matrix

| Family | Runs | Variations |
|---|---:|---|
| 1 | 8 | Joint/disjoint; 0/1/2/3 clients offline |
| 2 | 3 | 1/2/10 syncs per task; reuse E01-05 at 5 |
| 3 | 6 | 2/3/10 clients, 1/5 syncs; reuse 5-client E02-01 and E01-05 |
| 4 | 3 | 2/3/4 clients concurrently sharing task |
| 5 | 8 | Width 4/5, offline 0/1/2/3 |
| 6 | 3 | Task recurring on 2/3/4 different clients |
| 7 | 6 | 1/2/3/5/10/45-round communication-only outage |
| 8 | 3 | Width 5, exact overlap 2/3/4 |
| 9 | 5 | Recurrence after 1/2/5/10/15 intervening rounds |
| 10 | 3 | Width 3/5/10 |
| Total | 48 | One seed by default; independent seed expansion supported |

E10-02 deliberately executes the same width-5 full configuration as E05-05,
rather than silently reducing the requested 48-run count. Treat them as the same
seed/configuration, not as independent evidence. A seed expansion is not
automatically a confirmatory power calculation.

## Frozen design choices

- Joint: same two-class task on all clients at each stage, distinct private
  examples. Disjoint: different class sets simultaneously, globally disjoint
  across the base task inventory. Both cover 100 classes; width-2 five-client
  joint has 250 opportunities, disjoint has 50. Each source training example
  is allocated exactly once and sees ten local epochs. Round counts and replay
  exposure differ by design and are logged.
- Each task head is keyed by a stable sorted class set, and each client episode
  has its own start/end and owner. Labels are mapped explicitly. A final
  singleton head for width 3 is retained exactly as specified; its accuracy is
  trivial and its chance-adjusted score is null.
- E04: five target class pairs, each shared concurrently by m clients at one
  stage, with target client sets rotated. E06: the same five target pairs recur
  on m different clients at successive stages, with rotated recipients. Both
  have a 13-stage horizon, balancing remaining tasks and allowing idle slots.
  Each class's examples are divided between its scheduled occurrences.
- E08: each client has a private 20-class universe and 16 five-class episodes
  with exactly the requested overlap between successive local tasks. All
  configurations cover the same 100 classes and use the same episode count;
  each class's finite sample inventory is split across occurrences. Different
  overlap necessarily changes the per-class episode allocation.
- E09: client 0 repeats its first two-class task after exactly the specified
  number of intervening communication rounds. Whole filler episodes occupy
  the gap where possible, other clients ensure nonempty federation rounds,
  and the final horizon is 65 rounds. The gap is defined on new current-task
  exposure, not replay; replay remains enabled. The task inventory, owners,
  occurrence counts and nominal per-example exposure are held fixed.
- E07: client 0 disconnects in rounds 1 through D inclusive (0-based), trains
  throughout, and reconnects at round D+1. No terminal flush. The 45-round
  configuration has four remaining rounds including reconnection.
- All active clients train irrespective of selection. An idle client has no
  episode and is not an eligible uploader. Every round samples exactly the
  requested budget from eligible active clients (or all if fewer are active).
  No bootstrap training override or mandatory deadline. Initial distribution
  to all N clients is separately charged as MODEL_INITIAL. Selection uses a
  seeded uniform permutation independent of training and replay.
- Default five rounds per episode, two local epochs per round. Frequency
  studies retain ten local epochs per episode. Each client stores at most 500
  private examples, uniformly selected without replacement within task-balanced
  quotas. Each SGD step uses at most 13 current examples and one pooled replay
  batch of at most 13 historical examples total. The loss is current mean CE
  plus replay sample-mean CE, with coefficient 1. No dropped minibatches.
  Replay is sampled uniformly without replacement within each step.
- Global aggregation is direct sample-weighted FedAvg of the submitted models,
  including floating buffers (integer counters use maximum). Current private
  shard sizes are the weights. Stale models are averaged as submitted; the
  communication-only outage schedule is preserved. No buffered delta,
  staleness attenuation, server optimizer or extra reference-model correction.
  The shared backbone ID is fedavg_uniform_er_v1.
- 50 original training examples per class are reserved for validation.
  Original test data is held out. The default test evaluation is for a frozen
  specification; use TASK_IL_SPLIT=validation during development. Do not tune
  on test results. Recurring/overlapping task contexts may reuse the same
  evaluation images; they are never used in training or replay.
- Communication accounting uses the existing binary logical transport.
  MODEL_INITIAL, SYNC_REQUEST, MODEL_UPLOAD, MODEL_DOWNLOAD envelopes plus
  actual complete-state tensor byte sizes are counted per recipient. The
  predeclared common clock and curriculum are assumed locally available;
  no physical TCP/IP traffic is measured. No telemetry reports are exchanged.

## Outputs and verification

Each run records config, full schedule, source/data provenance, per-round
selected/trained/accepted/downloaded IDs, model versions/lags, current and replay
sample exposure, optimizer steps, per-task/class accuracy and CE, task-macro
and sample-micro global accuracy, current and previous task accuracy, forgetting,
chance-adjusted accuracy, local and exposure diagnostics, per-message bytes,
cumulative bytes, atomic checkpoint and final server state. Final summaries
explicitly identify validation or test. The final reported model is the
primary result, with no unrequested extra synchronization.

Data validation requires unique source IDs across all private episode shards,
disjoint train/validation ownership, exact complete source inventories, label
membership and per-shard hashes. Private replay stores only bounded cloned
exemplars with their source IDs, client-local episode identity and labels.
Completed raw shards are evicted from the in-memory cache; replay never reopens
an old raw shard. The server receives sample counts, not private data. A
source-bound runner checks resolved config and dataset fingerprints, expected
schedule rows, participation, exposure and byte totals before accepting completion.

    python -m unittest discover -s tests -p test_task_il.py -v
    TASK_IL_TRAINING_TEST=1 python -m unittest discover -s tests -p test_task_il.py -v

The tests and ledger are correctness gates, not evidence that the real CIFAR-100
campaign has completed. Execute full-data smoke tests and review the actual
results before drawing empirical conclusions.
