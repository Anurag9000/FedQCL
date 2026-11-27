# Temporal-overlap Task-IL mirror (joint in the temporal sense)

This independent, opt-in campaign mirrors the full 48-entry Task-IL catalog,
not its smaller default core subset. Here **joint** means consecutive local
tasks overlap, e.g. {0,1} -> {1,2} -> {2,3}. This is orthogonal to the original
catalog's "joint" flag (all clients share the same current task at the same
time). That original cross-client condition is preserved as an experimental
factor. Class IDs are seeded, shuffled and illustrative examples are not
literal CIFAR class orders.

## Launch and ledgers

    bash run_all_task_il_joint.sh --dry-run
    bash run_all_task_il_joint.sh --estimate-only
    bash run_all_task_il_joint.sh
    bash run_all_task_il_joint.sh --stage experiment_01
    bash run_all_task_il_joint.sh --gpu-ids 0,1 --workers 2

The dedicated launcher forces TASK_IL_SUITE=joint. Alternatively use
TASK_IL_SUITE=joint bash run_all_task_il.sh. It uses the same runner, strict
FedAvg + pooled vanilla ER training entry point, data validator, checkpoint
and artifact acceptance mechanisms as the base suite.

- Portable parameters: configs/task_il_joint/job_ledger.jsonl
- Portable dataset/training DAG: configs/task_il_joint/execution_dag.jsonl
- Resolved ledger: experiments/task_il_joint/job_ledger.jsonl
- Dataset root: research_data/task_il_joint/
- Output root: research_results/task_il_joint/
- Postprocessing is a dependency-gated final job; analysis requires all
  mirrored training runs to pass the shared verifier.

The default seed-42 graph has 44 training jobs, 16 data generation jobs,
16 validation jobs and one additional final analysis job (77 resolved jobs).
No base-suite training jobs are launched by this suite. The original
execution DAG, ledger, default core and extended launcher are unchanged.

## Deduplication and scope

Jxx-yy mirrors Exx-yy, preserving every original parameter other than the
explicit temporal-overlap flag and the necessary curriculum consequences.
E08-01, E08-02, E08-03 are already temporal-overlap interventions and are not
rerun. They remain in the original catalog. E10-02 would be identical to
J05-05 after mirroring; J05-05 is the single executable width-five baseline
and is reused in family-10 analysis. E02/E03 reuse their corresponding J01
and J02 baselines. All four original cross-client-joint E01 entries are
mirrored because their original temporal class sets did not overlap.

| Family | New training entries |
|---|---:|
| E01 partial participation | 8 |
| E02 synchronization frequency | 3 |
| E03 population | 6 |
| E04 simultaneous sharing | 3 |
| E05 width and partial participation | 8 |
| E06 cross-client recurrence | 3 |
| E07 outage | 6 |
| E08 existing temporal overlap | 0 |
| E09 recurrence delay | 5 |
| E10 width | 2 |
| Total | 44 |

## Frozen schedule design

Insert bridge task heads between each consecutive pair of original task
anchors, replacing one class at a time. For width w, every distinct
consecutive local task pair shares exactly w-1 classes: 1/2, 2/3, 3/4,
4/5 and 9/10 for the respective widths. For width 2 the transition from
{0,1} to {2,3} becomes {0,1}, {1,2}, {2,3}. These are ordinary Task-IL heads,
not a Class-IL classifier. Original class-set anchors and client membership
are preserved; bridge windows are inserted in aligned stage slots, with
globally empty slots removed. In the original cross-client-joint condition,
all clients also share bridge windows. In ordinary disjoint-client conditions
the private class universes remain disjoint across clients. In E04/E06
unique bridge identities avoid accidental additional shared/recurrent task
identities. Their five target tasks retain the required m-client simultaneous
or sequential exposure. Temporal overlap naturally creates additional
class-level exposure, distinct from the targeted task identity intervention.

E09 uses a special matched-inventory schedule: a repeated target pair,
three distinct anchor-sharing filler tasks, and the ordinary sliding
bridge path. The delay is exactly 1/2/5/10/15 intervening synchronizations.
A one- or two-round delay cannot accommodate a five-round intervening task:
the repeated identical target is the sole exception to the distinct-window
overlap rule in those cases. All other adjacent local episodes share exactly
one class. All five delays use the same class sets, occurrence inventory,
private allocation and 125-round horizon; short gaps can contain idle
client rounds, while other clients keep the federation active. E07 keeps
the original exact outage duration in rounds and extends the surrounding
curriculum naturally. E10 width 3 includes the original remainder class
in a full-width overlapping final window rather than a singleton head;
this is required for a meaningful temporal transition and changes the
chance baseline relative to the original singleton.

All 100 classes are covered. Every original source example is allocated
once across all private episodes, including recurring/overlapping class
occurrences. No raw example crosses clients or is reused as a new current
sample. Each episode has ten local epochs; sample inventories and total
current-example exposures are conserved, but the number of task heads,
rounds, minibatch boundaries, replay exposure and evaluation workload may
change. These are not pure equal-communication-budget contrasts. E01
five-client joint-across-clients has 495 sync opportunities rather than
250; the disjoint-across-clients mirror has 95 rather than 50. Report
comparisons against actual sample exposure, optimizer steps and bytes.
Evaluation includes each introduced task context every round. Shared
classes have multiple contexts, so task-micro scores count context/sample
pairs rather than unique test examples.

The shared fixed backbone is fedavg_uniform_er_v1: direct sample-weighted
FedAvg, private task-balanced uniform memory admission (500 total samples
per client), 13 current examples per batch, up to 13 pooled replay examples
per step, sample-mean replay CE plus current CE, SGD LR .005. No stale-delta
scaling, intelligent selection or FEDQCL queue regulation. No production
run is claimed complete on the strength of synthetic tests.

## Tests and outputs

    python -m unittest discover -s tests -p test_task_il_joint.py -v
    TASK_IL_TRAINING_TEST=1 python -m unittest discover -s tests -p test_task_il_joint.py -v

The tests check the entire matrix for three seeds, exact overlap and special
recurrence exceptions, cross-client conditions, outages, matching data,
private sample allocation, dependency graph, fixed backbone and synthetic
training/checkpoint/resume/result acceptance. Existing tests continue to
exercise the unmodified original suite. The shared analysis produces
per-round/task/class metrics, overlap retention for every mirror run,
recurrence diagnostics, baseline contrasts, plots and a completion receipt.

The existing runner still uses exclusive device slots per local ledger.
A two-host VRAM-aware campaign coordinator must implement shared claims
and measured admission as described in the operator instructions; this
launcher does not claim to be a distributed scheduler. Use the unique
J-prefixed IDs and separate ledger when adding it to that coordinator.
