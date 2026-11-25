# Controlled-stream experiment contract

The 24 matched trajectories use the same immutable generated CIFAR-100 stream.
The design is configs/sync/controlled_stream.json. Five regimes recur once
per task in cyclic order (two windows each). Similar, clustered, separated,
reassigned and conditional regimes use explicit profiles; independent
Dirichlet alpha and persistence are supported in the optional dirichlet mode,
not swept in the default ledger. Joint integer transportation preserves all
private class/domain inventory and balanced client-window workload.
Conditional domains are equal-frequency within-class brightness ranks, with
sample ID tie breaks; brightness is a controlled proxy, not semantic truth.
Validation independently reconstructs domains from normalized shard pixels.
Realization tolerances are prespecified and failures block training.

Ground truth consists of requested/realized label and joint counts, spatial,
temporal, cumulative and full 250x250 matrices, normalized JS, regime schedule,
manifest and verified hashes. Conditional TV uses overlapping class mass and
is undefined without overlap. Global labels do not align disjoint tasks.
All 24 jobs use disjoint incoming windows, not repeated task shards. This
changes optimizer workload compared with old runs. FedAvg, vanilla ER,
5 local epochs, seed 42, k=3 except bootstrap, memory 500, batch/replay 13
remain unchanged. Clients retain observed windows at task boundaries, even
if not selected for optimization. No future windows enter training or probes.

Controller gets no experimenter profile, regime schedule or oracle outcome.
Client integrity metadata is restricted to its own shard/public hashes;
the extractor receives public probe integrity metadata only. This remains
a centralized simulation, not process isolation or formal privacy. Raw
exemplars do not enter protocol messages; checkpoints contain private data
and use restricted file permissions. Do not publish them.

Every result includes a self-contained ground_truth directory, full reports,
decisions, server aggregate constituents, peer distances, actual supplied,
trained and replay class counts, counterfactual branches and offline
ground_truth_comparison.json. All ten size-three subsets plus full-client
aggregation are evaluated at ten prespecified nonbootstrap rounds. Branches
are isolated and restore RNG. Cost is tracked separately; this is immediate
omission harm, not an alternative full trajectory. Distribution oracle
coverage is equal-client weighted; mixture mass uses supplied sample mass.
M2 sample-weighted distribution comparison is an analogue, not identical
to per-block report pooling. Cached-update comparisons use source rounds,
while actual selection coverage uses contemporaneous distributions.

The default launcher builds the ledger and dependencies; validation and
result hashes must match. Consolidated analysis refuses mixed streams and
produces matrix, round/regime, signal and counterfactual CSV files plus
heatmaps. Single seed, short regimes and carryover limit inference; no
independent full/random baseline or adaptive participation budget claim.

Restricted allocation_index.npz preserves source IDs, round, client, global
class and domain bin. Never include this file in public exports. Counterfactual
validation preserves full-parameter candidate update geometry and discrepancy
from full aggregation, and checks actual-branch validation metrics against
the real run. Per-round replay and conditional coverage, transitions and
separate communication/optimization costs are included in the audit.

For A06, offline reconstruction verifies the actual current/replay JS decision
score against the exact supplied and retained counts, including block-support
pooling for M2. For other signals, rank associations are exploratory. Shift
thresholds are prespecified in the stream JSON; retrospective same-client
report differences and selection delay are recorded separately from the actual
scheduler score. Delays are censored at the next shift or task boundary.
Full candidate update comparisons use current-round counterfactuals and
retrospective report pairs, not privileged admission information.

## Production preflight and isolated profiling

Run `bash run_all_sync.sh --stage data --resume --device cpu`, then
`python scripts/preflight_sync.py --output preflight_summary.json` to check
all 60,000 source pixel/label identities against the official torchvision
CIFAR-100 source and report realized regime statistics. The CI preflight
runs the same data stage without launching training. Cached generation and
validation dependencies recheck every shard fingerprint.

On a production GPU, use `python scripts/profile_sync.py --experiment A08-M3 --stop-after-round 13 --gpu-id 0 --output sync_profiles/a08-m3`.
Repeat for different signals/mechanisms or use `--experiment all` serially.
Profiles use the exact production configuration and a separate scratch output;
no production result is marked complete. They report per-round CUDA allocator
peak allocated/reserved bytes, process high-water RSS and elapsed time. Prefix
peaks are not full-trajectory or device-wide limits; account for CUDA context,
other processes, startup/evaluation peaks and safety margin when admitting jobs.
