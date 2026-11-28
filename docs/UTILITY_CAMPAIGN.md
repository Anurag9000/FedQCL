# Utility follow-up (opt-in)

The archived core campaign (123 completed trajectories, 179 workflow jobs)
is not overwritten. This independent suite uses its validation-calibrated
learning rate 0.01 and original finalists A03-M1, A03-M2, A05-M3. New
results live under research_results/utility-<commit>. No archive model is
required. Data jobs reuse the same content-addressed controlled-stream
generator and independent source validation.

## Run
```bash
python scripts/build_job_ledger.py --suite utility
python scripts/run_campaign.py --ledger experiments/job_ledger_utility.jsonl --estimate-only
SYNC_SUITE=utility bash run_all_sync.sh --all --stage development
SYNC_SUITE=utility bash run_all_sync.sh --all
```

Fresh-start mode downloads the current server model immediately before
training only admitted clients, then uploads their dense model. There is no
post-aggregation model delivery. The initial provisioning is still charged
equally; contribution completion and held model version are different states.
FedAvg sample weights, SGD, replay and local memory capture are unchanged.
The isolated counterfactual/continuation engine uses the same fresh bases.

M1 utility sends a scalar local urgency and risk estimate; M2 collects
all available compact telemetry centrally, M3 sends it directly to all peers
and reconciles deterministic elections. New reports carry per-task batch
loss, signed fixed-memory deterioration, clipped CountSketch gradients,
support and model version; M2/M3 additionally use supplied label mass.
Model-dependent probes use cached local models, not free server access.
FP32/FP16/INT8 are actual binary encodings; statistics and controls are
charged per delivered recipient. Unknown evidence and heuristic confidence
fallback are explicitly recorded. The score is an analytical proxy,
not a trained or calibrated predictor. Per-task historical constraints are
predicted and relax visibly when infeasible. No raw private examples or
ground-truth distributions enter selection.

The fixed-count development ablation compares random, balanced, loss-only,
local, server, peer, full and fresh/stale archive selectors. Prospective
40/60% caps are fractions of the dense full-exchange *body* reference
(including initialization), not measured full baseline bytes. All bytes
including initialization and telemetry are debited. A conservative future
deadline schedule plus per-round overhead bound is reserved before selection.
Random and utility receive the same ceiling, not necessarily the same actual
bytes. Each adaptive statistical run has an exact-count random control.
Actual bytes, sample counts, validation learning curves and final held-out
results are recorded. Sensitivity runs vary mass, gradient matching, fairness,
retention, uncertainty, precision, sketch dimension, deadline and horizon.
Prespecified independent seed and setting confirmations and a 10-client
scale check are included. The primary claims are empirical, not guaranteed.

Limitations: this revision does not fit an archive-trained predictor directly (old reports lack the new causal features), measure physical networks, or implement trained shared proxy refresh. The
analytical risk penalty is not a calibrated confidence interval. The hard
cap reserve is intentionally conservative. The utility probes do not modify
local training but cost additional compute; local samples may differ at
equal participant counts. The exact subset search is limited to ten clients.
The archive excludes model checkpoints and source datasets; regenerated
streams must pass independent verification. No production results are
asserted by the ledger.

The new M2 calibration job fits a frozen six-feature ridge model on three S14 seed-42 analytical development trajectories and qualifies it using three separate S04 trajectories. It consumes validation counterfactual outcomes only, never held-out test metrics. Six seed-43 learned-policy runs depend on the model receipt; if the held-setting ranking fails qualification, the learned policy falls back to randomized selection. The empirical residual is not a statistical confidence guarantee. The original archive supplies LR and finalist identities, not new-probe training features.

The separate delta ablation applies sample-weighted floating-state displacements to the latest server state, with nonnegative running variances and max integer counters; dense upload size is unchanged. It is not vanilla FedAvg. The common classifier-head proxy is a zero-logit head on deterministic pooled image features, not a learned global reference; it is a causal diagnostic for common coordinates. Adaptive precision uses only the previous decision reason, requests INT8/256 dimensions after a confident utility decision and otherwise the default FP32/1024; request messages are charged. These are single-factor sensitivity studies, not proven improvements.

The offline archive audit reads all 123 completed matrix entries and their archived validation counterfactuals. It reports observed same-count regret, uniform expected gains only when the same-count subset inventory is complete, exact Shapley contributions only for complete powersets, and uniformly averaged pairwise coalition interactions. No training or fitted feature reuse is performed by this audit. Peer utility context (prior bytes, visit counts, ages, version, model size and cap) is explicitly sent and charged. Pareto membership is reported separately per setting and seed using validation accuracy, forgetting and actual bytes; target-accuracy byte crossings are observed, not interpolated.

## Expanded matched OFF and represented-mass study

The expanded utility suite spans all five-client OFF counts in S04 and S14 and all ten-client OFF counts in S14N10, subject to the archive-informed pruning below. Fresh local/server/peer utility and represented-mass server/peer variants also span the full ranges. Full runs are explicit controls; all clients bootstrap at round zero. The original budget, stale/delta and learned jobs remain. In the paired CSV, fixed_matched means the same setting, seed, client count, OFF count, deadline and initialization/delta protocol (represented weighting is the experimental treatment). Actual samples, steps and replay equality are recorded, not presumed. Exact-count and common-ceiling adaptive comparisons are separate; full/cross-protocol contrasts are descriptive. Fixed k may force deadline violations when mathematically infeasible.

The opt-in --sync-representative-beta (0 to 1) transfers a fraction of omitted current sample mass to a selected representative only when current label and every available current/historical gradient channel meet a discrepancy gate, with identical task/round and native-model version (fixed common proxy permits different versions). Unknown or incompatible evidence is unrepresented, never invented. Selected weights are normalized by FedAvg; this does not multiply the overall update magnitude. All transfer decisions and realized weights are logged and reconstructed from delivered telemetry. Peer elections carry their plan and are charged. Counterfactual branches use the same rule. The feature requires fresh server/peer telemetry and one-round counterfactuals. It is not a guarantee that two different datasets are interchangeable.

Twelve prespecified tuning profiles use S04/S14 seed 42 validation accuracy gain and forgetting for ranking. Seed 43/44 runs are independent and precomputed for every profile; tuning_selection.csv marks the development-selected profile and reports its confirmation contrasts. Test metrics do not determine the winner. This is a bounded profile search, not exhaustive hyperparameter optimization. Existing seed-43 learned-policy confirmations remain separate. Larger gains are hypotheses, not guaranteed outcomes; compare only matched controls, including additional report/compute costs.

Inventory: 503 training trajectories, 22 data generation/validation jobs and 3 analysis/calibration jobs = 528 total jobs. Ten-client mid-size coalition counterfactuals can be expensive; this is a deliberate full-budget expansion. The archived core and Task-IL ledgers are not changed.

### Archive-informed pruning (before any new training)

Remove exactly 31 low-priority cases from the initial 534-trajectory proposal: A02-M1, A04-M2 and A08-M2 at k=1..4 in both S14 and S04 (24); A03-M1 at k=1..4 in S04 (4, including the fresh-start k=3 case); and the three original finalists at k=6 in S14N10 (3). These are subjective priorities from the previous archive, not measured failures of the new protocol. All full-participation and random controls, new utility and representative-mass variants, tuning and independent confirmation cases remain. This pruned matrix is not a complete factorial comparison of all 24 original methods. Original archived results and shortlist calibration contract remain intact. The removal changes fixed-stage training from 331 to 301, and aggregation-stage training from 14 to 13; no dataset or analysis jobs are removed.
