# Active goal override

Only the 27 signal × mechanism configurations in
configs/sync/core_signal_ledger.jsonl are in the active default campaign.
Historical expanded and Task-IL catalogs below are optional and out of scope.
See EXPERIMENT_LEDGER.md for the updated exact signal definitions and caveats.

# Campaign scope update

The default campaign now selects a prespecified economical subset: 50 sync and
22 Task-IL training entries. The historical full research catalog is preserved
but is not the default execution scope. All algorithmic limitations and
scientific caveats below remain relevant to applicable configurations. The core
subset does not constitute exhaustive empirical validation of optional features.

# Current learning contract

The research backbone is strictly `fedavg_uniform_er_v1` across both suites.
The historical FEDQCL client remains isolated behind legacy execution only.
All queue-related and alternative learning-backbone jobs are excluded. The
shared backbone.py implements direct weighted FedAvg and one pooled replay loss.
The 48-run Task-IL suite no longer uses FedBuff or stores base model snapshots.
Earlier verification and counts below refer to pre-conversion code; consult CI
for the current revision. Remaining non-backbone scientific limitations are not
automatically resolved by this migration.

## Historical traceability (pre-conversion)

Authority: FEDQCL_CLIENT_SYNC_PROTOCOLS.md, FEDQCL_SYNC_IMPLEMENTATION_SPEC.md,
FEDQCL_COMMUNICATION_RESEARCH_PLAN.md, FEDQCL_SYNC_EXPERIMENT_ROADMAP.md and
FEDQCL_LOW_BANDWIDTH_SIGNALS.md. See EXPERIMENT_LEDGER.md for the verified ledger.

| Requirement | Code | Status |
|---|---|---|
| Legacy default, task/reference correctness, queue sample-mean ablation, V-normalized LR, aggregation weights | main.py; clientavg_tl.py; sync/engine.py; config.py | Implemented; native behavior retained, full legacy regression not established |
| Own/server/peer selection, due/bootstrap rules, bank views, delta and delivery ordering | sync/policies.py; protocol.py; engine.py | Implemented; unit and synthetic smoke coverage |
| Cached/weighted facility location and FedAvg-weighted update matching | sync/advanced.py | Implemented; sketches are approximations |
| Combined fixed/threshold and joint novelty/coverage objectives | sync/advanced.py; protocol.py | Implemented |
| Delivered-history staleness, dynamic threshold and block sampling | sync/protocol.py; advanced.py | Implemented |
| Adaptive linear/quadratic impact, feature guards, joint seed calibration | sync/impact.py; scripts/fit_impact.py | Implemented; no fitted empirical models |
| Budget and future reservation | sync/budgets.py | Reliable pre-training hard caps and necessary future-transfer lower bound; no optimal global planner |
| Binary/JSON/FP16/INT8 and quantization diagnostics | sync/wire.py | Implemented; multichannel quantization excluded |
| Count/byte-matched controls and achieved-gap assessment | scripts/extract_count_schedule.py; compare_matched_bytes.py; ledger | Implemented; exact equality not guaranteed |
| Frozen continual evaluation, class counts, metrics and checkpoints | sync/engine.py; metrics.py | Implemented; synthetic restart equivalence tested |
| Local memory capture, refresh, ER, posttraining, replay profiles | sync/engine.py; replay.py | Implemented; new replay profiles bounded and causally local |
| CIFAR-10/100/TinyImageNet, drift, unequal loads, domains, integrity | scripts/generate_streams.py; validate_streams.py; sync/engine.py | Implemented; TinyImageNet requires extracted files; full real-data checks pending |
| 30 signal types, including separate replay gradients and explicit channel masks | sync/signals.py; multichannel.py; extra_signals.py | Implemented; estimator approximations and computational costs explicit |
| Discrete MI, CMI, TC, interaction information, transfer entropy | sync/information.py; extra_signals.py | Plug-in estimators, finite-sample bias remains |
| Paired synthetic anchor CCA/SVCCA/PWCCA-style, CKA, RSA, HSIC, Procrustes, angles/neighborhood | sync/representation.py | Rank-limited estimators with public synthetic anchor, not exact real-data representational agreement |
| Bayesian, Fisher, Jacobian, Hessian, conditional moments | sync/extra_signals.py | Diagonal Laplace head and bounded derivative probes; not exact full-network inference |
| Counterfactual subset enumeration, interactions, oracle regret and continuation | sync/counterfactual.py; continuation.py; scripts/analyze_counterfactuals.py | Implemented; higher-order interactions, multiple continuation policies, false-safe/expensive diagnostics |
| Sparse/distributed peer election, dynamic topology and message faults | sync/peer_network.py; distributed_peer.py; message_integrity.py | Simulated gossip and reliable retransmission; not Byzantine consensus |
| Control accounting, randomized response, loopback benchmark | sync/protocol.py; scripts/benchmark_transport.py | Implemented; no end-to-end DP or physical WAN claim |
| Stratified sample-mass and gradient-sketch facility comparators | sync/protocol.py; advanced.py | Literature-inspired adaptations, not official Clustered Sampling/DivFL reproduction |
| Ledger, independent splits, power planning, parallel runner and postprocessing | scripts/build_job_ledger.py; run_campaign.py; plan_confirmation.py | 4,537 default jobs generated in CI, including 3,885 training configurations |

## Explicit engineering choices and boundaries

Native defaults remain available for comparability. The research engine uses sequential local clients within one training job, cloned historical snapshots, stateful replay cursors, and explicit phase ordering. Parallel execution is across independent jobs on a single host. A failed download retains the actual locally trained parameters. Model payload counts are analytic; tensor transport over separate physical machines is not implemented. TCP loopback is an application-level benchmark only. Seeded randomized response is a reproducible ablation, not a formal privacy guarantee for model updates or repeated releases.

Strict calibration uses 5 fit, 20 calibration and 5 held-out partition seeds. Grouped maximum residuals across current/past targets yield a joint bound only under exchangeability. No calibration model or power estimate has been fitted from actual research training. Development variance must determine final independent replication count. Multichannel fusion has fixed declared channel weights; learned nonlinear harm prediction is separate from learned channel-metric weights.

## External method reproduction

See EXTERNAL_BASELINES.md. Stratified mass sampling and gradient facility sketches are explicitly named adaptations. Uniform, class-balanced and rare-class replay are local comparators, not Re-Fed. FedWeIT's decomposition and cross-client knowledge transfer require a distinct model/training protocol; no fake FedWeIT job is included.

## Outstanding work distinct from full training

- Faithful official Clustered Sampling, DivFL, FedWeIT and Re-Fed reproduction and a harmonized task/data/evaluation adapter, if required for paper claims.
- True separate-host transport measurement, formal privacy guarantees, stronger adversarial consensus and globally optimal budget scheduling.
- Exact full-network Bayesian inference and architecture-specific attention/gating signals are not implemented; the ResNet-18 path has no attention module.
- Arbitrary replay-target similarity realization, learned channel weighting, exhaustive branch/fault integration, full legacy regression and GPU determinism checks remain unverified.
- Study design still requires development outcomes, adequate power and statistical analysis. No empirical superiority is claimed.

The protocol suite at 14a3bbd9 passed 48 tests (2 opt-in skipped); the earlier full CI at 2c7a7378 passed both synthetic neural tests. Check the latest workflow for new neural smoke coverage. Full research training remains unexecuted.
