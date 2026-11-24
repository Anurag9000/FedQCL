# Implementation traceability and current limits

Authority: FEDQCL_CLIENT_SYNC_PROTOCOLS.md, FEDQCL_SYNC_IMPLEMENTATION_SPEC.md,
FEDQCL_COMMUNICATION_RESEARCH_PLAN.md, FEDQCL_SYNC_EXPERIMENT_ROADMAP.md,
FEDQCL_LOW_BANDWIDTH_SIGNALS.md. See EXPERIMENT_LEDGER.md for the job matrix.

| Requirement | Code | Status |
|---|---|---|
| Legacy default; mutual exclusion | main.py; sync/config.py | Implemented; legacy regression pending |
| Bootstrap, due rules, empty rounds | sync/policies.py; sync/engine.py | Implemented; core unit tests pass |
| Own history references | sync/policies.py; sync/protocol.py | Implemented; core unit tests pass |
| Delivered server bank, ages/task, views, delta | sync/policies.py; sync/protocol.py | Implemented; core unit tests pass |
| Direct/relay peer threshold/exact/greedy | sync/policies.py; sync/protocol.py | Implemented; core unit tests pass |
| Cached age-aware facility location | sync/advanced.py | Implemented; pure tests pass |
| Cached actual-update sketch mean matching | sync/advanced.py | Implemented; pure tests pass; equal client weights |
| Combined 12/13/23/123 | sync/advanced.py; sync/protocol.py | Implemented; fixed and threshold variants; not all neural tests run |
| Impact model, calibrated upper residual, adaptive selection | sync/impact.py; scripts/fit_impact.py | Implemented; exploratory calibration only; no fitted model yet |
| Compact binary reports | sync/wire.py; sync/protocol.py | Implemented; round-trip tests pass |
| Hard logical byte cap | sync/budgets.py | Implemented for reliable pretraining; adaptive hard bytes pending |
| Frozen server and continual metrics | sync/engine.py; sync/metrics.py | Implemented; earlier CPU training passed |
| Local capture, refresh, ER, posttrain | sync/engine.py | Implemented; dedicated variant integration pending |
| Allocation, similarity, windows | scripts/generate_streams.py | Exact class inventory; approximate similarity optimization; real CIFAR run pending |
| Histogram, replay, feature, gradient/update | sync/signals.py | Implemented |
| Replay+histogram, queue, prototypes, covariance, spectrum, loss, predictions, confidence, entropy, norms, normalization, Jacobian and empirical head Fisher, gradient variance | sync/signals.py; sync/statistics.py | Implemented; expanded extractor smoke pending |
| TV, JS, Hellinger, cosine, bounded L2 | sync/policies.py; sync/statistics.py | Implemented; some standalone math metrics |
| Exhaustive subset enumeration, omission interactions, regret/opportunity | sync/counterfactual.py; scripts/analyze_counterfactuals.py | Implemented; real-data study pending |
| Cross-task continuation and explicit branch equivalence | sync/continuation.py; sync/counterfactual.py | Implemented; expensive branch tests pending |
| Sparse peer graph and reconciliation | sync/peer_network.py; sync/protocol.py | Implemented; sparse pure test, network integration pending |
| Failures, availability, charged relay fallback | sync/engine.py; sync/protocol.py | Implemented; robustness integration pending |
| Rate-matched counts | scripts/extract_count_schedule.py; sync/protocol.py | Implemented; offline |
| Checkpoint/resume | sync/engine.py | Earlier two-task CPU deterministic restore passed |
| Job ledger and provenance | scripts/build_job_ledger.py; run_campaign.py | Implemented; current expanded matrix in CI |
| Paired summaries and frontier plots | scripts/aggregate_results.py; scripts/plot_results.py | Implemented; requires experimental results |

## Explicit engineering choices

Research execution is sequential, uses cloned historical snapshots, stateful replay
cursors, proportion-preserving sampling and explicit native replay boundary behavior.
The legacy path stays default. Failed downloads retain actual locally trained
parameters. Binary reports use uint32 counts or float64 real vectors; the latter
preserves exact transmitted scalar values but is not FP16 compression. Model bytes
are analytic and common clock/availability traffic is free by declared convention.
Private diagnostic reports are not public artifacts and must not be interpreted as
delivered policy evidence.

Cached update matching uses equal-client-weight cached sketches in declared native
parameter coordinates; it is not an oracle or a guarantee of matching the current
aggregate. Combined methods use union/intersection for novelty and sequential peer
filtering. Sparse graph is a centralized arbiter over delivered edge distances,
not a distributed-consensus implementation. Reconciliation assumes a reliable
fallback relay. Impact calibration is preliminary and low-confidence (0.5) because
only two independent calibration seeds are enumerated.

Development validation reserves 50 examples per CIFAR-100 class; held-out test is
used in adaptive seeds 45-47 only in the generated ledger. Similarity target
realization is approximate; integer allocation solves an exact minimum absolute
count discrepancy problem and explicitly verifies class/client totals. Training
class counts are recorded from actual SGD batches.

## Remaining work

- Full real CIFAR-100/GPU training smoke, legacy regression and complete campaign.
- Exhaustive integration/leakage/branch equivalence and deterministic GPU tests.
- Full advanced catalog: Bayesian posterior inference, MI/conditional MI, transfer
  entropy, CCA/SVCCA/PWCCA, richer paired-anchor representations and other estimators.
  Standalone linear CKA does not constitute a deployable paired-probe protocol.
- Weighted and genuinely multichannel cached coverage/update matching, weighted
  peer objectives, separate current/replay gradient decomposition and learned
  impact features beyond the first cached histogram model.
- Future-budget reserve, rate-matched byte rather than count comparisons,
  independently calibrated confirmatory thresholds, larger independent calibration
  samples and statistical power analysis.
- External baseline reproduction, measured physical networking, additional
  transport failure modes, truly distributed sparse election and privacy analysis.
- Additional allocation/window profiles and full per-message phase order tests;
  report ingestion currently occurs at completion in the simulator's reliable path.

No empirical accuracy, forgetting or physical communication superiority is claimed.
