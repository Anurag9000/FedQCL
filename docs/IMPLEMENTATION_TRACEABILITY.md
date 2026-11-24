# Implementation traceability and current limits

The authoritative specifications remain FEDQCL_CLIENT_SYNC_PROTOCOLS.md,
FEDQCL_SYNC_IMPLEMENTATION_SPEC.md, FEDQCL_COMMUNICATION_RESEARCH_PLAN.md,
FEDQCL_SYNC_EXPERIMENT_ROADMAP.md and FEDQCL_LOW_BANDWIDTH_SIGNALS.md.

"Implemented" below means code exists on the research path; it does not imply
empirical performance, integration-test success or production readiness.

| Requirement | Code | Verification/status |
|---|---|---|
| Legacy CLI default and mutual exclusion | system/main.py; sync/config.py | Implemented, legacy regression not run |
| Full bootstrap, due at round 5, empty round | sync/policies.py; sync/engine.py | Pure due/empty-roster tests pass; training integration pending |
| Local last-sync / previous / window references | sync/policies.py; sync/protocol.py | Last-sync and tie/drift tests pass |
| Fixed-count unknown priority | sync/policies.py | Unit test passes |
| Server delivered bank, task/age and self filter | sync/policies.py; sync/protocol.py | Undelivered-bank test passes |
| Bank mean, envelope, prior, latest, history | sync/policies.py | Implemented, variant tests pending |
| Bank piggyback/broadcast/telemetry | sync/protocol.py | Implemented, integration tests pending |
| Snapshot and delta catch-up | sync/protocol.py | Delta test passes |
| Peer direct unicast/relay; threshold/exact/greedy | sync/protocol.py; sync/policies.py | Chain, exact unknown, direct traffic tests pass |
| Peer accepted-coverage recomputation | sync/protocol.py | Unit test passes |
| Serialized logical bytes | sync/protocol.py; sync/engine.py | Failed-attempt test passes; model tensor bytes analytical |
| Hard byte cap | sync/budgets.py | Implemented, not integration tested; reliable pre-training only |
| Frozen server Task-IL evaluation | sync/engine.py | Implemented, not GPU tested |
| Native learning adapter | sync/engine.py; clientavg_tl.py | Implemented with explicit changes below |
| Separate refresh, local capture, ER, posttrain | sync/engine.py | Implemented, not training tested |
| Controlled allocation and disjoint streams | scripts/generate_streams.py; validate_streams.py | Implemented, real-data run pending |
| Report extractors | sync/signals.py | Histogram, replay count, fixed features, gradient/update CountSketch |
| Checkpoint/resume | sync/engine.py | Implemented, exact restore test pending |
| Same-checkpoint subset analysis | sync/counterfactual.py | One-round candidate reuse only; equivalence test pending |
| Campaign | scripts/build_job_ledger.py; run_campaign.py; run_all_sync.sh | Exact ledger generator executed locally before checkpoint flag addition |
| Paired pilot contrasts and budget matching | scripts/aggregate_results.py; match_budgets.py | Implemented, needs run artifacts |

## Explicit engineering decisions

The research adapter uses sequential client execution to avoid shared RNG races,
clones the CPU reference snapshot, uses a stateful replay cursor and resets it when
memory changes, and defaults to proportion-preserving sampling. These differ from
the original threaded training path. All research policies use the same adapter.
Native replay admission timing, stale rejoin, and legacy parameter aggregation remain
the primary setting. Complete state aggregation is an explicitly labeled ablation;
its weighting and buffer rules differ from legacy. The failed-download path retains
local trained parameters. The generated training split reserves 50 validation
examples per CIFAR-100 class. Default report minimum is 32. Histograms and other
reports use canonical JSON; tensor payload bytes are counted analytically. This is
not the proposed compact binary wire format or measured physical networking.
The simulator exposes a common availability/clock convention without extra traffic.
Private reports in local offline diagnostic artifacts must not be treated as delivered
network reports or released as public data.

## Not yet complete

The previous master plan also includes true decentralized/fault-tolerant peer
delivery, sparse peers, central cached-coverage and aggregate-update matching,
combined policies, uncertainty-aware learned impact selection, full advanced
signal/distance catalog, stronger data similarity optimization/feasibility guarantees,
rate-matched random schedules, multi-round counterfactual branches, explicit branch
equivalence tests, comprehensive integration/regression and resume tests, larger
cohorts and external baselines, and held-out confirmatory campaigns. These are not
implemented completely or verified by the current changes. See the experiment
ledger for the actual executable job matrix. No statistical superiority or physical
network savings have been established.
