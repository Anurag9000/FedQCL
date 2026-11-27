# Implementation specification for the three synchronization methods

This is the proposed first implementation for the five-client CIFAR-100 simulator. The [protocol design](FEDQCL_CLIENT_SYNC_PROTOCOLS.md) describes the research variants and the [experiment roadmap](FEDQCL_SYNC_EXPERIMENT_ROADMAP.md) defines how to evaluate them. The defaults below are starting settings for a pilot, not tuned values.

## Shared inputs and state

| Item | First implementation |
| --- | --- |
| Decision statistic | Counts over the 20 current-task labels, from up to 128 examples in a reproducibly sampled, currently revealed local observation window |
| Distance | Total variation on normalized counts, `d(p,q) = 0.5 * sum(abs(p-q))`, between 0 and 1 |
| Example thresholds | `0.05, 0.10, 0.20, 0.35` on development streams. A distance equal to the threshold counts as covered. Use independent thresholds for the three methods. |
| Cold start | Full participation in round 0. Include those transfers in every policy's budget. |
| Staleness guard | Default at most four consecutive skipped rounds for a continuously available client, where the next round is due. Make the limit a shared sweep parameter. |
| Report validity | Require correct schema, current decision round for peer reports, valid task and label order, positive count sum, and compatible reference space. Require a configurable minimum window for normal threshold decisions. |
| Experimental workload | Use the native stale-model rejoin and replay timing in the initial controlled comparison. Record skipped task-end actions and test a separately labeled local memory-capture variant. |

At each round reveal only observations permitted by the stream definition. In the existing static-shard setting, rounds expose a task at a time. Reusing the same task histogram reveals no real within-task drift. Add an explicitly scheduled sequence of within-task windows when studying that hypothesis, and record both population and sampled histograms. Use independent observation RNG so a report never consumes training sampler draws. For model-dependent statistics added later, require a versioned extractor and probe definition before treating reports as comparable.

Each client stores its previous valid observation, its last successfully synchronized statistic, its last complete sync round, the version of the model it holds, and the latest server bank it actually received. The server stores the accepted roster per round, global model version, and a record of report arrivals. A bank record identifies the source client, task, observation round and window, statistic kind and values, and whether the associated model contributed. The common decision record includes candidate IDs, scores, reference versions, reasons, the planned roster, estimated cost, and actual outcomes.

### Proposed integration points

- Add the policy and configuration at the existing CLI entry point in [main.py](../system/main.py). Default to the existing behavior. Make research policies and legacy hidden-count overrides mutually exclusive. Reject invalid thresholds, client budgets, or report windows at startup.
- Add interfaces for gathering statistics, deciding eligibility and admission, and recording communication under a dedicated module in `system/flcore`. Policies operate on small immutable reports and round context. They do not read model tensors, other clients' datasets, or held-out evaluation data. A transport abstraction handles logical deliveries in the simulator.
- Call the research scheduler from [the FedAvg round loop](../system/flcore/servers/serveravg.py) before creating client training threads. The legacy selector has a fallback that can select all clients, so the explicit full-participation baseline and new policies need their own rosters.
- Pass the accepted model roster and explicit download recipients through [the server aggregation path](../system/flcore/servers/serverbase.py). The primary research path sends the aggregate to accepted contributors. A client whose upload was filtered still incurs the attempted upload cost if its payload arrived. Record the outcome separately from a policy skip. Add a frozen-server evaluator with explicit round and task IDs.

## Round sequence

1. Freeze round ID, current task, available client roster, budget, and starting model versions. Reveal the round's observation windows.
2. Extract reports without running full local optimization. Apply only the statistics deliveries allowed by the policy. Finish this phase before any reported value affects admission.
3. Determine which clients are due from their last complete sync round. With a last complete sync at round 0 and a limit of four, a continuously available client may skip rounds 1–4 and is due in round 5. Put due clients first. Methods 1 and 2 admit the remaining requests in threshold runs and fill remaining slots by descending novelty in fixed-count runs. Method 3 uses the representative rule below. Apply oldest first and client ID for novelty ties. Log any fixed-count admission that the threshold would have deferred.
4. Train admitted clients from their held model versions, collect their uploads, and aggregate successful accepted contributions. Capture worker exceptions explicitly. An empty accepted roster carries the server model and version forward, avoiding the existing nonempty aggregation assertion. A bootstrap failure is a run integrity failure.
5. Deliver the aggregate to accepted contributors, apply any policy-specific report distribution, and evaluate the frozen server model. Update last-complete-sync state only for clients that both contributed and received the aggregate. A failed receive leaves that client's model and sync reference stale. If a hard cap prevents visiting every due client, record a deadline violation and use oldest-first precedence. Evaluate cap and deadline feasibility under each tested schedule.

Keep evaluation and observation separate. A failed or absent report is an unknown, not evidence for suppression. Put unknowns ahead of optional covered clients in the fixed-count variant, after mandatory visits. In threshold runs request their participation and log the reason. A checkpoint commits at the end of a round and includes policy state, reference versions, counters, model and optimizer state, replay contents and iterator state, and RNG state needed for a reproducible resume.

## Method 1. Client compares with its own last full sync

1. Compute the current normalized histogram on the client. Retrieve the last-sync histogram from the same task and schema. An absent or incompatible reference generates a novelty request.
2. Otherwise compute `d(current, last_sync)` locally. Request a full sync above `tau_local`, or defer at or below it unless the age rule applies. In the fixed-count experiment, transmit a scalar score and eligibility reason to the simulated arbiter as needed, and charge that control traffic. The raw histogram stays local.
3. Write the observation into the local timeline whether or not the client synced. Replace the last-sync histogram only on completion of a two-way exchange. A previous-observation variant compares to the saved previous observation before replacing it. A windowed variant compares to a rolling reference computed from prior observations.

Example with `tau_local=0.10`: a score of 0.08 defers, 0.10 defers, and 0.11 requests. Observations each 0.04 apart can still trigger by accumulating more than 0.10 difference from the last successfully synchronized statistic. No report bytes are needed for the threshold decision itself. Admission messages are counted if the transport sends them.

## Method 2. Client compares with a shared server bank

1. During bootstrap each client sends its report with the upload. After collection the server writes an ordered bank with one latest valid report per source and task. It distributes the bank to recipients alongside the aggregate. Use a monotonic bank version. A client with version 3 cannot use version 4 until it receives it.
2. Before training on round `r`, client `i` reads its own cached bank. For the first run compare to other clients' same-task records only, with age at most five rounds. Compute the minimum distance to an eligible record. With no eligible reference, request a full sync and record cold or stale reference. Otherwise request if the minimum exceeds `tau_server`. Preserve the closest record ID and age as decision evidence.
3. Selected contributors send new reports with their model uploads. The server updates the bank for valid arrivals and records which model uploads it accepted. For the primary bank include only reports associated with accepted model uploads. Deliver full snapshots in the first experiment, to aggregate recipients only. Clients that defer retain their old bank. In the separate statistics-only mode, ingest and mark reports even if their weights are omitted. All eligible clients can receive a new bank. Charge both legs of that phase and exclude each client's own record from its comparison.

For the same-task example, distances of 0.08 and 0.30 give a nearest score of 0.08 and defer at `tau_server=0.10` unless due. In the first round of a new task, an old-task record cannot suppress selection. Add a bounded past-round bank, a sample-weighted centroid, and self-inclusive matching as separate ablations. A logical snapshot event carries bank version, records, and recipient set. If a later implementation sends deltas, a client applies one only to the advertised base version. A missed delta requires a paid catch-up or snapshot.

## Method 3. Clients elect representatives from current reports

1. Each eligible client prepares the current round's report. Use an agreed roster and deadline. Every deciding peer sees the same ordered set of valid reports and the same age information before election. Start with an in-process emulation of reliable peer unicast. Charge `N(N-1)` report deliveries for a complete exchange and account for any roster or election messages. Label a server relay as a transport variant, and a centralized selector as a separate decision-placement variant.
2. Seed the representative set with due clients, ordered by oldest sync then client ID. For the threshold variant, walk the rest in that order. Add a client if its minimum distance to the selected set exceeds `tau_peer`, or if it has no valid comparison. If the selected set is empty, admit the first eligible client. Assign each suppressed client to its closest admitted representative, using client ID to break ties. Similarity is checked against a representative directly, so a chain of similar clients cannot suppress a distant endpoint.
3. For a fixed cap, enumerate feasible subsets of size `k` for five clients, including due clients so far as feasible. Minimize the sum of distances from each eligible report to its closest selected report, using equal client weights in the first run. An unselected unknown receives a fixed penalty greater than the maximum total distance across valid reports so known records cannot mask unknowns. Break ties by oldest sync and then client ID. Log clients beyond the threshold as uncovered. Train and sync the elected roster only. If an elected upload fails, record its clients as uncovered unless another accepted client covers them. Fallback recruitment is a separate variant with charged messages and training.

For example, if A covers B, B covers C, and A does not cover C, admitting A first suppresses B and can still admit C. If A fails to upload, B's coverage is recomputed against accepted representatives. If realistic peer delivery yields different rosters, do not silently emulate agreement. Use an explicit common-roster handshake or abort the election and run a predefined fallback.

## Acceptance checks before training campaigns

- Hand-check deterministic traces for exact threshold ties, gradual local drift, stale and missing bank entries, same-round report ordering, three-client similarity chains, capped overdue clients, empty rounds, failed uploads and downloads, and restart from a committed round.
- Check that an omitted client receives no model and contributes no model, that the server evaluation reads one model version, and that a replay boundary miss is visible in logs. Count all reports as serialized messages even when a simulator can access their values directly.
- Verify history scope by adding a report the protocol has not delivered and confirming that it cannot affect the current decision. Compare all policies first on the same reports, then add richer signals independently.
- Freeze pilot settings, preserve per-event bytes and the task-by-round accuracy matrix, and use the matched-budget results in the [experiment roadmap](FEDQCL_SYNC_EXPERIMENT_ROADMAP.md) before claiming any accuracy or bandwidth gain.
