> **Current executable campaign:** [101-run compact single-seed design](CONTROLLED_CAMPAIGN.md). This document preserves historical methodology and optional extensions; where counts, default suite names, or execution instructions differ, the compact campaign specification and generated ledger are authoritative. All 24 datasets remain independently validated; only six receive full neural trajectories. The former 35,208-run matrix is opt-in `exhaustive`.

# Client decisions for selective synchronization

The [first implementation specification](FEDQCL_SYNC_IMPLEMENTATION_SPEC.md) fixes the round sequence, state, message formats, policy rules, and expected behavior for all three methods.

## Objective and common round contract

Reduce the bandwidth spent exchanging full model weights while minimizing the drop in final server accuracy and any increase in forgetting relative to full participation. A **full sync** consists of an upload of the client's locally trained model followed by a download of the server aggregate. A client that skips the full sync sends neither model payload. It may still send or receive a compact statistic in protocols that explicitly allow it. A **silent round** has no client traffic beyond whatever common control-plane convention the experiment declares and charges.

Let `s[i,r]` be a compact statistic computed by client `i` from its observations available at the decision point in round `r`. It can later be instantiated as a histogram, representation summary, replay statistic, Jacobian sketch, or another bounded report. Let `d(x,y)` measure dissimilarity, with larger values meaning more difference, and let `tau` be a threshold calibrated on development streams. A report carries enough context for comparison, such as its task, observation window, model version when applicable, and schema version. The [signal catalog](FEDQCL_LOW_BANDWIDTH_SIGNALS.md) defines candidate statistics and encoding costs.

The decision must precede the model upload. In the first experiment, each client computes its statistic from a local observation window or a small probe, then only selected clients run the full local optimizer. This follows the repo's skipped-training convention while giving all clients a way to observe new data. For statistics available only after training, run a separate variant in which clients compute candidate updates before deciding which to upload. Charge the local computation in both variants. Reports extracted from future observations or from weights that would have been uploaded only under another decision are reserved for offline analysis.

Once admitted, clients upload their weights, the server aggregates the accepted models using the declared FedAvg rules, and the participating clients receive the resulting aggregate. Under native rejoin behavior they started the round from the last model they received. Any pre-training refresh adds a download and defines a separate experimental variant.

## Method 1. Compare with the client's own history

**Question:** Has this client's stream changed enough to warrant another full sync?

1. Each client retains its own statistics locally. Store both the latest observation and the statistic associated with its last successful full sync. A bounded local history can also support a windowed reference.
2. Before uploading weights, the client compares its current statistic with a chosen local reference. The primary rule requests a full sync if `d(s[i,r], reference[i]) > tau_local`. With an eligible comparison and a distance at or below the threshold, it withholds both model transfers.
3. Advance the observation history every time the client observes data. Advance the last-sync reference only after a successful full sync. The first eligible participation initializes that reference.

Compare **previous observation**, **last successful sync**, and **recent window** as distinct reference choices. Consecutive observations can each look similar even while the stream gradually moves away from what the server last received. Keeping the last-sync reference makes that cumulative change visible. The decision statistic stays on the device in this method. Charge any admission request or acknowledgment the network protocol uses.

This method detects change over time for each client. It can schedule several clients that all underwent the same change, and a stable client can still carry useful updates for an underrepresented class or replay task. Measure those cases explicitly.

## Method 2. Compare with statistics returned by the server

**Question:** Is this client's current statistic already represented in information previously shared with the federation?

1. In the bootstrap exchange, each participating client sends its individual statistic alongside its model. The server retains the received records with client, task, and round IDs. It distributes a versioned copy or agreed summary of those records alongside the averaged weights, or through a separate compact broadcast.
2. Each client stores the latest reference it received. On a later round it compares its current statistic against that cached server reference before deciding whether to request a full sync. For a reference bank `R[i,r]` available to client `i`, a simple novelty score is `min(d(s[i,r], x) for x in R[i,r])`. Request a full sync above `tau_server`. A statistic close to eligible prior evidence can suppress the full sync.
3. The server adds newly received statistics to the bank and distributes the next version. A full sync updates the client model and the bank version it has received. A lightweight broadcast can update the bank independently of the client model.

Evaluate these references independently:

| Reference | What the client compares |
| --- | --- |
| Aggregate summary | A mean or weighted mean. Coordinate minima and maxima can define a reference envelope when coordinates have compatible meanings. |
| Prior round | The individual records received in the last reference round |
| Latest per client | One most recent record per client, with age and task labels |
| History | Individual records from several prior rounds, with configurable age or task filtering, deduplication, or representative prototypes |

Comparing to an average and comparing to the nearest individual answer different questions. An average can sit far from any actual client or hide small groups. Report separate results for both. Compare banks that include the client's own history with banks that exclude it, to isolate the added value of knowing what other clients shared.

Use two delivery policies. With **piggybacked reports**, participants send their statistics alongside full weights. The server can return the bank alongside the model, while a skipped client retains its older reference. With **separate telemetry**, the server can send the compact reference to all clients, including those skipping the model. A statistics-only exchange can also collect client reports and disseminate a bank without aggregating models. Record whether a reference represents statistics observed by the server or only contributors whose models were incorporated. These sets can differ. A decision may use the most recent bank the client actually received, including a just-completed statistics exchange when that phase is enabled and charged.

Communicating a history is plausibly cheaper than sending model weights, but size depends on recipients, message frequency, and retention. For `N` clients, `H` rounds, and `S` bytes per record, a dense history can occupy `NHS` bytes before metadata. Sending that full history separately to all `N` recipients costs up to `N²HS` delivered downlink bytes per refresh. Compare a bounded history with delta updates and version-based catch-up. Keep the archive on the server if replaying its full contents to clients exceeds the budget.

## Method 3. Elect representatives using current peer statistics

**Question:** Which clients have distinct information this round, and which can be covered by a peer?

1. Eligible clients compute their current statistics at the same decision stage and exchange compact reports. The direct protocol sends them peer to peer. A server relay can distribute the same reports when direct connections are unavailable. Account for the two transport variants separately.
2. Compare current reports with a common metric and threshold `tau_peer`. Choose a deterministic order, for example oldest last sync first, then client ID. Traverse the reports in that order. Select a client as a representative if it is farther than the threshold from all representatives already selected. Otherwise assign it to the closest covering representative and suppress its model exchange.
3. Representatives send full model weights to the server. The server aggregates their accepted uploads and returns the aggregate to the participants. Suppressed clients have exchanged only statistics and control messages. Record the representative that covered each suppressed client.

The direct rule ensures every suppressed client is within the threshold of a selected representative when there is no hard cap. Avoid grouping all clients through chains of pairwise similarity. If A is close to B and B is close to C, A can still be far from C. For budgeted rounds, choose a subset that minimizes uncovered clients or total distance to their closest representative, give overdue clients priority under the same eligibility rules, and log any uncovered clients or threshold violations. Reserve coverage of an apparently distinct stream when the budget permits.

Keep a consistent round ID and report roster so clients elect the same representatives. Use a defined collection deadline. A missing report leaves coverage unknown and can trigger an audit or priority on a later round. If an elected representative fails to upload, use a predefined fallback or record the missed coverage. Do not infer successful representation merely from election.

At five clients, exact subset scoring is feasible. For larger cohorts, evaluate greedy selection, a sparse peer graph, or relay-based selection as separate implementations. Direct all-to-all unicast sends `N(N-1)` reports per decision round. A server relay adds report uploads and downlink deliveries. Charge application payload and specify whether downlinks are counted per recipient or as measured shared broadcast traffic.

## Evaluation and decision records

| Method | Evidence for the decision | Main failure to test |
| --- | --- | --- |
| 1. Own history | Client's own recent or last-synced statistic | Stable statistics hide useful learning progress, or gradual drift escapes a one-round comparison |
| 2. Server history | Shared prior statistics or a current statistics-only collection | The bank is stale, its average conceals distinct groups, or telemetry fan-out offsets savings |
| 3. Current peers | Reports from other clients in the same decision phase | Similar statistics hide complementary updates, or an elected representative fails to contribute |

- Compare full participation, random skipping, and round robin with the three methods on identical stream partitions and paired seeds. Include near-identical, distinct, clustered, and changing relationships. Run a threshold sweep for each method to compare at matched total byte budgets. For a fixed participant count, compare how each method ranks clients or covers a subset.
- Use a shared rule for cold starts, the maximum interval between successful syncs, and required participation. If every client defers and the evaluation uses a fixed round clock, explicitly record a no-aggregation round that carries the server model forward. A forced minimum participant count is a separate policy setting. Log forced visits and charge their traffic.
- Track the chosen reference and its age, the statistic distance, threshold, selection reason, local optimizer work, accepted uploads, aggregate recipients, and actual bytes by message type and direction. Compare final server Task-IL accuracy and forgetting with full participation. Include class and task metrics so hidden losses are visible.
- Use same-checkpoint subset experiments to check the core assumption behind each rule: whether clients labeled redundant can be omitted together while preserving current and previous tasks. Record how often a rule skips a client with a useful update and how often it pays for a redundant one. Compare the methods first with the same statistic, then vary the statistic separately.

The three methods can later be combined. Local novelty can trigger a request, the server history can provide a broader reference, and peer comparison can resolve simultaneous requests from similar clients. Evaluate them independently first so that any improvement can be attributed to a decision mechanism and its real bandwidth cost.
