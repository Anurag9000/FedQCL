# Selective synchronization in federated continual learning

## Research question

Can a server use compact, possibly stale evidence about client data, replay memory, and prior model updates to decide which clients should skip a round, reducing total two-way communication while preserving final global accuracy and limiting forgetting?

A sync in this study means a client uploads its trained model and receives the resulting aggregate. A skipped client performs neither transfer. Under the current simulator's skip semantics it also skips that round's training and keeps its previous model. Its next training round starts from that retained state unless a separately measured rejoin policy refreshes it first.

The bandwidth motivation and candidate controller inputs are specified in the [low-bandwidth signal catalog](FEDQCL_LOW_BANDWIDTH_SIGNALS.md). This includes separate accounting for full model payloads, compact reports, and strictly silent rounds.

The three client decision protocols are specified in [Client decisions for selective synchronization](FEDQCL_CLIENT_SYNC_PROTOCOLS.md). They use a client's own history, statistics returned by the server, or current statistics exchanged among peers.

The [implementation and experiment roadmap](FEDQCL_SYNC_EXPERIMENT_ROADMAP.md) defines the first build, pilot, acceptance criteria, and evidence needed for a paper.

The main comparison is the accuracy and forgetting achieved at the same total byte budget. A second comparison fixes acceptable accuracy and forgetting losses relative to full participation and asks how many bytes each policy can save. Choose those tolerances on validation runs before reporting held-out results. Treat the observed degradation under skipping as a motivating observation. The protocol will test whether it holds across seeds, budgets, and stream patterns.

## Starting protocol and implementation checks

- Use the CIFAR-100 launcher configuration: five clients, five tasks of 20 consecutive labels, ten rounds per task, task IDs supplied at inference, ResNet-18 with task heads, and FedAvg. Start with five local epochs per participating round. Keep client dataset sizes, local batch size, learning rate, and replay budget fixed across selectors.
- Preserve the current FedQCL client objective in the primary study, including the Q-weighted replay term and task-boundary state. If simple experience replay is part of the intended claim, add a separate FedAvg plus ordinary replay condition with an explicit replay coefficient. Run representative selectors under both objectives to see whether the effect carries over. The current method is defined in [clientavg_tl.py](../system/flcore/clients/clientavg_tl.py).
- The server starts selected client training threads, collects eligible uploads, aggregates, then broadcasts to visible clients. Log selection, accepted uploads, and broadcast recipients separately. Treat worker exceptions and unexpected missing uploads as run integrity failures for the main comparison. Compare aggregation of parameters and handling of model buffers, including BatchNorm, consistently across policies. See [serveravg.py](../system/flcore/servers/serveravg.py) and [serverbase.py](../system/flcore/servers/serverbase.py).
- Evaluate the **same frozen server model** on a fixed held-out evaluation set for every policy. The existing evaluator calls each client's own model, which can be stale after skipping. Its saved accuracy list also appends one value per seen task per round. Fifty rounds can therefore produce 150 values. The maximum of that list is a peak task-specific value, and the list needs explicit round and task IDs before it can support a longitudinal analysis. Historical visibility results should be audited against this distinction and against client-thread failures before use.
- A client hidden on the last round of a task can miss the code path that inserts task examples into replay memory. Compare selectors first with native boundary behavior. Then use a separate controlled replay-admission variant in which each client can perform a local boundary memory capture from examples already available to it, regardless of sync participation. Record missing memory, first and last participation, and Q state per task. Any changes to Q timing or model refresh belong in separately labeled variants.

## Controlled data streams

Separate stream construction from scheduling. The generator knows the ground-truth allocation. The practical scheduler sees only information the protocol would actually deliver.

For each task, accept either positive per-client class profiles, or a desired symmetric matrix of pairwise similarities. For a profile, accept an exact count table or use its proportions as the mean of a seeded Dirichlet draw. Concentration controls variation around that profile. For a similarity matrix, start from seeded profiles and fit feasible proportions to the requested relationships. Use total variation similarity, defined as one minus half the sum of absolute differences in class proportions. Hold every client above a configurable minimum number of examples per class.

Convert profiles to integer allocations with client totals and original class inventories as constraints. For direct count inputs, validate row sums, column sums, and coverage. For similarity targets, try multiple seeded initializations, report residuals after rounding and allow a configured tolerance. Distinguish a proven conflict, such as triangle inequality violations in the corresponding distances, from failure to find a fit. Many profiles can realize the same similarities, so save the profiles as well. Use one seeded permutation of each class and assign examples without replacement. Save original example IDs to audit uniqueness. Keep validation and test allocations fixed across training regimes.

Start with four regimes:

| Regime | Controlled relationship across clients |
| --- | --- |
| Nearly identical | All pairs have high similarity, with separate examples per client |
| Clustered | Configurable individual pairs have high or low similarity |
| Strongly separated | All pairs are as different as the coverage floor and class inventories allow |
| Changing | Reassign which clients are similar across tasks, with controlled timing and known ground truth |

Include the original Dirichlet setup as a reference. Use a controlled-mode sampler that preserves the intended class proportions during training. The existing inverse-frequency sampler tends to equalize exposure among present classes, so log actual sampled label counts as well as shard counts. Treat sampler choice as a separate factor when comparing against the legacy protocol. Record the requested and realized matrices, minimum coverage, partition seed, and image assignment provenance.

## Selection methodologies

### 1. Distribution coverage with cached summaries

At a participating round, a client supplies a compact label histogram for the data it has observed. The server caches the histogram with its task and round, and uses it only in subsequent decisions. Seed the cache with an explicitly charged initial probe or full sync. For each round, select a subset that represents the available profiles. A simple score sums, over all clients, the similarity to the most representative selected client, weighted by local data size or equally in the balanced setting. Use a fixed number of selected clients and solve exactly at five clients. Use greedy marginal improvement for larger settings.

Explore alternate subsets when two candidates provide similar coverage. Set a maximum consecutive skip length and promote overdue clients. Treat a task change or missing summary as increased uncertainty. For a completely silent skipped client, its cached evidence ages until it participates again. Any optional probe is a separate protocol with charged traffic.

### 2. Replay and update coverage

Extend the first selector with summaries of prior-task replay and model updates. Compare three inputs separately before combining them:

- Replay coverage by task and optionally by class, with explicit indicators for missing memories.
- A fixed-size projection of an already uploaded model update, computed by the server. Compute each delta relative to the model version the client actually started from. Keep the starting version, model age, update norm, and projection timestamp.
- Current label proportions and their age, as in the distribution-only selector.

Reward subsets that cover both current data and historical replay, and retain clients whose evidence is missing or outdated when the budget allows. Tune the signal weights on development partitions. A simple combination of distances is the first implementation. A second candidate matches the projected weighted mean update of all clients using cached deltas and penalizes uncertainty. Compare both against a distribution-only policy to find where replay or update history adds value.

### 3. Direct prediction of round impact

Train a lightweight predictor on counterfactual round outcomes. Candidate features include subset coverage, oldest skipped age, how many clients are near their maximum gap, replay support, and prior-update dispersion. Predict two outcomes separately: current-task validation loss change and past-task validation loss change relative to full participation. Fit on development streams, hold out entire partition seeds and shift patterns, and compare ranking quality and calibration with simple heuristics.

Start with a fixed participant count. Once the predictor is calibrated, try choosing the cheapest subset whose predicted losses plus an uncertainty margin meet both tolerances. Reserve room for mandatory visits. If no candidate meets the tolerances under a hard byte cap, choose the feasible subset with the lowest predicted loss and log the violation. Under a soft budget, permit an extra sync and charge it. Keep this adaptive policy separate from the fixed-budget comparison.

## Counterfactual study

At prespecified rounds early in a task, near task boundaries, and after clients have accumulated different skip ages, snapshot server and client parameters, buffers, optimizer state, replay contents and iterator state, Q state, version counters, and random state. Re-run all ten three-client subsets and the full cohort from each same starting state. Use controlled per-client randomness and isolated branches so counterfactuals do not advance the actual training trajectory. Evaluate each resulting aggregate on separate validation examples from current and past tasks.

Report how well each signal ranks the realized subsets, the loss gap from the best subset at the same budget, and how often a supposedly redundant client matters to old tasks. Use several-step branches under a shared continuation rule and full end-to-end runs to test whether one-round rankings predict final outcomes. The exhaustive oracle spends extra simulator compute and uses counterfactual validation. Report it as a diagnostic upper reference, separate from practical schedulers. A second oracle that sees the true stream matrix isolates the value of perfect distribution information without future outcome access.

## Comparison and measurement

| Factor | Initial choices |
| --- | --- |
| Selection policy | Full, uniform random, round robin or oldest first, distribution coverage, cached update diversity, replay-aware coverage |
| Per-round budget | Five as reference, then four, three, and two selected clients |
| Skip duration | Tight and loose maximum gaps, plus random independent versus block skips at matched participation counts |
| Data | Four controlled regimes plus the original Dirichlet reference |
| Mechanism ablations | Current histogram, replay summary, update history, freshness controls, and their combinations |
| Rejoin behavior | Native stale start, then an optional pre-training refresh that is charged separately |

Start at three of five with full, random, round robin, histogram-only, and replay-aware methods. Compare paired runs on the same independently generated partitions and training seeds. Then sweep budgets and skip horizons and add larger client populations after validating the mechanism at five clients. When studying duration, hold total sync counts approximately equal, report the realized gap distribution by client, and enforce the same eligible schedules for matched policy comparisons. Keep hyperparameter tuning separate from final test evaluation.

Use a fixed schedule of evaluations of the server model. Let `A[r,t]` be accuracy on task `t` after round `r`. Primary final accuracy is the mean of the five task accuracies at the final round. Report macro class accuracy and the worst task as secondary measures. For each old task, subtract its final accuracy from its maximum observed server accuracy after learning that task. Report the mean and individual forgetting values, with the measurement grid fixed across runs. Also log acquisition of the current task, local training work, selected and contributing clients, skip gaps, missing replay, and model-version gaps.

Count logical bytes separately for uploads, downloads, and controller traffic. This is an in-process simulator, so use actual serialized message lengths or a declared tensor payload model. Distinguish either estimate from measured network traffic. Include probes, refresh downloads, and any fixed initialization traffic in every policy under the same convention. If a client ordinarily receives one model and sends one model, selecting three of five gives a nominal 40% reduction in round model payload before overhead and exceptions. Report realized savings and total compute separately. Include uncertainty across paired partition and training seeds and show the full accuracy, forgetting, and bytes trade-off rather than a single favorable point.

## Decision sequence

1. Instrument the simulator and reproduce full and random visibility baselines with direct server evaluation and event accounting.
2. Generate controlled streams and audit class coverage, actual training exposure, and achieved similarity before launching comparisons.
3. Run the counterfactual study to judge whether data similarity predicts interchangeable updates and where replay or model age changes the answer.
4. Compare the fixed-budget practical selectors against matched baselines. Move to adaptive budgets only after the fixed-budget results identify useful signals.
5. Report failure regimes as findings. If label similarity explains little under stable streams, that weakens the distribution-only hypothesis. If only the replay or age signals improve the result, quantify their independent contribution. If savings disappear after extra traffic, report that trade-off directly.

## Relation to prior work

[Clustered Sampling](https://proceedings.mlr.press/v139/fraboni21a.html) uses client sample size or model similarity for representative FL sampling. [DivFL](https://openreview.net/pdf?id=nwKXyFvaUm) chooses representative gradients using a submodular objective. They motivate matched-budget baselines. A cached approximation should be labeled as such if it changes their information access or scoring.

[FedWeIT](https://proceedings.mlr.press/v139/yoon21b.html) studies communication and forgetting with selective transfer of decomposed parameters. [Re-Fed](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Towards_Efficient_Replay_in_Federated_Incremental_Learning_CVPR_2024_paper.html) studies replay sample selection. The proposed study tests whether stale, replay-aware evidence improves pre-round selection under whole-model, two-way skips. It needs empirical evidence and a broader prior-art review before making a novelty claim.
