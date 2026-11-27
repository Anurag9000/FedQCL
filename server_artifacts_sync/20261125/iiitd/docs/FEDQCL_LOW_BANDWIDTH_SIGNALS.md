# Low bandwidth signals for selective federated synchronization

## Bandwidth motivation

The expensive operation in this study is transferring a full model between a client and server. For a dense model with `P` transmitted scalars at `b` bytes each, an upload and download of equal size cost `2Pb` bytes before metadata. Larger CNNs, ResNets, vision transformers, foundation models, and language models make this baseline more expensive as the transmitted parameter count rises. For systems that already communicate only adapters or compressed deltas, compare against their actual transmitted payload separately.

A client can derive a much smaller report from its data, replay memory, forward pass, loss, gradients, or model state. The controller uses reports to decide when a full exchange is worth its bandwidth. The lower cost comes from fixing the number of transmitted scalars and their precision. Full gradients and dense model deltas can be model-sized. Per-example activations, feature maps, attention tensors, Jacobians, and posterior covariances can be much larger than their summaries. Some statistics are cheap to transmit after computation and expensive to estimate on a client.

## Size examples

These are illustrative payload sizes before identifiers, framing, or transport overhead. Sizes for the experiment should use the actual serialized payload and reporting frequency.

| Payload | Assumption | Size |
| --- | --- | ---: |
| Full two-way model exchange | 11 million parameters, FP32 each direction | 88 MB |
| Full two-way model exchange | 1 billion parameters, 16 bits each direction | 4 GB |
| Class histogram | 20 unsigned 32-bit counts | 80 B |
| Loss or entropy report | 8 FP32 summary values | 32 B |
| Gradient or update sketch | 128 FP16 coordinates | 256 B |
| Feature report | Mean and variance of 128 projected coordinates, FP16 | 512 B |
| Class prototypes | 20 classes × 128 projected coordinates, FP16 | 5,120 B |
| Raw CIFAR image | 32 × 32 × 3 unsigned bytes, before compression | 3,072 B |
| Raw intermediate feature map | 64 × 32 × 32 FP32 values, one example | 262,144 B |

The example class-prototype report exceeds one raw CIFAR image and is still small relative to a full model exchange. Compare reports with the actual alternative, such as an image batch or model exchange, over the same training interval. Count upload and download bytes separately. Include report requests, headers, anchor distribution, initialization, and any rejoin refreshes. The simulator can report logical serialized bytes independently of physical network traffic. Report client computation, memory, and elapsed time separately from bytes.

## Tensors and practical summaries

| Source | Possible bounded report | What the controller could learn |
| --- | --- | --- |
| Weights, biases, and model updates | Layer norms, distances to the start model or prior checkpoint, fixed random projection, or selected layer or head deltas | Model drift, update magnitude and direction |
| Gradients and gradient statistics | Gradient norm, coordinate or layer variance, sign agreement, projected mean gradient, separate current-task and replay sketches | Whether updates point in similar directions and whether they differ on older tasks |
| Pre-activations, activations, feature maps, representations, embeddings, hidden states, residual streams, and their statistics | Pool spatial or token axes, then send projected means, diagonal variances, quantiles, or class prototypes | Data or representation drift, class-conditional similarity |
| Logits, predictions, and loss | Mean logits or probabilities on a fixed probe, per-class loss, accuracy or disagreement, predictive entropy, margins, quantiles | Functional differences, difficult classes, potential forgetting |
| Replay memory | Per-task or per-class counts, ages, loss summaries, or projected feature prototypes | Historical coverage and which retained knowledge may be missing from a subset |
| Attention matrices and query, key, and value vectors | Per-head entropy, norm, token-distance buckets, or pooled projected moments | How attention or layer behavior changes over time |
| Gating signals | Gate activation counts, occupancy, entropy, and routing frequencies | Which experts or paths cover a client stream |
| Input-output and parameter-output Jacobians | Directional Jacobian products on a small probe, projected sensitivity, trace or norm estimates | Sensitivity to inputs or parameter directions |
| Normalization parameters and statistics | Selected batch or running means and variances, scaling and shifting parameter summaries, or drift scores | Changes in data, model state, or calibration |

For normalization diagnostics, track feature-wise means and variances, normalized activation distributions, covariance before and after normalization, and feature decorrelation. Compare train with test, client with client, task with task, and current with prior checkpoints for normalization mismatch, internal activation distribution shift, and scale consistency. Include model mode and the sample count. Batch statistics and running statistics answer different questions.

## Information, dependence, and discrepancy

| Quantity | Interpretation and use |
| --- | --- |
| Shannon entropy, differential entropy, Rényi entropy, Tsallis entropy | Discrete uncertainty, continuous dispersion relative to a specified coordinate system, or generalized uncertainty with a declared order |
| Conditional entropy, joint entropy, cross-entropy | Remaining uncertainty given a variable, combined uncertainty, and the prediction cost under a reference distribution |
| Mutual information, conditional mutual information, total correlation | Shared information, shared information after conditioning, and multivariate dependence |
| KL divergence, Jensen–Shannon divergence, Rényi divergence, Hellinger distance | Directional discrepancy, symmetric discrepancy, order-dependent discrepancy, and distributional separation |
| Wasserstein distance, maximum mean discrepancy | Transport cost under a declared ground metric and distribution discrepancy under a declared kernel |
| Information gain, information bottleneck, information retention | Reduction in uncertainty, compression subject to predictive information, and evidence of preserved prior knowledge |
| Information redundancy, information synergy | Overlapping evidence across clients and evidence available jointly. Use a declared decomposition if estimating these formally. Compare single and joint omissions as a separate operational measure of update interaction. |
| Transfer entropy | Directed temporal dependence given past observations. Requires a time series and a declared conditioning history. |
| Fisher information | Sensitivity of the likelihood to parameters. A trace, layer summary, or diagonal sketch can serve as a sensitivity signal. |

Start with low-dimensional histograms and prediction probabilities for entropy, cross-entropy, Jensen–Shannon divergence, or Hellinger distance. Define the probability axes, histogram smoothing, reference distribution, and sample size. Estimate Wasserstein or MMD on shared projected features, with a specified distance or kernel. Mutual information, conditional information, transfer entropy, and information decompositions need a defensible estimator and enough data for the intended comparison. A scalar sent over the network can still require many samples, richer joint statistics, or substantial computation.

## Bayesian and uncertainty statistics

| Group | Candidate quantities |
| --- | --- |
| Parameter beliefs | Prior mean and variance, posterior mean, variance, covariance, precision, and parameter correlations |
| Belief evolution | Posterior entropy, contraction rate, drift, prior-posterior KL divergence, Bayesian surprise, and posterior signal-to-noise ratio |
| Predictive behavior | Posterior predictive distribution and variance, epistemic and aleatoric uncertainty, predictive entropy, expected conditional entropy, mutual information between predictions and parameters, and predictive surprise |
| Intervals and calibration | Credible intervals, confidence intervals, uncertainty propagation, uncertainty calibration, uncertainty-error correlation, and uncertainty-drift correlation |
| Approximation and evidence | Model evidence or marginal likelihood, evidence lower bound (ELBO), effective sample size, Monte Carlo estimation variance, variational approximation error, and posterior multimodality |
| Ensemble behavior | Ensemble disagreement and diversity |

In a deterministic training run, start with observed entropy, loss dispersion, and disagreement where repeated predictions are available. Bayesian quantities require a specified likelihood, prior, and posterior approximation. Ensembles and repeated stochastic passes add compute. For a fixed set of posterior predictive samples, predictive entropy minus expected conditional entropy can estimate information between predictions and parameters. Use explicit assumptions, such as a last-layer approximation or diagonal covariance, to keep reports bounded. Dense posterior mean and variance can each scale with parameter count, and dense covariance scales quadratically.

## Representation similarity

| Method | What it compares |
| --- | --- |
| CCA, SVCCA, PWCCA | Shared linear representation directions, with singular-vector reduction or projection weighting |
| Centered kernel alignment (CKA), representational similarity analysis (RSA), Gram matrix similarity, HSIC | Relationships among observations, alignment of representation geometry, or kernel dependence |
| Procrustes distance, orthogonal Procrustes alignment | Representations after a specified alignment |
| Principal angles, subspace overlap, eigenspace similarity, singular-vector alignment | Agreement between dominant subspaces or directions |
| Neighborhood overlap | Agreement among nearest neighbors |
| Functional similarity, parameter similarity, prediction agreement | Behavior on common inputs, parameter-space proximity, or output agreement |

For direct cross-client comparisons on common inputs, use paired observations in a common order, such as a versioned public or otherwise authorized anchor set. Share small projected outputs or a sketch suited to the chosen statistic. CKA can compare layers of different widths on common observations. For unpaired private examples, use a distribution comparison on compatible embeddings or class-conditional summaries. Track feature extractor and model versions so drift in the encoder can be distinguished from drift in the inputs when the design permits. [Kornblith et al. on CKA](https://proceedings.mlr.press/v97/kornblith19a.html)

## Spectral and geometric statistics

| Family | Candidate quantities |
| --- | --- |
| Matrix spectra | Eigenvalues and eigenvectors, singular values and vectors, spectral density, radius, gap, entropy, norm, decay rate, concentration and drift, power-law exponent, dominant eigenspace, and explained variance ratio |
| Rank and conditioning | Effective rank, stable rank, numerical rank, participation ratio, and condition number |
| Temporal spectra | Frequency spectrum, power spectral density, cross-spectral density, and spectral coherence |
| Vector distances | Euclidean, Manhattan, Minkowski, Mahalanobis, angular, cosine, and Hamming distances |
| Distribution and manifold distances | Wasserstein, Fréchet, Hausdorff, Bregman divergence, Fisher–Rao, geodesic, Grassmannian, and subspace distances |
| Reference distances | Distance from initialization, the previous checkpoint, a class centroid, a decision boundary, or an explicitly defined, available reference optimum |
| Trajectories and local geometry | Parameter trajectory length, trajectory curvature, convex hull volume, local manifold volume, neighborhood radius, orthogonality, collinearity, geometric curvature, and tangent-space alignment |

Applicable sources include weights, attention, graphs and kernels, representation covariances, Jacobians, and Hessians. Prefer scalar rank or norm summaries, top eigenvalues, or a projected subspace where it fits the byte budget. Record whether the spectrum is computed exactly or estimated. Temporal spectra and coherence require adequate sampling across time. Distances require compatible coordinates, a meaningful reference, and stated scale. A reference optimum unavailable to the live controller belongs in offline analysis.

## Reporting protocol and ablations

1. **Start with inexpensive fields.** Include a schema version, client ID, round, task ID, source model version, sample count, last sync age, class counts, replay counts, and a few loss or uncertainty summaries. Add a fixed-size gradient, model update, or feature sketch in separate variants before combining them. Use an agreed projection seed, dimensions, and precision. Measure report size and extraction time.
2. **Define when evidence arrives.** In strict silence mode, reports ride along with a full sync and influence later decisions. An update sketch computed by the server from an already received update costs no extra transmitted coordinates. In a separate telemetry mode, a client can send a compact report while skipping both full-model transfers. Score a report generated during a round only for decisions that occur after it arrives, and charge the messages in the telemetry mode.
3. **Compare signals under matched byte budgets.** Start with class counts and replay coverage. Compare them with feature summaries, functional probes, and signed update sketches. Use identical class proportions with different within-class examples, and controlled replay differences, to test what each additional signal reveals. Record how sampled training proportions differ from shard proportions.
4. **Measure predictive value and deployment cost.** Use matched checkpoint branches to test whether each report predicts the loss from skipping subsets on current and past tasks. Then compare final global accuracy, forgetting, and total bytes in complete runs. Sweep both report size and cadence. Rank signals by the improvement they add at a given byte and compute budget.
5. **Define what the server is allowed to observe.** Treat raw examples, per-example tensors, and full training gradients as distinct from aggregate reports. Specify visibility, protection, minimum sample counts, and any clipping, aggregation, or noise in the protocol. Even small reports can disclose client properties. Gradient reconstruction has been demonstrated for shared training gradients under specific conditions. [Zhu et al. on gradient leakage](https://proceedings.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)
