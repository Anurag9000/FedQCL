"""Research flags; no effect on legacy execution."""
def add_arguments(parser):
    p=parser
    p.add_argument("--sync-policy", choices=["legacy","full","random","oldest","own","server","peer","cached","update_match","combined12","combined13","combined23","combined123","adaptive","divfl_sketch","block_random"], default="legacy")
    p.add_argument("--sync-mode",choices=["threshold","fixed"],default="fixed")
    p.add_argument("--sync-k",type=int,default=3)
    p.add_argument("--sync-threshold",type=float,default=.1)
    p.add_argument("--sync-max-skip",type=int,default=4)
    p.add_argument("--sync-observation-size",type=int,default=128)
    p.add_argument("--sync-min-observation",type=int,default=32)
    p.add_argument("--sync-history",type=int,default=5)
    p.add_argument("--sync-local-reference",choices=["last_sync","previous","window"],default="last_sync")
    p.add_argument("--sync-bank-reference",choices=["latest","prior","mean","weighted_mean","envelope","history","aggregate"],default="latest")
    p.add_argument("--sync-bank-age",type=int,default=5)
    p.add_argument("--sync-include-self",action="store_true")
    p.add_argument("--sync-delivery",choices=["piggyback","broadcast","telemetry"],default="piggyback")
    p.add_argument("--sync-peer-transport",choices=["direct","relay","sparse","distributed_sparse"],default="direct")
    p.add_argument("--sync-fallback",action="store_true",help="Charged replacement after failed model upload; peer relay only")
    p.add_argument("--sync-dynamic-topology",action="store_true")
    p.add_argument("--sync-gossip-fault",choices=["none","delay","duplicate","reorder","corrupt"],default="none")
    p.add_argument("--sync-charge-control",action="store_true")
    p.add_argument("--sync-bandwidth-audit",action="store_true",help="Charge initial state and acknowledgements; reconcile logical traffic")
    p.add_argument("--sync-histogram-rr-epsilon",type=float,default=0.,help="Per-observation randomized response; not full training DP")
    p.add_argument("--sync-peer-degree",type=int,default=1)
    p.add_argument("--sync-peer-loss",type=float,default=0.)
    p.add_argument("--sync-peer-solver",choices=["exact","greedy"],default="exact")
    p.add_argument("--sync-coverage-weight",choices=["equal","samples","replay","aggregation"],default="equal")
    p.add_argument("--sync-update-weighted",action="store_true")
    p.add_argument("--sync-peer-objective",choices=["distance","coverage"],default="distance")
    p.add_argument("--sync-eval-split",choices=["validation","test"],default="test")
    p.add_argument("--sync-output",default="research_results")
    p.add_argument("--sync-refresh",action="store_true")
    p.add_argument("--sync-fresh-start",action="store_true",help="Download current global model before selected training, not afterwards")
    p.add_argument("--sync-utility",choices=["off","local","server","peer","loss","balanced","random"],default="off")
    p.add_argument("--sync-utility-budget",type=float,default=0.,help="Prospective logical byte ceiling as fraction of dense full exchange reference")
    p.add_argument("--sync-utility-price",type=float,default=.015)
    p.add_argument("--sync-utility-past",type=float,default=1.)
    p.add_argument("--sync-utility-mass",type=float,default=.1)
    p.add_argument("--sync-utility-match",type=float,default=.1)
    p.add_argument("--sync-utility-fair",type=float,default=.05)
    p.add_argument("--sync-utility-risk",type=float,default=.1)
    p.add_argument("--sync-utility-gate",type=float,default=.01)
    p.add_argument("--sync-utility-retention",type=float,default=.05)
    p.add_argument("--sync-delta-aggregation",action="store_true",help="Separate stale-base delta ablation, not ordinary FedAvg")
    p.add_argument("--sync-utility-proxy",action="store_true",help="Common fixed classifier-head proxy instead of native gradients")
    p.add_argument("--sync-utility-adaptive-precision",action="store_true")
    p.add_argument("--sync-utility-model",help="Frozen qualified offline model, M2 only")
    p.add_argument("--sync-utility-precision",choices=["fp64","fp32","fp16","int8"],default="fp32")
    p.add_argument("--sync-local-capture",action="store_true",default=True)
    p.add_argument("--sync-balanced-sampler",action="store_true")
    p.add_argument("--sync-queue-mode",choices=["native","sample_mean"],default="native")
    p.add_argument("--sync-reference-recovery",action="store_true")
    p.add_argument("--sync-v-normalized-lr",action="store_true",help="Divide local LR by V for a controlled penalty-weight ablation")
    p.add_argument("--sync-weighting",choices=["current"],default="current")
    p.add_argument("--sync-aggregation",choices=["complete"],default="complete")
    p.add_argument("--sync-resume",action="store_true")
    p.add_argument("--sync-stop-after-round",type=int,help="Pause at committed round for restart verification")
    p.add_argument("--sync-checkpoint",action="store_true")
    p.add_argument("--sync-profile",action="store_true",help="Diagnostic per-round CUDA allocator peaks and process high-water RSS")
    p.add_argument("--sync-counterfactual",action="store_true")
    p.add_argument("--sync-er",action="store_true",default=True,help="Mandatory vanilla replay in research runs")
    p.add_argument("--sync-memory-profile",choices=["uniform"],default="uniform")
    p.add_argument("--sync-windowed",action="store_true")
    p.add_argument("--sync-controlled-stream",action="store_true",help="Require verified controlled-stream bundle and offline ground-truth audit")
    p.add_argument("--sync-audit-references",action="store_true",help="Record researcher-only exact report references")
    p.add_argument("--sync-posttrain",action="store_true")
    p.add_argument("--sync-bank-transfer",choices=["snapshot","delta"],default="snapshot")
    p.add_argument("--sync-failures",help="JSON mapping round to upload/reject/download client IDs")
    p.add_argument("--sync-signal",choices=["histogram","replay","feature","gradient","update","replay_histogram","prototypes","feature_covariance","spectral","parameter_norms","normalization","predictions","loss","uncertainty","entropy","jacobian","fisher","gradient_variance","multichannel","gradient_conflict","information","temporal","bayesian_head","representation_anchor","replay_diagnostics","fisher_full","parameter_jacobian","head_hessian","conditional_moments","replay_composition","mutual_information","cka","gradient_variance_only","tier_a_features","tier_a_gradients","tier_a_functional","tier_a_replay_loss","tier_a_interference","tier_a_prototypes","tier_a_distributions","tier_a_cached_update"],default="histogram")
    p.add_argument("--sync-histogram-distance",choices=["tv","js","hellinger"],default="tv")
    p.add_argument("--sync-sketch-dim",type=int,default=128)
    p.add_argument("--sync-byte-cap",type=int,default=0,help="Hard logical bytes per round including bootstrap")
    p.add_argument("--sync-bootstrap-byte-cap",type=int,default=0)
    p.add_argument("--sync-budget-reserve",action="store_true",help="Reserve a lower bound on future deadline-required model transfers")
    p.add_argument("--sync-byte-schedule",help="Offline per-round source byte caps for matched-budget random baseline")
    p.add_argument("--sync-campaign-byte-cap",type=int,default=0,help="Hard total logical bytes; exhausted budget is reported infeasible")
    p.add_argument("--sync-wire-format",choices=["binary","json","fp16","int8"],default="binary")
    p.add_argument("--sync-count-schedule",help="JSON list of predeclared participant counts for random baseline")
    p.add_argument("--sync-counterfactual-horizon",type=int,default=1)
    p.add_argument("--sync-continuation-policy",choices=["full","oldest","random"],default="full")
    p.add_argument("--sync-oracle-past-weight",type=float,default=0.)
    p.add_argument("--sync-verify-branches",action="store_true")
    p.add_argument("--sync-all-subsets",action="store_true")
    p.add_argument("--sync-impact-model")
    p.add_argument("--sync-current-tolerance",type=float,default=.02)
    p.add_argument("--sync-past-tolerance",type=float,default=.02)
    p.add_argument("--sync-soft-budget",action="store_true")
    p.add_argument("--sync-staleness-predict",action="store_true")
    p.add_argument("--sync-dynamic-threshold",action="store_true")
    p.add_argument("--sync-joint-hybrid",action="store_true")
    p.add_argument("--sync-novelty-weight",type=float,default=1.)
    p.add_argument("--sync-block-length",type=int,default=3)
    p.add_argument("--sync-combine-rule",choices=["union","intersection"],default="union")
    p.add_argument("--sync-native-space",action="store_true",help="Allow comparison across model versions for native gradient/update sketches; explicit ablation")

def validate(a):
    if a.sync_policy=="legacy":
        return
    import math
    fresh=getattr(a,"sync_fresh_start",False);utility=getattr(a,"sync_utility","off")
    if fresh and (not a.sync_controlled_stream or not a.sync_bandwidth_audit or a.sync_refresh or a.sync_posttrain or a.sync_failures):
        raise ValueError("fresh-start requires reliable audited controlled pretraining")
    if getattr(a,"sync_delta_aggregation",False) and (fresh or utility!="off" or not a.sync_bandwidth_audit or not a.sync_controlled_stream or a.sync_refresh):
        raise ValueError("delta aggregation is a separate reliable stale-base ablation")
    if (getattr(a,"sync_utility_proxy",False) or getattr(a,"sync_utility_adaptive_precision",False)) and utility not in ("server","peer"):
        raise ValueError("proxy and adaptive precision require collective telemetry")
    if utility!="off":
        expected={"local":"own","loss":"own","server":"server","peer":"peer","balanced":"random","random":"random"}
        if a.sync_utility_model and utility!="server":raise ValueError("learned utility model requires server telemetry")
        if not fresh or a.sync_policy!=expected[utility] or a.num_clients>10 or a.sync_wire_format!="binary" or a.sync_count_schedule:
            raise ValueError("utility protocol requires fresh audited start, compatible mechanism and <=10 clients")
        if not 0<=a.sync_utility_budget<=1 or not 1<=a.sync_sketch_dim<=4096 or a.sync_observation_size>128:
            raise ValueError("utility budget/probe bounds")
        if any(not math.isfinite(x) or x<0 for x in (a.sync_utility_price,a.sync_utility_past,a.sync_utility_mass,a.sync_utility_match,a.sync_utility_fair,a.sync_utility_risk,a.sync_utility_gate,a.sync_utility_retention)):
            raise ValueError("invalid utility coefficients")
    if a.hide_one_client or a.hidden_clients_per_round:
        raise ValueError("research policies and legacy hiding are mutually exclusive")
    if a.model!="resnet18" or a.algorithm!="FedAvg" or a.times!=1 or a.prev!=0:
        raise ValueError("research requires one ResNet18 FedAvg run")
    if a.tasks<1 or a.global_rounds<0 or (a.global_rounds+1)%a.tasks or a.global_rounds+1<a.tasks:
        raise ValueError("invalid round schedule")
    if not 1<=a.sync_k<=a.num_clients or not 0<=a.sync_threshold<=1:
        raise ValueError("invalid synchronization budget or threshold")
    if not 1<=a.sync_min_observation<=a.sync_observation_size or a.sync_history<1:
        raise ValueError("invalid report window")
    if min(a.sync_max_skip,a.sync_bank_age)<0:
        raise ValueError("invalid age")
    if a.sync_block_length<1 or a.sync_novelty_weight<0:raise ValueError("invalid scheduling setting")
    if a.sync_dynamic_threshold and (a.sync_policy not in ("own","server","combined12") or a.sync_mode!="threshold"):
        raise ValueError("dynamic threshold requires threshold novelty policy")
    if a.sync_joint_hybrid and (a.sync_policy not in ("combined13","combined23","combined123") or a.sync_mode!="fixed"):
        raise ValueError("joint hybrid requires fixed count peer hybrid")
    if a.sync_staleness_predict and a.sync_policy!="cached":raise ValueError("trend prediction requires cached coverage")
    if a.sync_policy=="divfl_sketch" and (a.sync_signal!="gradient" or not a.sync_native_space or a.sync_mode!="fixed"):
        raise ValueError("gradient sketch comparator requires native gradient coordinates and fixed count")
    if a.sync_policy in ("random","oldest","block_random") and a.sync_mode!="fixed":
        raise ValueError("random and oldest require fixed count")
    if a.auto_break or a.client_drop_rate or a.train_slow_rate or a.send_slow_rate or a.time_select or a.num_new_clients or a.privacy:
        raise ValueError("legacy dropout, early stopping, DP and new-client options not supported by research engine")
    if min(a.sync_byte_cap,a.sync_campaign_byte_cap,a.sync_bootstrap_byte_cap)<0:
        raise ValueError("invalid byte cap")
    if getattr(a,"sync_bandwidth_audit",False) and (not a.sync_charge_control or a.sync_byte_cap or a.sync_campaign_byte_cap or a.sync_byte_schedule or a.sync_failures or a.sync_posttrain or not a.sync_controlled_stream):
        raise ValueError("audited traffic requires charged controls and reliable controlled pre-training; hard caps use a separate predictor")
    if getattr(a,"sync_bandwidth_audit",False):
        if a.sync_wire_format!="binary" or a.sync_dynamic_threshold or (a.sync_policy=="own" and a.sync_local_reference!="last_sync") or (a.sync_policy=="server" and (a.sync_bank_reference!="aggregate" or a.sync_delivery!="piggyback")) or (a.sync_policy=="peer" and (a.sync_peer_transport!="direct" or a.sync_peer_loss)):
            raise ValueError("primary bandwidth/decision audit requires binary, last-sync, aggregate piggyback or reliable direct peer policy")
    if a.sync_budget_reserve and not a.sync_campaign_byte_cap:
        raise ValueError("future reserve requires campaign byte cap")
    if a.sync_bootstrap_byte_cap and not (a.sync_byte_cap or a.sync_campaign_byte_cap):
        raise ValueError("bootstrap override requires a byte cap")
    if a.sync_byte_schedule and (a.sync_policy!="random" or a.sync_count_schedule or a.sync_byte_cap or a.sync_campaign_byte_cap):
        raise ValueError("byte schedule requires standalone random baseline")
    if (a.sync_byte_cap or a.sync_campaign_byte_cap or a.sync_byte_schedule) and (a.sync_posttrain or a.sync_failures or a.sync_policy=="update_match" or a.sync_mode!="fixed"):
        raise ValueError("hard caps currently require reliable pre-training delivery")
    if a.memory_size<0 or a.batch_size<1 or a.memory_batch_size<1 or a.local_epochs<1:
        raise ValueError("invalid local workload")
    if not a.sync_er or a.sync_reference_recovery or a.sync_v_normalized_lr or a.sync_queue_mode!="native" or a.sync_balanced_sampler:
        raise ValueError("research permits only vanilla SGD + ER, without FEDQCL queues, reference models or weighted sampling")
    if a.sync_weighting!="current" or a.sync_aggregation!="complete" or a.sync_memory_profile!="uniform" or not a.sync_local_capture or a.sync_policy=="mass_stratified" or a.sync_signal=="replay_queue":
        raise ValueError("research requires current-shard-weighted full-state FedAvg and uniform task-balanced replay")
    if a.sync_sketch_dim<1:
        raise ValueError("invalid sketch dimension")
    if a.sync_fallback and (a.sync_policy!="peer" or a.sync_peer_transport!="relay" or a.sync_posttrain or a.sync_counterfactual or a.sync_byte_cap or a.sync_campaign_byte_cap):
        raise ValueError("fallback requires reliable peer relay, pre-training, and no hard cap/counterfactual")
    if a.sync_peer_degree<1 or not 0<=a.sync_peer_loss<=1:
        raise ValueError("invalid peer topology/loss")
    if a.sync_histogram_rr_epsilon<0 or a.sync_histogram_rr_epsilon>50:
        raise ValueError("invalid randomized-response epsilon")
    if a.sync_histogram_rr_epsilon and a.sync_signal!="histogram":
        raise ValueError("randomized response only applies to histogram reports")
    if (a.sync_dynamic_topology or a.sync_gossip_fault!="none") and a.sync_peer_transport!="distributed_sparse":
        raise ValueError("gossip fault/topology options require distributed_sparse")
    if a.sync_peer_transport in ("sparse","distributed_sparse") and a.sync_policy!="peer":
        raise ValueError("sparse graph is a standalone centralized peer variant")
    if a.sync_peer_loss and a.sync_peer_transport not in ("direct","distributed_sparse"):
        raise ValueError("peer loss is supported for direct transport only")
    if a.sync_count_schedule and (a.sync_policy!="random" or a.sync_byte_cap or a.sync_campaign_byte_cap):
        raise ValueError("count schedule requires random policy without byte cap")
    if a.sync_policy=="adaptive" and (not a.sync_impact_model or a.sync_mode!="fixed" or a.sync_signal not in ("histogram","multichannel")):
        raise ValueError("adaptive requires calibrated impact model, histogram and fixed upper cap")
    if min(a.sync_current_tolerance,a.sync_past_tolerance)<0:
        raise ValueError("negative impact tolerance")
    if a.sync_policy in ("cached","update_match") and a.sync_mode!="fixed":
        raise ValueError("cached selectors require fixed count")
    if a.sync_policy=="update_match" and (a.sync_signal!="update" or a.sync_posttrain or not a.sync_native_space):
        raise ValueError("cached update matching requires update signal, native coordinates, and pre-training selection")
    if a.sync_signal=="update" and not a.sync_posttrain and a.sync_policy!="update_match":
        raise ValueError("update statistic requires post-training selection")
    if a.sync_wire_format in ("fp16","int8") and a.sync_signal=="multichannel":
        raise ValueError("multichannel quantization requires block-specific scales")
    if a.sync_coverage_weight=="aggregation" and a.sync_weighting=="processed":
        raise ValueError("processed aggregation weight is not available at pre-training admission")
    if a.sync_signal.startswith("tier_a_"):
        if a.sync_wire_format not in ("binary","json") or a.sync_posttrain:
            raise ValueError("Tier-A reports require full-precision pre-training selection")
        if a.sync_signal in ("tier_a_gradients","tier_a_interference","tier_a_cached_update") and not a.sync_native_space:
            raise ValueError("gradient and update sketches explicitly compare native coordinates")
    if a.sync_bank_reference=="aggregate" and (a.sync_bank_transfer!="snapshot" or a.sync_signal=="multichannel"):
        raise ValueError("aggregate reference requires snapshot transfer and poolable signal")
    if a.sync_signal=="multichannel" and (a.sync_local_reference=="window" or a.sync_bank_reference in ("mean","weighted_mean","envelope")):
        raise ValueError("multichannel reports require non-pooled references")
    if a.sync_update_weighted and a.sync_policy!="update_match":
        raise ValueError("weighted update matching requires update_match policy")
    if a.sync_coverage_weight!="equal" and a.sync_policy not in ("cached","peer"):
        raise ValueError("weighted coverage requires cached or peer policy")
    if a.sync_coverage_weight!="equal" and a.sync_peer_transport in ("sparse","distributed_sparse"):
        raise ValueError("weighted sparse coverage not supported")
    if a.sync_signal not in ("histogram","replay") and a.sync_bank_reference=="envelope":
        raise ValueError("coordinate envelope only supported for histogram/replay")
    if a.sync_oracle_past_weight<0:raise ValueError("negative oracle loss weight")
    if a.sync_counterfactual_horizon<1 or ((a.sync_counterfactual_horizon>1 or a.sync_verify_branches or a.sync_all_subsets) and not a.sync_counterfactual):
        raise ValueError("invalid counterfactual settings")
    if a.sync_counterfactual and a.sync_failures:
        raise ValueError("counterfactual reuse assumes reliable model delivery")
    if a.sync_posttrain and (a.sync_refresh or a.sync_counterfactual):
        raise ValueError("post-training admission cannot be combined with refresh or pre-training counterfactual")
    if getattr(a,"sync_controlled_stream",False) and (not a.sync_windowed or not a.sync_counterfactual or not a.sync_audit_references or a.sync_failures or a.sync_eval_split!="validation" or a.sync_policy not in ("own","server","peer","full","random","oldest")):
        raise ValueError("controlled stream requires reliable windowed audited selection and counterfactual validation")
    if a.sync_windowed and a.few_shot:
        raise ValueError("windowed few-shot indexing is unsupported")
    if a.sync_stop_after_round is not None and (not a.sync_checkpoint or not 0<=a.sync_stop_after_round<=a.global_rounds):
        raise ValueError("pause requires valid round and checkpoints")
    if a.sync_resume and not a.sync_checkpoint:
        raise ValueError("resume requires checkpoints")
