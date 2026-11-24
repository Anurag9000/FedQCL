"""Research flags; no effect on legacy execution."""
def add_arguments(parser):
    p=parser
    p.add_argument("--sync-policy", choices=["legacy","full","random","oldest","own","server","peer","cached","update_match","combined12","combined13","combined23","combined123","adaptive"], default="legacy")
    p.add_argument("--sync-mode",choices=["threshold","fixed"],default="fixed")
    p.add_argument("--sync-k",type=int,default=3)
    p.add_argument("--sync-threshold",type=float,default=.1)
    p.add_argument("--sync-max-skip",type=int,default=4)
    p.add_argument("--sync-observation-size",type=int,default=128)
    p.add_argument("--sync-min-observation",type=int,default=32)
    p.add_argument("--sync-history",type=int,default=5)
    p.add_argument("--sync-local-reference",choices=["last_sync","previous","window"],default="last_sync")
    p.add_argument("--sync-bank-reference",choices=["latest","prior","mean","weighted_mean","envelope","history"],default="latest")
    p.add_argument("--sync-bank-age",type=int,default=5)
    p.add_argument("--sync-include-self",action="store_true")
    p.add_argument("--sync-delivery",choices=["piggyback","broadcast","telemetry"],default="piggyback")
    p.add_argument("--sync-peer-transport",choices=["direct","relay"],default="direct")
    p.add_argument("--sync-peer-solver",choices=["exact","greedy"],default="exact")
    p.add_argument("--sync-peer-objective",choices=["distance","coverage"],default="distance")
    p.add_argument("--sync-output",default="research_results")
    p.add_argument("--sync-refresh",action="store_true")
    p.add_argument("--sync-local-capture",action="store_true")
    p.add_argument("--sync-balanced-sampler",action="store_true")
    p.add_argument("--sync-aggregation",choices=["legacy","complete"],default="legacy")
    p.add_argument("--sync-resume",action="store_true")
    p.add_argument("--sync-stop-after-round",type=int,help="Pause at committed round for restart verification")
    p.add_argument("--sync-checkpoint",action="store_true")
    p.add_argument("--sync-counterfactual",action="store_true")
    p.add_argument("--sync-er",action="store_true")
    p.add_argument("--sync-windowed",action="store_true")
    p.add_argument("--sync-posttrain",action="store_true")
    p.add_argument("--sync-bank-transfer",choices=["snapshot","delta"],default="snapshot")
    p.add_argument("--sync-failures",help="JSON mapping round to upload/reject/download client IDs")
    p.add_argument("--sync-signal",choices=["histogram","replay","feature","gradient","update","replay_histogram","replay_queue","prototypes","feature_covariance","spectral","parameter_norms","normalization","predictions","loss","uncertainty","entropy","jacobian","fisher","gradient_variance"],default="histogram")
    p.add_argument("--sync-histogram-distance",choices=["tv","js","hellinger"],default="tv")
    p.add_argument("--sync-sketch-dim",type=int,default=128)
    p.add_argument("--sync-byte-cap",type=int,default=0,help="Hard logical bytes per round including bootstrap")
    p.add_argument("--sync-campaign-byte-cap",type=int,default=0,help="Hard total logical bytes; exhausted budget is reported infeasible")
    p.add_argument("--sync-wire-format",choices=["binary","json"],default="binary")
    p.add_argument("--sync-count-schedule",help="JSON list of predeclared participant counts for random baseline")
    p.add_argument("--sync-all-subsets",action="store_true")
    p.add_argument("--sync-impact-model")
    p.add_argument("--sync-current-tolerance",type=float,default=.02)
    p.add_argument("--sync-past-tolerance",type=float,default=.02)
    p.add_argument("--sync-soft-budget",action="store_true")
    p.add_argument("--sync-combine-rule",choices=["union","intersection"],default="union")
    p.add_argument("--sync-native-space",action="store_true",help="Allow comparison across model versions for native gradient/update sketches; explicit ablation")

def validate(a):
    if a.sync_policy=="legacy":
        return
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
    if a.sync_policy in ("random","oldest") and a.sync_mode!="fixed":
        raise ValueError("random and oldest require fixed count")
    if a.auto_break or a.client_drop_rate or a.train_slow_rate or a.send_slow_rate or a.time_select or a.num_new_clients or a.privacy:
        raise ValueError("legacy dropout, early stopping, DP and new-client options not supported by research engine")
    if min(a.sync_byte_cap,a.sync_campaign_byte_cap)<0:
        raise ValueError("invalid byte cap")
    if (a.sync_byte_cap or a.sync_campaign_byte_cap) and (a.sync_posttrain or a.sync_failures or a.sync_policy=="adaptive"):
        raise ValueError("hard caps currently require reliable pre-training delivery")
    if a.memory_size<0 or a.batch_size<1 or a.memory_batch_size<1 or a.local_epochs<1:
        raise ValueError("invalid local workload")
    if a.sync_sketch_dim<1:
        raise ValueError("invalid sketch dimension")
    if a.sync_count_schedule and (a.sync_policy!="random" or a.sync_byte_cap or a.sync_campaign_byte_cap):
        raise ValueError("count schedule requires random policy without byte cap")
    if a.sync_policy=="adaptive" and (not a.sync_impact_model or a.sync_mode!="fixed" or a.sync_signal!="histogram"):
        raise ValueError("adaptive requires calibrated impact model, histogram and fixed upper cap")
    if min(a.sync_current_tolerance,a.sync_past_tolerance)<0:
        raise ValueError("negative impact tolerance")
    if a.sync_policy in ("cached","update_match") and a.sync_mode!="fixed":
        raise ValueError("cached selectors require fixed count")
    if a.sync_policy=="update_match" and (a.sync_signal!="update" or a.sync_posttrain or not a.sync_native_space):
        raise ValueError("cached update matching requires update signal, native coordinates, and pre-training selection")
    if a.sync_signal=="update" and not a.sync_posttrain and a.sync_policy!="update_match":
        raise ValueError("update statistic requires post-training selection")
    if a.sync_signal not in ("histogram","replay") and a.sync_bank_reference=="envelope":
        raise ValueError("coordinate envelope only supported for histogram/replay")
    if a.sync_posttrain and (a.sync_refresh or a.sync_counterfactual):
        raise ValueError("post-training admission cannot be combined with refresh or pre-training counterfactual")
    if a.sync_windowed and a.few_shot:
        raise ValueError("windowed few-shot indexing is unsupported")
    if a.sync_stop_after_round is not None and (not a.sync_checkpoint or not 0<=a.sync_stop_after_round<=a.global_rounds):
        raise ValueError("pause requires valid round and checkpoints")
    if a.sync_resume and not a.sync_checkpoint:
        raise ValueError("resume requires checkpoints")
