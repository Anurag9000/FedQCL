"""Prespecified economical research matrix; no Cartesian expansion at runtime.

Select complete dependency closures from the existing, fixed-backbone catalog.
Retain exact source configurations and matched baselines; never tune on results.
"""
CORE_TASK_IL_IDS=(
    "E01-01","E01-03","E01-05","E01-07",
    "E02-01","E02-03",
    "E03-02","E03-06",
    "E04-02",
    "E05-01","E05-03","E05-05","E05-07",
    "E06-02",
    "E07-01","E07-05",
    "E08-01","E08-03",
    "E09-01","E09-05",
    "E10-01","E10-03",
)

def core_sync_jobs(jobs):
    by={j["id"]:j for j in jobs}
    chosen=set()
    for j in jobs:
        if j["kind"]!="train":continue
        c=j["config"];stage=j["stage"];policy=c["policy"]
        regime=c["regime"];seed=c["partition_seed"];flags=c["flags"];k=c["k"]
        if stage=="pilot":
            if (regime in ("near","separated") and seed in (42,43)) or (regime=="changing" and seed==42 and policy!="oldest"):
                chosen.add(j["id"])
            continue
        if not (regime=="near" and seed==42):continue
        keep=(
            (stage=="budget" and policy in ("random","peer") and k==2)
            or (stage=="threshold" and flags==["--sync-mode","threshold","--sync-threshold","0.1"])
            or (stage=="ablation" and (policy,tuple(flags)) in (
                ("own",("--sync-local-reference","previous")),
                ("server",("--sync-bank-reference","mean")),
                ("peer",("--sync-peer-transport","relay"))))
            or (stage=="mechanism" and policy=="peer" and flags==["--sync-refresh"])
            or (stage=="counterfactual" and policy=="peer")
            or (stage=="signals" and policy=="peer" and flags[1] in ("replay","feature","gradient"))
            or (stage=="advanced" and policy in ("cached","combined123") and not flags)
            or (stage=="robustness" and policy=="peer")
            or (stage=="hard_bytes" and policy in ("random","peer") and flags==["--sync-byte-cap","280000000","--sync-bootstrap-byte-cap","500000000"])
        )
        if keep:chosen.add(j["id"])
    # One matched-count and one matched-byte random comparator use the same
    # predeclared threshold-peer source. No data-dependent winner selection.
    sources={j["id"] for j in jobs if j["id"] in chosen and j["stage"]=="threshold" and j["config"]["policy"]=="peer"}
    if len(sources)!=1:raise ValueError("core matched baseline source must be unique")
    for j in jobs:
        if j["kind"]=="train" and j["stage"] in ("rate_matched","byte_matched"):
            if any(by[d]["kind"]=="schedule" and by[d]["config"].get("source") in sources for d in j["dependencies"]):
                chosen.add(j["id"])
        elif j["kind"]=="analysis" and j["stage"]=="counterfactual_analysis" and set(j["dependencies"])<=chosen:
            chosen.add(j["id"])
    if sum(by[i]["kind"]=="train" for i in chosen)!=50:
        raise ValueError("unexpected core synchronization training count")
    stack=list(chosen)
    while stack:
        for d in by[stack.pop()]["dependencies"]:
            if d not in chosen:chosen.add(d);stack.append(d)
    return [j for j in jobs if j["id"] in chosen]
