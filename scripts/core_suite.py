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

CORE_SIGNAL_COUNT=24
PUBLIC_PROBE_PER_CLASS=4
CORE_REGIME='controlled_dynamic'
CORE_RUNTIME_FLAGS=('--sync-windowed','--sync-controlled-stream','--sync-audit-references','--sync-counterfactual','--sync-charge-control','--sync-bandwidth-audit')
def read_core_signal_entries():
    import json
    from pathlib import Path
    rows=[json.loads(line) for line in (Path(__file__).resolve().parents[1]/"configs/sync/core_signal_ledger.jsonl").read_text().splitlines() if line.strip()]
    signals={"activation_distribution":("tier_a_features",[]),
             "functional_predictions":("tier_a_functional",[]),
             "replay_loss_forgetting":("tier_a_replay_loss",[]),
             "gradient_interference":("tier_a_interference",["--sync-native-space"]),
             "class_conditional_prototypes":("tier_a_prototypes",[]),
             "current_historical_js":("tier_a_distributions",[]),
             "model_update_similarity":("tier_a_cached_update",["--sync-native-space"]),
             "replay_aware_gradients":("tier_a_gradients",["--sync-native-space"])}
    methods={"M1":"own","M2":"server","M3":"peer"}
    mechanism_flags={"M1":["--sync-local-reference","last_sync"],"M2":["--sync-bank-reference","aggregate"],"M3":["--sync-peer-transport","direct"]}
    if len(rows)!=CORE_SIGNAL_COUNT or len({x["id"] for x in rows})!=CORE_SIGNAL_COUNT or {(x["signal"],x["method"]) for x in rows}!={(s,m) for s in signals for m in methods}:
        raise ValueError("core requires exactly eight high-priority signal families crossed with three mechanisms")
    for x in rows:
        extractor,options=signals[x["signal"]]
        if x["flags"]!=["--sync-signal",extractor,*options,*mechanism_flags[x["method"]]]:
            raise ValueError("core signal or mechanism flags differ from fixed contract")
        if x["policy"]!=methods[x["method"]] or (x["regime"],x["seed"],x["k"])!=(CORE_REGIME,42,3):
            raise ValueError("core matrix changes a fixed condition")
    return rows
