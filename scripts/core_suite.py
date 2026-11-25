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

def read_core_signal_entries():
    import json
    from pathlib import Path
    rows=[json.loads(line) for line in (Path(__file__).resolve().parents[1]/"configs/sync/core_signal_ledger.jsonl").read_text().splitlines() if line.strip()]
    signals={"replay","gradients","activations","jensen_shannon","mutual_information","bayesian","cka","gradient_variance","fisher"}
    methods={"M1":"own","M2":"server","M3":"peer"}
    if len(rows)!=27 or len({x["id"] for x in rows})!=27 or {(x["signal"],x["method"]) for x in rows}!={(s,m) for s in signals for m in methods}:
        raise ValueError("core requires exactly nine signals crossed with three mechanisms")
    for x in rows:
        if x["policy"]!=methods[x["method"]] or (x["regime"],x["seed"],x["k"])!=("clustered",42,3):
            raise ValueError("core matrix changes a fixed condition")
    return rows
