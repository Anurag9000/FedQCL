"""Portable ridge calibration from fresh M2 development trajectories.
Split by setting and trajectory. No test metrics or privileged features.
The calibration error is an empirical held-setting residual, not coverage.
"""
import json,math
from pathlib import Path
import numpy as np

FEATURES=("intercept","analytical_score","risk","retention_ok","fraction_selected","round_progress")

def load_model(path):
    model=json.loads(Path(path).read_text())
    if model.get("schema")!=1 or model.get("features")!=list(FEATURES) or len(model["coef"])!=len(FEATURES) or not all(math.isfinite(v) for v in model["coef"]) or not math.isfinite(model["calibration_error"]) or model["calibration_error"]<0 or type(model["qualified"]) is not bool:raise ValueError("incompatible utility model")
    return model

def records(root):
    root=Path(root);cfg=json.loads((root/"config.json").read_text())
    if cfg.get("sync_utility")!="server" or cfg.get("sync_utility_model") or not cfg.get("sync_fresh_start") or cfg["sync_eval_split"]!="validation":raise ValueError("not analytical validation development")
    decisions={x["round"]:x for x in map(json.loads,(root/"decisions.jsonl").read_text().splitlines())}
    cf=[json.loads(x) for x in (root/"counterfactuals.jsonl").read_text().splitlines()]
    out=[]
    for r in sorted({x["round"] for x in cf}):
        group=[x for x in cf if x["round"]==r];full=next(x for x in group if x["size"]==cfg["num_clients"])
        outcomes={tuple(x["subset"]):x for x in group}
        for x in decisions[r]["utility"]["candidates"]:
            y=outcomes.get(tuple(x["subset"]))
            if y is None:continue
            target=(full["current_loss"]-y["current_loss"])+(full["past_loss"] or 0.)-(y["past_loss"] or 0.)
            out.append({"group":str(root),"round":r,"subset":x["subset"],"features":x["features"],"target":target})
    return out

def fit(train,cal):
    if not train or not cal or {x["group"] for x in train}&{x["group"] for x in cal}:raise ValueError("independent calibration groups required")
    X=np.array([x["features"] for x in train]);y=np.array([x["target"] for x in train])
    penalty=np.eye(X.shape[1]);penalty[0,0]=0.
    coef=np.linalg.solve(X.T@X+penalty,X.T@y)
    by={}
    for x in cal:by.setdefault((x["group"],x["round"],len(x["subset"])),[]).append(x)
    gains=[];analytic=[];errors=[]
    for group in by.values():
        for x in group:errors.append(abs(float(coef@x["features"])-x["target"]))
        if len(group)<2:continue
        avg=np.mean([x["target"] for x in group])
        gains.append(float(max(group,key=lambda x:float(coef@x["features"]))["target"]-avg))
        analytic.append(float(max(group,key=lambda x:x["features"][1])["target"]-avg))
    if not gains:raise ValueError("no matched calibration subsets")
    # Prespecified qualification: better than uniform same-size expectation
    # and no worse than the analytical ranking on the held setting.
    gain=float(np.mean(gains));baseline=float(np.mean(analytic))
    return {"schema":1,"features":list(FEATURES),"coef":coef.tolist(),"calibration_error":float(np.quantile(errors,.9)),
            "qualified":bool(gain>0 and gain>=baseline),"calibration_mean_gain":gain,"analytical_mean_gain":baseline,
            "calibration_checkpoints_by_size":len(gains),"training_rows":len(train),"calibration_rows":len(cal),
            "calibration_semantics":"held-setting empirical residual; not a calibrated confidence guarantee"}
