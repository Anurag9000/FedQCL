#!/usr/bin/env python3
"""Fit ridge harm predictors, calibrate upper residual on held-out DEVELOPMENT seeds.
No final-test data are used. Calibration uses maximum per-run residual (not iid rows).
"""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.impact import FEATURES,feature_map

def fit(train,cal,alpha,coverage,mode="linear",ood_guard=False):
    result={}
    for kind in ("current","past"):
        target=kind+"_harm"
        rows=[x for _,x in train if x[target] is not None]
        if not rows:raise ValueError("missing training targets")
        X=np.asarray([feature_map(x["features"],mode) for x in rows]);y=np.asarray([x[target] for x in rows])
        penalty=np.eye(X.shape[1])*alpha;penalty[0,0]=0.
        coef=np.linalg.lstsq(X.T@X+penalty,X.T@y,rcond=None)[0]
        by_group={}
        for group,x in cal:
            if x[target] is not None:
                residual=x[target]-float(np.dot(coef,feature_map(x["features"],mode)))
                by_group[group]=max(by_group.get(group,-math.inf),residual)
        residuals=sorted(by_group.values())
        if not residuals:raise ValueError("missing calibration targets")
        # Finite-sample conformal bound may be infinite at requested coverage.
        rank=math.ceil((len(residuals)+1)*coverage)
        if rank>len(residuals):
            raise ValueError(f"insufficient independent calibration groups for coverage {coverage}")
        result[kind]={"coef":coef.tolist(),"upper_residual":max(0.,residuals[rank-1]),
                      "calibration_groups":len(residuals),"fit_rows":len(rows)}
    values=np.asarray([x["features"] for _,x in train])
    return {"schema":2,"features":list(FEATURES),"coverage":coverage,"targets":result,"feature_map":mode,
            "fit_bounds":np.stack((values.min(0),values.max(0)),1).tolist() if ood_guard else None}

def main():
    p=argparse.ArgumentParser();p.add_argument("--ledger",type=Path,default=Path("experiments/job_ledger.jsonl"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--fit-seeds",nargs="+",type=int,required=True)
    p.add_argument("--cal-seeds",nargs="+",type=int,required=True);p.add_argument("--alpha",type=float,default=1.)
    p.add_argument("--coverage",type=float,default=.5);p.add_argument("--data-stage",default="impact_data")
    p.add_argument("--feature-map",choices=["linear","quadratic"],default="linear");p.add_argument("--ood-guard",action="store_true");a=p.parse_args()
    if set(a.fit_seeds)&set(a.cal_seeds) or not 0<a.coverage<1 or a.alpha<=0:raise ValueError("invalid split or fit settings")
    train=[];cal=[]
    for line in a.ledger.read_text().splitlines():
        j=json.loads(line)
        if j["stage"]!=a.data_stage or j["kind"]!="train":continue
        if j["config"].get("evaluation_split")!="validation":raise ValueError("impact fitting cannot use test runs")
        seed=j["config"]["partition_seed"]
        dest=train if seed in a.fit_seeds else cal if seed in a.cal_seeds else None
        if dest is None:continue
        out=Path(j["artifacts"][0]).parent
        if json.loads((out/"status.json").read_text())["status"]!="completed":raise ValueError("incomplete development run")
        dest.extend((seed,json.loads(line)) for line in (out/"counterfactuals.jsonl").read_text().splitlines())
    if {g for g,_ in train}!=set(a.fit_seeds) or {g for g,_ in cal}!=set(a.cal_seeds):raise ValueError("missing independent seed group")
    model=fit(train,cal,a.alpha,a.coverage,a.feature_map,a.ood_guard)
    model.update(fit_seeds=a.fit_seeds,cal_seeds=a.cal_seeds,alpha=a.alpha)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(model,indent=2))
if __name__=="__main__":main()
