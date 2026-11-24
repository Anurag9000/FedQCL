#!/usr/bin/env python3
"""Predeclare paired-study sample size from development-only differences.
Normal approximation for a one-sided noninferiority test; not a power guarantee.
"""
import argparse,json,math,statistics
from statistics import NormalDist
from pathlib import Path
def plan(values,margin,alpha,power,assumed_mean=0.):
    if len(values)<2 or not all(math.isfinite(x) for x in values):raise ValueError("at least two finite development pairs")
    if not 0<alpha<.5 or not .5<power<1 or margin<=0 or margin+assumed_mean<=0:raise ValueError("invalid design")
    sd=statistics.stdev(values)
    n=max(2,math.ceil(((NormalDist().inv_cdf(1-alpha)+NormalDist().inv_cdf(power))*sd/(margin+assumed_mean))**2))
    return {"development_pairs":len(values),"paired_sd":sd,"margin":margin,"alpha":alpha,"power_target":power,"assumed_mean":assumed_mean,"planned_pairs":n,"method":"one-sided paired normal approximation","warning":"Independent seed groups required; pilot variance uncertain; adjust multiplicity before confirmatory execution."}
def main():
    p=argparse.ArgumentParser();p.add_argument("--development-differences",type=Path,required=True);p.add_argument("--margin",type=float,required=True);p.add_argument("--alpha",type=float,default=.05);p.add_argument("--power",type=float,default=.8);p.add_argument("--assumed-mean",type=float,default=0.);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=plan(json.loads(a.development_differences.read_text()),a.margin,a.alpha,a.power,a.assumed_mean)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
if __name__=="__main__":main()
