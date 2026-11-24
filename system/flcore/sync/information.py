"""Discrete plug-in information estimators, natural logarithm units.
Finite-sample estimates are biased; no causal inference or significance is implied.
"""
from collections import Counter
import math

def _rows(*columns):
    if not columns or not len(columns[0]) or any(len(c)!=len(columns[0]) for c in columns):
        raise ValueError("aligned nonempty observations required")
    return list(zip(*columns))

def joint_entropy(*columns):
    rows=_rows(*columns);n=len(rows)
    return -sum((c/n)*math.log(c/n) for c in Counter(rows).values())

def mutual_information(x,y):
    return max(0.,joint_entropy(x)+joint_entropy(y)-joint_entropy(x,y))

def conditional_mi(x,y,z):
    return max(0.,joint_entropy(x,z)+joint_entropy(y,z)-joint_entropy(z)-joint_entropy(x,y,z))

def total_correlation(*columns):
    return max(0.,sum(joint_entropy(c) for c in columns)-joint_entropy(*columns))

def interaction_information(x,y,z):
    # Signed co-information, not a uniquely identified partial information decomposition.
    return mutual_information(x,y)-conditional_mi(x,y,z)

def transfer_entropy(source,target,lag=1):
    if len(source)!=len(target) or lag<1 or len(source)<=lag:
        raise ValueError("invalid lagged observations")
    return conditional_mi(source[:-lag],target[lag:],target[:-lag])

def bin_unit(values,bins=4):
    if bins<2 or any(not math.isfinite(float(x)) or x<0 or x>1 for x in values):
        raise ValueError("invalid bounded observations")
    return [min(bins-1,int(x*bins)) for x in values]
