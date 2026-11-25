"""Pure policy functions. No model, loader, server or raw data access."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import random
import math

@dataclass(frozen=True)
class Report:
    client: int
    task: int
    round: int
    labels: tuple
    counts: tuple
    window: str = "static"
    schema: int = 1
    model_version: int = 0
    kind: str = "histogram"
    space: str = "task_labels"
    sample_count: int = 0
    metric: str = "tv"
    histogram_distance: str = "tv"
    importance: float = 1.0
    aggregation_weight: float = 1.0
    quant_scale: float = 0.0

    @property
    def id(self):
        return f"{self.client}:{self.task}:{self.round}:{self.window}:{self.schema}"

    def valid(self, minimum=1):
        basic=(self.schema == 1 and len(self.labels) == len(self.counts)
               and bool(self.labels) and len(set(self.labels)) == len(self.labels)
               and math.isfinite(self.importance) and self.importance>0
               and math.isfinite(self.aggregation_weight) and self.aggregation_weight>0)
        if not basic or not math.isfinite(self.quant_scale) or self.quant_scale<0 or self.histogram_distance not in ("tv","js","hellinger"):
            return False
        if self.metric=="tv":
            return (all(type(x) is int and x>=0 for x in self.counts)
                    and sum(self.counts)>=minimum)
        valid=(self.metric in ("cosine","bounded_l2","channels","cka","tier_a") and self.sample_count>=minimum
                and all(isinstance(x,(float,int)) and math.isfinite(x) for x in self.counts))
        if valid and self.metric=="tier_a":
            from .tier_a_reports import valid_layout
            return valid_layout(self)
        if valid and self.metric=="channels":
            from .multichannel import valid_layout
            return valid_layout(self)
        return valid

def distance(a, b):
    if (a is None or b is None or not a.valid() or not b.valid()
            or (a.task!=b.task and a.kind!="replay_composition")
            or (a.labels,a.schema,a.kind,a.space,a.metric,a.histogram_distance) != (b.labels,b.schema,b.kind,b.space,b.metric,b.histogram_distance)):
        return None
    if a.metric=="tier_a":
        from .tier_a_reports import report_distance
        return report_distance(a,b)
    if a.metric=="channels":
        from .multichannel import channel_distance
        return channel_distance(a,b)
    if a.metric in ("cosine","cka"):
        na=math.sqrt(sum(x*x for x in a.counts));nb=math.sqrt(sum(x*x for x in b.counts))
        if na==0 or nb==0:
            return 0.0 if na==nb==0 else 1.0
        cosine=sum(x*y for x,y in zip(a.counts,b.counts))/(na*nb)
        return max(0.,min(1.,1-cosine if a.metric=="cka" else (1-cosine)/2))
    if a.metric=="bounded_l2":
        d=math.sqrt(sum((x-y)**2 for x,y in zip(a.counts,b.counts)))
        return d/(1+d)
    if a.histogram_distance!="tv":
        from .statistics import js,hellinger
        return (js if a.histogram_distance=="js" else hellinger)(a.counts,b.counts)
    na, nb = sum(a.counts), sum(b.counts)
    return sum(abs(Fraction(x, na)-Fraction(y, nb))
               for x, y in zip(a.counts, b.counts))/2

def threshold_value(value):
    return Fraction(str(value))

def priority(i, last):
    return (last.get(i, -1), i)

def due_clients(ids, last, r, gap):
    return tuple(sorted((i for i in ids if r-last.get(i, -1) > gap),
                        key=lambda i: priority(i, last)))

def local_reference(current, last_report, history, kind, length):
    if kind == "last_sync":
        return last_report
    if kind == "previous":
        return history[-1] if history else None
    candidates = [x for x in history[-length:] if distance(current, x) is not None]
    return pooled(candidates, current) if candidates else None

def pooled(records, template, weighted=True):
    if not records:
        return None
    if template.metric=="tier_a":
        from .tier_a_reports import pooled_report
        return pooled_report(records,template,weighted)
    if template.metric=="channels":
        raise ValueError("pooling incompatible with channel-presence masks")
    if template.metric!="tv":
        from dataclasses import replace
        weights=[x.sample_count if weighted else 1 for x in records]
        values=tuple(sum(x.counts[c]*w for x,w in zip(records,weights))/sum(weights) for c in range(len(template.labels)))
        return replace(template,client=-1,round=max(x.round for x in records),counts=values,window="pooled",sample_count=sum(weights))
    if weighted:
        counts = tuple(sum(x.counts[c] for x in records) for c in range(len(template.labels)))
    else:
        from math import lcm
        denom = lcm(*(sum(x.counts) for x in records))
        counts = tuple(sum(x.counts[c]*(denom//sum(x.counts)) for x in records)
                       for c in range(len(template.labels)))
    return Report(-1, template.task, max(x.round for x in records),
                  template.labels, counts, "pooled",kind=template.kind,space=template.space,metric=template.metric,histogram_distance=template.histogram_distance)

def evidence_round(report):
    """Cached update age is measured from training, not the report resend."""
    if report.kind=="tier_a_cached_update" and report.window.startswith("cached_update:"):
        return int(report.window.split(":",1)[1])
    return report.round

def bank_score(report, bank, round_id, age=5, include_self=False, kind="latest"):
    records = [x for x in bank if (include_self or x.client != report.client)
               and 0 <= round_id-evidence_round(x) <= age and distance(report,x) is not None]
    if kind=="aggregate":
        records=[x for x in records if x.client==-1]
        if not records:return None,None
        ref=max(records,key=lambda x:(x.task==report.task,x.round,x.task))
        return distance(report,ref),ref.id
    if kind == "prior":
        records = [x for x in records if x.round == round_id-1]
    if kind == "latest":
        latest = {}
        for x in sorted(records, key=lambda x:(x.round,x.id)):
            latest[x.client] = x
        records = list(latest.values())
    if not records:
        return None, None
    if kind in ("mean", "weighted_mean"):
        ref = pooled(records, report, weighted=kind=="weighted_mean")
        return distance(report,ref), ref.id
    if kind == "envelope":
        p = [Fraction(x,sum(report.counts)) for x in report.counts]
        cols = [[Fraction(x.counts[c],sum(x.counts)) for x in records]
                for c in range(len(p))]
        return sum(max(0,min(col)-v,v-max(col)) for col,v in zip(cols,p))/2, "envelope"
    ref = min(records, key=lambda x:(distance(report,x),x.client,x.round,x.id))
    return distance(report,ref), ref.id

def novelty_admit(ids, last, due, scores, threshold, k=None):
    ids = tuple(sorted(ids))
    mandatory = list(due if k is None else due[:k])
    remaining = [i for i in ids if i not in mandatory and i not in due]
    remaining.sort(key=lambda i:(scores.get(i) is not None,
                                 -(scores.get(i) or 0),priority(i,last)))
    if k is None:
        chosen = mandatory+[i for i in remaining
                            if scores.get(i) is None or scores[i]>threshold]
    else:
        chosen = mandatory+remaining[:max(0,k-len(mandatory))]
    return tuple(sorted(chosen))

def nearest(i, selected, reports):
    options = [(distance(reports.get(i),reports.get(j)),j) for j in selected]
    options = [(d,j) for d,j in options if d is not None]
    return min(options) if options else (None,None)

def coverage(ids, selected, reports, threshold):
    result = {}
    for i in ids:
        if i not in selected:
            d,j = nearest(i,selected,reports)
            result[i] = {"representative":j,"distance":None if d is None else float(d),
                         "covered":d is not None and d<=threshold}
    return result

def peer_select(ids, last, due, reports, threshold, k=None, solver="exact", objective="distance", weighted=False):
    ids = tuple(sorted(ids))
    if k is None:
        selected = list(due)
        for i in sorted(ids,key=lambda i:priority(i,last)):
            if i not in selected:
                d,_ = nearest(i,selected,reports)
                if d is None or d>threshold:
                    selected.append(i)
        return tuple(sorted(selected))
    k = min(k,len(ids))
    mandatory = tuple(due[:k])
    remaining = tuple(i for i in ids if i not in mandatory)
    def cost(s):
        total = 0
        missing = 0
        for i in ids:
            if i in s:
                continue
            if reports.get(i) is None or not reports[i].valid():
                missing += 1
                continue
            d,_ = nearest(i,s,reports)
            weight=reports[i].importance if weighted else 1
            total += weight*(int(d is None or d>threshold) if objective=="coverage"
                      else 1 if d is None else d)
        return missing,total,tuple(sorted(priority(i,last) for i in s))
    if solver == "greedy":
        selected = list(mandatory)
        while len(selected)<k:
            selected.append(min((i for i in remaining if i not in selected),
                                key=lambda i:cost(selected+[i])))
        return tuple(sorted(selected))
    return tuple(sorted(min((mandatory+s for s in combinations(remaining,k-len(mandatory))),
                            key=cost)))

def baseline_select(policy, ids, due, last, k, seed, round_id):
    if policy == "full":
        return tuple(sorted(ids))
    selected = list(due[:k])
    remaining = [i for i in sorted(ids) if i not in due]
    if policy == "random":
        random.Random(f"{seed}:selection:{round_id}").shuffle(remaining)
    else:
        remaining.sort(key=lambda i:priority(i,last))
    return tuple(sorted(selected+remaining[:k-len(selected)]))
