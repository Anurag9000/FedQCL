"""Reliable pre-training hard byte limits using the actual logical serializer.
Future model reserve is a necessary lower bound, not a guarantee of feasibility
under all future telemetry costs or simultaneous deadlines.
"""
import copy

def future_model_reserve(ids,last,roster,r,end,gap,payload):
    return 2*payload*sum(max(0,(end-(r if i in roster else last[i]))//(gap+1)) for i in ids)

def predicted(controller,r,reports,payload,k):
    c=copy.deepcopy(controller)
    c.args.sync_mode="fixed";c.args.sync_k=k
    c.args.sync_soft_budget=False
    offset=len(c.transport.events)
    roster=c.decide(r,reports)
    starts={i:c.local[i].model_version for i in roster}
    for i in roster:
        if c.args.sync_refresh and c.local[i].model_version!=c.model_version:
            c.transport.send(r,"refresh","server",i,"MODEL_REFRESH",{"version":c.model_version},payload)
            starts[i]=c.model_version
        c.transport.send(r,"upload",i,"server","MODEL_UPLOAD",
                         {"start_version":starts[i],"accepted":True},payload)
    c.ingest_uploads(r,reports,roster,roster)
    if roster:c.model_version+=1
    for i in roster:
        c.transport.send(r,"download","server",i,"MODEL_DOWNLOAD",{"version":c.model_version},payload)
    c.complete(r,reports,roster,roster,roster,ingested=True)
    return roster,sum(x["serialized_bytes"] for x in c.transport.events[offset:])

def choose_budgeted(controller,r,reports,payload,limit,campaign_remaining=None):
    a=controller.args;n=len(controller.ids)
    requested=n if a.sync_policy=="full" or r==0 or (a.sync_policy=="adaptive" and a.sync_soft_budget) else a.sync_k
    choices=[requested] if r==0 or a.sync_policy=="full" else range(requested,-1,-1)
    last={i:controller.local[i].last_sync for i in controller.ids}
    for k in choices:
        if a.sync_policy=="adaptive" and k==0:continue
        expected,cost=predicted(controller,r,reports,payload,k)
        reserve=0
        if campaign_remaining is not None:
            if any(r-last[i]>a.sync_max_skip and i not in expected for i in controller.ids):
                continue
            reserve=future_model_reserve(controller.ids,last,expected,r,a.global_rounds,a.sync_max_skip,payload)
        if cost<=limit and (campaign_remaining is None or cost+reserve<=campaign_remaining):
            old=(a.sync_mode,a.sync_k,getattr(a,"sync_soft_budget",False))
            try:
                a.sync_mode="fixed";a.sync_k=k;a.sync_soft_budget=False
                selected=controller.decide(r,reports)
            finally:a.sync_mode,a.sync_k,a.sync_soft_budget=old
            if selected!=expected:raise RuntimeError("budgeted roster differs from prediction")
            return selected,{"limit":limit,"estimated_bytes":cost,"admitted_k":len(selected),
                             "requested_k":requested,"budget_limited":len(selected)<requested,
                             "future_model_reserve_lower_bound":reserve}
    raise RuntimeError(f"hard byte cap infeasible at round {r}: bootstrap, deadline, reserve or telemetry cannot fit")
