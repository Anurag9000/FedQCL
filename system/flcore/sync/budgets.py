"""Deterministic hard logical-byte admission, including policy telemetry.
Simulates the declared reliable exchange using the exact canonical serializer;
it does not run local training or inspect evaluation outcomes.
"""
import copy

def predicted(controller,r,reports,payload,k):
    c=copy.deepcopy(controller)
    c.args.sync_mode="fixed"
    c.args.sync_k=k
    offset=len(c.transport.events)
    roster=c.decide(r,reports)
    starts={i:c.local[i].model_version for i in roster}
    for i in roster:
        if c.args.sync_refresh and c.local[i].model_version!=c.model_version:
            c.transport.send(r,"refresh","server",i,"MODEL_REFRESH",{"version":c.model_version},payload)
            starts[i]=c.model_version
        c.transport.send(r,"upload",i,"server","MODEL_UPLOAD",
                         {"start_version":starts[i],"accepted":True},payload)
    if roster:
        c.model_version+=1
    for i in roster:
        c.transport.send(r,"download","server",i,"MODEL_DOWNLOAD",{"version":c.model_version},payload)
    c.complete(r,reports,roster,roster,roster)
    return roster,sum(x["serialized_bytes"] for x in c.transport.events[offset:])

def choose_budgeted(controller,r,reports,payload,limit):
    # Fixed-cardinality variants are compared under a byte cap. No future
    # reports, candidate updates or validation data are involved.
    n=len(controller.ids)
    requested=n if controller.args.sync_policy=="full" or r==0 else controller.args.sync_k
    choices=[requested] if r==0 or controller.args.sync_policy=="full" else range(requested,-1,-1)
    for k in choices:
        expected,cost=predicted(controller,r,reports,payload,k)
        if cost<=limit:
            old_mode,old_k=controller.args.sync_mode,controller.args.sync_k
            try:
                controller.args.sync_mode="fixed";controller.args.sync_k=k
                selected=controller.decide(r,reports)
            finally:
                controller.args.sync_mode=old_mode;controller.args.sync_k=old_k
            if selected!=expected:
                raise RuntimeError("budgeted roster differs from prediction")
            return selected,{"limit":limit,"estimated_bytes":cost,"admitted_k":len(selected),
                             "requested_k":requested,"budget_limited":len(selected)<requested}
    raise RuntimeError(f"hard byte cap infeasible at round {r}: mandatory bootstrap or telemetry cannot fit")
