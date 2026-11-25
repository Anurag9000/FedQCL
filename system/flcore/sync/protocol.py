"""Logical transport, versioned local state, and pre-training admission."""
from dataclasses import dataclass, asdict, field
import json
import random
import hashlib
from .advanced import cached_views, select_cached, update_match, combine_scores, peer_filter
from .policies import (Report, distance, threshold_value, due_clients, above_threshold,
                       local_reference, bank_score, novelty_admit, peer_select, pooled,
                       baseline_select, coverage)

def encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(",",":"),
                      allow_nan=False).encode("utf-8")

@dataclass
class LocalState:
    last_sync: int = -1
    model_version: int = 0
    local_revision: int = 0
    last_report: Report | None = None
    history: list = field(default_factory=list)
    bank_version: int = 0
    bank: tuple = ()
    audit_bank_sources: dict = field(default_factory=dict)

class Transport:
    """In-process reliable logical deliveries, per-recipient accounting."""
    def __init__(self,wire_format="json",audit=False):
        self.audit=audit
        self.events = []
        self.wire_format=wire_format
    def send(self, r, phase, sender, receiver, kind, payload, tensor_bytes=0, delivered=True):
        envelope = {"round":r,"phase":phase,"sender":sender,"receiver":receiver,
                    "kind":kind,"payload":payload}
        if self.wire_format in ("binary","fp16","int8"):
            from .wire import pack,unpack
            wire=pack(envelope,self.wire_format)
            decoded=unpack(wire)
            payload_size=len(pack(payload,self.wire_format))
        else:
            wire=encoded(envelope)
            decoded=json.loads(wire)
            payload_size=len(encoded(payload))
        n = len(wire)+tensor_bytes
        from .bandwidth import category,direction
        from .wire import binary_vector_bytes
        vector_bytes=binary_vector_bytes(wire) if self.wire_format!="json" else 0
        if type(tensor_bytes) is not int or tensor_bytes<0:raise ValueError("invalid tensor payload bytes")
        self.events.append({"event_id":len(self.events),"round":r,"phase":phase,
                            "sender":sender,"receiver":receiver,"kind":kind,
                            "payload_bytes":tensor_bytes+payload_size,"wire_format":self.wire_format,
                            "serialized_bytes":n,"delivered":delivered,
                            "category":category(kind),"direction":direction(sender,receiver),
                            "tensor_bytes":tensor_bytes,"vector_bytes":vector_bytes,
                            "envelope_bytes":len(wire)-vector_bytes,"encoded_envelope_bytes":len(wire),
                            "scalar_admission_score":kind=="ADMISSION_REQUEST" and isinstance(payload,dict) and payload.get("score") is not None,
                            "delivered_bytes":n if delivered else 0})
        if getattr(self,"audit",False):
            import base64
            self.events[-1].update(encoded_envelope_b64=base64.b64encode(wire).decode("ascii"),envelope_sha256=hashlib.sha256(wire).hexdigest())
        return decoded["payload"] if delivered else None

def observe_local(a,i,task,r,dataset,model_version):
    n = min(a.sync_observation_size,len(dataset))
    labels = tuple(range(task*a.num_classes,(task+1)*a.num_classes))
    counts = dict.fromkeys(labels,0)
    population = dict.fromkeys(labels,0)
    for _,y in dataset:
        y = int(y)
        if y not in population:
            raise ValueError("observation label outside task")
        population[y]+=1
    rng = random.Random(f"{a.seed}:observation:{i}:{task}:{r}")
    epsilon=getattr(a,"sync_histogram_rr_epsilon",0.)
    from math import exp
    rr=random.Random(f"{a.seed}:rr:{i}:{task}:{r}")
    for idx in rng.sample(range(len(dataset)),n):
        y=int(dataset[idx][1])
        if epsilon and len(labels)>1 and rr.random()>=exp(epsilon)/(exp(epsilon)+len(labels)-1):
            y=rr.choice([label for label in labels if label!=y])
        counts[y]+=1
    report = Report(i,task,r,labels,tuple(counts.values()),
                    model_version=model_version,histogram_distance=getattr(a,"sync_histogram_distance","tv"),
                    space=f"task_labels_rr:{epsilon}" if epsilon else "task_labels")
    return (report if report.valid(a.sync_min_observation) else None,
            tuple(population.values()))


class Controller:
    def __init__(self, args, ids):
        self.args = args
        self.ids = tuple(sorted(ids))
        self.local = {i:LocalState() for i in self.ids}
        self.transport = Transport(getattr(args,"sync_wire_format","json"),getattr(args,"sync_bandwidth_audit",False))
        self.bank = {}
        self.bank_version = 0
        self.bank_snapshots = {0:()}
        self.model_version = 0
        self.arrivals = []
        self.decisions = []
        self.score_history=[]
        self.delivered_history=[]
        self.impact_model=None
        if getattr(args,"sync_policy",None)=="adaptive":
            from pathlib import Path
            self.impact_model=json.loads(Path(args.sync_impact_model).read_text())

    def observe(self,i,task,r,dataset):
        return observe_local(self.args,i,task,r,dataset,self.local[i].model_version)

    def ingest(self, reports, r, accepted, observed=False):
        changed = False
        for i,report in sorted(reports.items()):
            if report is None or not report.valid() or report.client!=i or report.round!=r:
                continue
            self.arrivals.append({"round":r,"report":asdict(report),
                                  "accepted":i in accepted,"observed_only":observed})
            if observed or i in accepted:
                key = (i,report.task,report.round if self.args.sync_bank_reference in ("history","prior") else -1)
                self.bank[key] = report
                if not any(x.id==report.id for x in self.delivered_history):self.delivered_history.append(report)
                changed = True
        self.delivered_history=[x for x in self.delivered_history if 0<=r-x.round<=max(2,self.args.sync_bank_age)]
        previous_size=len(self.bank)
        self.bank = {k:v for k,v in self.bank.items() if r-v.round<=self.args.sync_bank_age}
        if changed or len(self.bank)!=previous_size:
            self.bank_version += 1
            self.bank_snapshots[self.bank_version]=tuple(sorted(self.bank.values(),key=lambda x:(x.client,x.task,x.round)))
            self.bank_snapshots={k:v for k,v in self.bank_snapshots.items() if k>=self.bank_version-self.args.sync_history}

    def deliver_bank(self,r,recipients,phase):
        records = tuple(sorted(self.bank.values(),key=lambda x:(x.client,x.task,x.round)))
        for i in recipients:
            local=self.local[i]
            if self.args.sync_bank_reference=="aggregate":
                from dataclasses import replace
                from .policies import evidence_round
                groups={}
                for x in records:
                    if (not self.args.sync_include_self and x.client==i) or not x.valid() or not 0<=r-evidence_round(x)<=self.args.sync_bank_age:continue
                    key=(x.task,x.labels,x.schema,x.kind,x.space,x.metric,x.histogram_distance)
                    group=groups.setdefault(key,{})
                    if x.client not in group or group[x.client].round<x.round:group[x.client]=x
                aggregates=[];audit_sources={}
                for group in groups.values():
                    members=list(group.values())
                    aggregate=pooled(members,members[0],weighted=True)
                    aggregate=replace(aggregate,round=min(evidence_round(x) for x in members))
                    aggregates.append(aggregate)
                    if getattr(self.args,"sync_audit_references",False):
                        audit_sources[aggregate.id]=[{"id":x.id,"client":x.client,"round":x.round,"source_round":evidence_round(x),"sample_count":x.sample_count} for x in members]
                payload=self.transport.send(r,phase,"server",i,"BANK_AGGREGATE",
                    {"version":self.bank_version,"records":[asdict(x) for x in aggregates]})
                local.bank=tuple(Report(**{**x,"labels":tuple(x["labels"]),"counts":tuple(x["counts"])}) for x in payload["records"])
                local.bank_version=payload["version"]
                local.audit_bank_sources=audit_sources
                continue
            if self.args.sync_bank_transfer=="delta" and local.bank_version in self.bank_snapshots:
                old={x.id:x for x in self.bank_snapshots[local.bank_version]}
                new={x.id:x for x in records}
                payload=self.transport.send(r,phase,"server",i,"BANK_DELTA",
                    {"base":local.bank_version,"version":self.bank_version,
                     "remove":sorted(old.keys()-new.keys()),
                     "upsert":[asdict(x) for key,x in new.items() if old.get(key)!=x]})
                if payload["base"]!=local.bank_version:
                    raise RuntimeError("bank delta base mismatch")
                current={x.id:x for x in local.bank if x.id not in payload["remove"]}
                for x in payload["upsert"]:
                    record=Report(**{**x,"labels":tuple(x["labels"]),"counts":tuple(x["counts"])})
                    current[record.id]=record
                local.bank=tuple(sorted(current.values(),key=lambda x:(x.client,x.task,x.round)))
            else:
                if self.args.sync_bank_transfer=="delta":
                    self.transport.send(r,phase,i,"server","BANK_CATCHUP_REQUEST",{"base":local.bank_version})
                payload = self.transport.send(r,phase,"server",i,"BANK_SNAPSHOT",
                    {"version":self.bank_version,"records":[asdict(x) for x in records]})
                local.bank = tuple(Report(**{**x,"labels":tuple(x["labels"]),
                                             "counts":tuple(x["counts"])}) for x in payload["records"])
            local.bank_version=payload["version"]

    def decide(self,r,reports,available=None):
        a = self.args
        policy=a.sync_policy
        own=policy in ("own","combined12","combined13","combined123")
        server=policy in ("server","combined12","combined23","combined123")
        peer=policy in ("peer","combined13","combined23","combined123")
        ids = self.ids if available is None else tuple(sorted(available))
        expected_task=r//((a.global_rounds+1)//a.tasks) if hasattr(a,"global_rounds") else None
        reports={i:(x if x is not None and x.client==i and x.round==r and x.valid()
                    and (expected_task is None or x.task==expected_task) else None) for i in ids for x in [reports.get(i)]}
        if getattr(a,"sync_charge_control",False):
            for i in self.ids:
                self.transport.send(r,"observe","server",i,"ROUND_CLOCK",{"task":expected_task,"version":self.model_version})
                if i in ids:self.transport.send(r,"observe",i,"server","AVAILABILITY",{"available":True})
            for i in ids:self.transport.send(r,"observe","server",i,"AVAILABLE_ROSTER",{"clients":ids,**({"last_sync":{j:self.local[j].last_sync for j in ids}} if peer and getattr(a,"sync_bandwidth_audit",False) else {})})
        last = {i:self.local[i].last_sync for i in ids}
        due = due_clients(ids,last,r,a.sync_max_skip)
        threshold = threshold_value(a.sync_threshold)
        if getattr(a,"sync_dynamic_threshold",False) and self.score_history:
            from statistics import median
            threshold=threshold_value(max(.01,min(.99,median(self.score_history))))
            for i in ids:self.transport.send(r,"observe","server",i,"DYNAMIC_THRESHOLD",{"threshold":float(threshold)})
        k = None if a.sync_mode=="threshold" else min(a.sync_k,len(ids))
        if hasattr(self,"count_schedule"):
            k=min(self.count_schedule[r],len(ids))
        scores, refs, assignments = {},{},{}
        impact=None
        multiplicity=None
        versions = {i:self.local[i].bank_version for i in ids}
        # Every available client sees the common clock and roster under a declared
        # free common-control convention; all additional messages below are charged.
        if server and a.sync_delivery=="telemetry":
            delivered = {}
            for i in ids:
                if reports[i] is not None:
                    msg = self.transport.send(r,"observe",i,"server","STAT_REPORT",asdict(reports[i]))
                    delivered[i] = Report(**{**msg,"labels":tuple(msg["labels"]),"counts":tuple(msg["counts"])})
            self.ingest(delivered,r,(),observed=True)
            self.deliver_bank(r,ids,"observe")
            versions = {i:self.local[i].bank_version for i in ids}
        if own:
            for i in ids:
                cur = reports[i]
                ref = local_reference(cur,self.local[i].last_report,self.local[i].history,
                                      a.sync_local_reference,a.sync_history) if cur is not None else None
                scores[i] = distance(cur,ref)
                refs[i] = ref.id if ref else None
        if server:
            for i in ids:
                cur = reports[i]
                score,ref = bank_score(cur,self.local[i].bank,r,a.sync_bank_age,
                    a.sync_include_self,a.sync_bank_reference) if cur is not None else (None,None)
                if own:
                    scores[i]=combine_scores(scores[i],score,getattr(a,"sync_combine_rule","union"))
                    refs[i]={"local":refs[i],"server":ref}
                else:
                    scores[i],refs[i]=score,ref
        if peer and policy!="peer":
            # Hybrid peer election is a charged centralized relay variant:
            # clients transmit scalar novelty, never their private history.
            for i in ids:
                self.transport.send(r,"observe",i,"server","HYBRID_SCORE",
                    {"score":None if scores[i] is None else float(scores[i])})
        if peer and a.sync_peer_transport=="distributed_sparse":
            from .distributed_peer import distributed_election
            selected,assignments=distributed_election(self.transport,ids,last,due,reports,r,threshold,k,
                a.sync_peer_degree,a.sync_peer_solver,a.sync_peer_objective,a.seed,a.sync_dynamic_topology,a.sync_peer_loss,a.sync_gossip_fault)
        elif peer and a.sync_peer_transport=="sparse":
            from .peer_network import sparse_election
            selected,assignments=sparse_election(self.transport,ids,last,due,reports,r,threshold,k,
                getattr(a,"sync_peer_degree",2),a.sync_peer_solver,a.sync_peer_objective)
            if r==0:selected=ids;assignments={}
        elif peer:
            views = {i:{i:reports[i]} for i in ids}
            if a.sync_peer_transport=="relay":
                arrived = {}
                for i in ids:
                    msg = self.transport.send(r,"observe",i,"server","PEER_STAT_REPORT",
                                              asdict(reports[i]) if reports[i] else None)
                    arrived[i] = msg
                self.relay_view={i:Report(**{**x,"labels":tuple(x["labels"]),"counts":tuple(x["counts"])}) if x else None for i,x in arrived.items()}
                for j in ids:
                    payload = self.transport.send(r,"observe","server",j,"PEER_ROSTER",arrived)
                    views[j] = {i:Report(**{**x,"labels":tuple(x["labels"]),"counts":tuple(x["counts"])})
                                if x else None for i,x in ((int(i),x) for i,x in payload.items())}
            else:
                for i in ids:
                    for j in ids:
                        if i==j:
                            continue
                        msg = self.transport.send(r,"observe",i,j,"PEER_STAT_REPORT",
                                                  asdict(reports[i]) if reports[i] else None,
                                                  delivered=random.Random(f"{a.seed}:peer:{r}:{i}:{j}").random()>=getattr(a,"sync_peer_loss",0.))
                        views[j][i] = Report(**{**msg,"labels":tuple(msg["labels"]),
                                                "counts":tuple(msg["counts"])}) if msg else None
            if getattr(a,"sync_peer_loss",0.)>0:
                # Digest exchange observes different views; if needed use a paid
                # reliable relay reconciliation before selection.
                digests=[]
                for i in ids:
                    digest=hashlib.sha256(encoded({j:asdict(x) if x else None for j,x in views[i].items()})).hexdigest()
                    payload=self.transport.send(r,"observe",i,"server","ROSTER_DIGEST",{"digest":digest})
                    digests.append(payload["digest"])
                if len(set(digests))>1:
                    arrived={}
                    for i in ids:
                        arrived[i]=self.transport.send(r,"reconcile",i,"server","PEER_RECOVERY_REPORT",asdict(reports[i]) if reports[i] else None)
                    for i in ids:
                        payload=self.transport.send(r,"reconcile","server",i,"PEER_RECOVERY_ROSTER",arrived)
                        views[i]={int(j):Report(**{**x,"labels":tuple(x["labels"]),"counts":tuple(x["counts"])}) if x else None for j,x in payload.items()}
            if policy=="peer":
                elections = [tuple(ids) if r==0 else peer_select(ids,last,due,views[i],threshold,k,
                               a.sync_peer_solver,a.sync_peer_objective,getattr(a,"sync_coverage_weight","equal")!="equal") for i in ids]
            else:
                # Deliver common scalar scores before deterministic hybrid election.
                for i in ids:
                    self.transport.send(r,"observe","server",i,"HYBRID_SCORE_ROSTER",
                        {j:None if scores[j] is None else float(scores[j]) for j in ids})
                from .advanced import joint_peer_select
                if getattr(a,"sync_joint_hybrid",False):
                    elections=[tuple(ids) if r==0 else joint_peer_select(ids,last,due,views[i],scores,threshold,k,a.sync_peer_solver,a.sync_novelty_weight) for i in ids]
                else:
                    elections=[tuple(ids) if r==0 else peer_filter(ids,last,due,views[i],scores,threshold,k,
                            a.sync_peer_solver,a.sync_peer_objective) for i in ids]
            for i,election in zip(ids,elections):
                self.transport.send(r,'admission',i,'server','ELECTION_RESULT',{'selected':election})
            if elections and any(x!=elections[0] for x in elections):
                raise RuntimeError("peer election disagreement")
            selected = elections[0] if elections else ()
            assignments = coverage(ids,selected,reports,threshold)
        if policy=="mass_stratified":
            from .baselines import mass_stratified
            weights={}
            for i in ids:
                payload=self.transport.send(r,"observe",i,"server","CLIENT_MASS",{"mass":reports[i].aggregation_weight if reports[i] else 1.})
                weights[i]=payload["mass"]
            if r==0:selected=ids;multiplicity={i:1 for i in ids}
            else:selected,multiplicity=mass_stratified(ids,weights,k,a.seed,r)
        elif policy=="divfl_sketch":
            from .baselines import gradient_facility
            delivered={}
            for i in ids:
                msg=self.transport.send(r,"observe",i,"server","GRADIENT_PROBE",asdict(reports[i]) if reports[i] else None)
                delivered[i]=Report(**{**msg,"labels":tuple(msg["labels"]),"counts":tuple(msg["counts"])}) if msg else None
            selected=ids if r==0 else gradient_facility(ids,last,due,delivered,k)
            for i in selected:self.transport.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
        elif r==0:
            selected = ids
        elif a.sync_policy=="block_random":
            selected=baseline_select("random",ids,due,last,k,a.seed,(r-1)//a.sync_block_length)
        elif a.sync_policy in ("full","random","oldest"):
            selected = baseline_select(a.sync_policy,ids,due,last,len(ids) if k is None else k,a.seed,r)
        elif policy in ("cached","update_match","adaptive"):
            task=r//((a.global_rounds+1)//a.tasks)
            cached=cached_views(self.bank.values(),ids,task,r,a.sync_bank_age)
            if policy=="cached":
                weights={i:x.importance if x is not None else 1. for i,x in cached.items()} if getattr(a,"sync_coverage_weight","equal")!="equal" else None
                from .advanced import observed_trends
                trends=observed_trends(self.delivered_history,ids,task,r) if getattr(a,"sync_staleness_predict",False) else None
                selected=select_cached(ids,last,due,cached,k,r,a.sync_bank_age,a.sync_peer_solver,weights,trends)
            elif policy=="update_match":
                selected=update_match(ids,last,due,cached,k,r,a.sync_bank_age,a.sync_peer_solver,getattr(a,"sync_update_weighted",False))
            else:
                from .impact import adaptive_select
                selected,impact=adaptive_select(ids,last,due,cached,r,task,(a.global_rounds+1)//a.tasks,
                    a.sync_bank_age,self.impact_model,k,a.sync_current_tolerance,a.sync_past_tolerance,a.sync_soft_budget)
            refs={i:None if x is None else x.id for i,x in cached.items()}
            assignments=coverage(ids,selected,cached,threshold)
            for i in selected:
                self.transport.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
        elif (own or server) and not peer:
            selected = novelty_admit(ids,last,due,scores,threshold,k)
        if (own or server) and not peer and r!=0:
            for i in ids:
                score = scores.get(i)
                requested = i in due or score is None or above_threshold(score,threshold)
                if k is not None or requested:
                    self.transport.send(r,"admission",i,"server","ADMISSION_REQUEST",
                         {"requested":requested,"score":None if k is None or score is None else float(score),
                          "mandatory":i in due,"unknown":score is None})
                if i in selected:
                    self.transport.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
        if getattr(a,"sync_bandwidth_audit",False) and (policy in ("full","random","oldest") or peer or r==0):
            for i in selected:self.transport.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
        if getattr(a,"sync_dynamic_threshold",False):
            for i in ids:
                msg=self.transport.send(r,"admission",i,"server","THRESHOLD_CALIBRATION_SCORE",{"score":None if scores.get(i) is None else float(scores[i])})
                if msg["score"] is not None:self.score_history.append(msg["score"])
            self.score_history=self.score_history[-max(1,a.sync_history*len(ids)):]
        for i in ids:
            if reports.get(i) is not None:
                self.local[i].history.append(reports[i])
                self.local[i].history = self.local[i].history[-a.sync_history:]
        d = {"round":r,"available":ids,"due":due,"selected":selected,"multiplicity":multiplicity,"threshold":float(threshold),
             "deadline_cap_violations":[i for i in due if i not in selected],
             "scores":{i:None if x is None else float(x) for i,x in scores.items()},
             "references":refs,"bank_versions":versions,"coverage":assignments,"impact":impact,
             "reasons":{i:("bootstrap" if r==0 else "mandatory" if i in due and i in selected
                           else "deadline_cap" if i in due else "unknown" if i in scores and scores[i] is None
                           else "fixed_count_fill" if i in selected and i in scores and not above_threshold(scores[i],threshold)
                           else "selected" if i in selected else "deferred") for i in ids}}
        if getattr(a,"sync_audit_references",False):
            d["reference_sources"]={i:self.local[i].audit_bank_sources.get(refs.get(i),[]) if server and isinstance(refs.get(i),str) else [] for i in ids}
            if peer:
                d["peer_distances"]=[{"clients":[i,j],"distance":None if (v:=distance(reports[i],reports[j])) is None else float(v)} for i in ids for j in ids if i<j]
        self.decisions.append(d)
        return selected

    def ingest_uploads(self,r,reports,accepted,uploaded):
        policy=self.args.sync_policy
        uses_bank=policy in ("server","combined12","combined23","combined123")
        if (uses_bank and self.args.sync_delivery!="telemetry") or policy in ("cached","update_match","adaptive"):
            arrived={}
            for i in uploaded:
                report=reports.get(i)
                if report is not None:
                    msg=self.transport.send(r,"upload",i,"server","STAT_REPORT",asdict(report))
                    arrived[i]=Report(**{**msg,"labels":tuple(msg["labels"]),"counts":tuple(msg["counts"])})
            self.ingest(arrived,r,accepted)

    def complete(self,r,reports,accepted,downloaded,uploaded=None,ingested=False):
        uploaded=accepted if uploaded is None else uploaded
        complete=set(accepted)&set(downloaded)
        policy=self.args.sync_policy
        uses_bank=policy in ("server","combined12","combined23","combined123")
        if not ingested:
            self.ingest_uploads(r,reports,accepted,uploaded)
        if uses_bank and self.args.sync_delivery!="telemetry":
            recipients=self.decisions[-1]["available"] if self.args.sync_delivery=="broadcast" else downloaded
            self.deliver_bank(r,recipients,"download")
        for i in downloaded:
            self.local[i].model_version = self.model_version
            self.local[i].local_revision = 0
        for i in complete:
            self.local[i].last_sync = r
            self.local[i].last_report = reports.get(i)
        if policy in ("peer","combined13","combined23","combined123"):
            if self.args.sync_peer_transport in ("sparse","distributed_sparse"):
                # Sparse graph must not retroactively infer unobserved edges.
                original=self.decisions[-1]["coverage"]
                self.decisions[-1]["accepted_coverage"]={i:{**x,"covered":x["covered"] and x["representative"] in accepted} for i,x in original.items()}
            else:
                self.decisions[-1]["accepted_coverage"] = coverage(self.ids,accepted,reports,
                                                                   threshold_value(self.args.sync_threshold))
        self.decisions[-1].update(accepted=list(accepted),downloaded=list(downloaded),
                                  complete=sorted(complete),model_version=self.model_version)
        return complete
