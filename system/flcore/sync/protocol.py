"""Logical transport, versioned local state, and pre-training admission."""
from dataclasses import dataclass, asdict, field
import json
import random
from .policies import (Report, distance, threshold_value, due_clients,
                       local_reference, bank_score, novelty_admit, peer_select,
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

class Transport:
    """In-process reliable logical deliveries, per-recipient accounting."""
    def __init__(self):
        self.events = []
    def send(self, r, phase, sender, receiver, kind, payload, tensor_bytes=0, delivered=True):
        envelope = {"round":r,"phase":phase,"sender":sender,"receiver":receiver,
                    "kind":kind,"payload":payload}
        wire = encoded(envelope)
        n = len(wire)+tensor_bytes
        self.events.append({"event_id":len(self.events),"round":r,"phase":phase,
                            "sender":sender,"receiver":receiver,"kind":kind,
                            "payload_bytes":tensor_bytes+len(encoded(payload)),
                            "serialized_bytes":n,"delivered":delivered,
                            "delivered_bytes":n if delivered else 0})
        return json.loads(wire)["payload"] if delivered else None

class Controller:
    def __init__(self, args, ids):
        self.args = args
        self.ids = tuple(sorted(ids))
        self.local = {i:LocalState() for i in self.ids}
        self.transport = Transport()
        self.bank = {}
        self.bank_version = 0
        self.bank_snapshots = {0:()}
        self.model_version = 0
        self.arrivals = []
        self.decisions = []

    def observe(self, i, task, r, dataset):
        a = self.args
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
        for idx in rng.sample(range(len(dataset)),n):
            counts[int(dataset[idx][1])]+=1
        report = Report(i,task,r,labels,tuple(counts.values()),
                        model_version=self.local[i].model_version)
        return (report if report.valid(a.sync_min_observation) else None,
                tuple(population.values()))

    def ingest(self, reports, r, accepted, observed=False):
        changed = False
        for i,report in sorted(reports.items()):
            if report is None:
                continue
            self.arrivals.append({"round":r,"report":asdict(report),
                                  "accepted":i in accepted,"observed_only":observed})
            if observed or i in accepted:
                key = (i,report.task,report.round if self.args.sync_bank_reference in ("history","prior") else -1)
                self.bank[key] = report
                changed = True
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
        ids = self.ids if available is None else tuple(sorted(available))
        last = {i:self.local[i].last_sync for i in ids}
        due = due_clients(ids,last,r,a.sync_max_skip)
        threshold = threshold_value(a.sync_threshold)
        k = None if a.sync_mode=="threshold" else min(a.sync_k,len(ids))
        scores, refs, assignments = {},{},{}
        versions = {i:self.local[i].bank_version for i in ids}
        # Every available client sees the common clock and roster under a declared
        # free common-control convention; all additional messages below are charged.
        if a.sync_policy=="server" and a.sync_delivery=="telemetry":
            delivered = {}
            for i in ids:
                if reports[i] is not None:
                    msg = self.transport.send(r,"observe",i,"server","STAT_REPORT",asdict(reports[i]))
                    delivered[i] = Report(**{**msg,"labels":tuple(msg["labels"]),"counts":tuple(msg["counts"])})
            self.ingest(delivered,r,(),observed=True)
            self.deliver_bank(r,ids,"observe")
            versions = {i:self.local[i].bank_version for i in ids}
        if a.sync_policy=="own":
            for i in ids:
                cur = reports[i]
                ref = local_reference(cur,self.local[i].last_report,self.local[i].history,
                                      a.sync_local_reference,a.sync_history) if cur is not None else None
                scores[i] = distance(cur,ref)
                refs[i] = ref.id if ref else None
        elif a.sync_policy=="server":
            for i in ids:
                cur = reports[i]
                scores[i],refs[i] = bank_score(cur,self.local[i].bank,r,a.sync_bank_age,
                    a.sync_include_self,a.sync_bank_reference) if cur is not None else (None,None)
        elif a.sync_policy=="peer":
            views = {i:{i:reports[i]} for i in ids}
            if a.sync_peer_transport=="relay":
                arrived = {}
                for i in ids:
                    msg = self.transport.send(r,"observe",i,"server","PEER_STAT_REPORT",
                                              asdict(reports[i]) if reports[i] else None)
                    arrived[i] = msg
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
                                                  asdict(reports[i]) if reports[i] else None)
                        views[j][i] = Report(**{**msg,"labels":tuple(msg["labels"]),
                                                "counts":tuple(msg["counts"])}) if msg else None
            elections = [tuple(ids) if r==0 else peer_select(ids,last,due,views[i],threshold,k,
                           a.sync_peer_solver,a.sync_peer_objective) for i in ids]
            for i,election in zip(ids,elections):
                self.transport.send(r,'admission',i,'server','ELECTION_RESULT',{'selected':election})
            if elections and any(x!=elections[0] for x in elections):
                raise RuntimeError("peer election disagreement")
            selected = elections[0] if elections else ()
            assignments = coverage(ids,selected,reports,threshold)
        if r==0:
            selected = ids
        elif a.sync_policy in ("full","random","oldest"):
            selected = baseline_select(a.sync_policy,ids,due,last,k or len(ids),a.seed,r)
        elif a.sync_policy in ("own","server"):
            selected = novelty_admit(ids,last,due,scores,threshold,k)
        if a.sync_policy in ("own","server") and r!=0:
            for i in ids:
                score = scores.get(i)
                requested = i in due or score is None or score>threshold
                if k is not None or requested:
                    self.transport.send(r,"admission",i,"server","ADMISSION_REQUEST",
                         {"requested":requested,"score":None if k is None or score is None else float(score),
                          "mandatory":i in due,"unknown":score is None})
                if i in selected:
                    self.transport.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
        for i in ids:
            if reports.get(i) is not None:
                self.local[i].history.append(reports[i])
                self.local[i].history = self.local[i].history[-a.sync_history:]
        d = {"round":r,"available":ids,"due":due,"selected":selected,
             "deadline_cap_violations":[i for i in due if i not in selected],
             "scores":{i:None if x is None else float(x) for i,x in scores.items()},
             "references":refs,"bank_versions":versions,"coverage":assignments,
             "reasons":{i:("bootstrap" if r==0 else "mandatory" if i in due and i in selected
                           else "deadline_cap" if i in due else "unknown" if i in scores and scores[i] is None
                           else "fixed_count_fill" if i in selected and i in scores and scores[i]<=threshold
                           else "selected" if i in selected else "deferred") for i in ids}}
        self.decisions.append(d)
        return selected

    def complete(self,r,reports,accepted,downloaded):
        complete = set(accepted)&set(downloaded)
        if self.args.sync_policy=="server" and self.args.sync_delivery!="telemetry":
            arrived = {}
            for i in accepted:
                if reports.get(i) is not None:
                    msg = self.transport.send(r,"upload",i,"server","STAT_REPORT",asdict(reports[i]))
                    arrived[i] = Report(**{**msg,"labels":tuple(msg["labels"]),"counts":tuple(msg["counts"])})
            self.ingest(arrived,r,accepted)
            recipients = self.ids if self.args.sync_delivery=="broadcast" else downloaded
            self.deliver_bank(r,recipients,"download")
        for i in downloaded:
            self.local[i].model_version = self.model_version
            self.local[i].local_revision = 0
        for i in complete:
            self.local[i].last_sync = r
            self.local[i].last_report = reports.get(i)
        if self.args.sync_policy=="peer":
            self.decisions[-1]["accepted_coverage"] = coverage(self.ids,accepted,reports,
                                                               threshold_value(self.args.sync_threshold))
        self.decisions[-1].update(accepted=list(accepted),downloaded=list(downloaded),
                                  complete=sorted(complete),model_version=self.model_version)
        return complete
