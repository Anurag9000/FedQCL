"""Portable Task-IL study matrix and deterministic round-level curricula.

A task is a stable sorted class set; an episode is one client's exposure to it.
No training example is assigned by this module; the private allocator handles that.
"""
import json
import hashlib
from collections import Counter
from pathlib import Path
import random
from .backbone import BACKBONE

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "configs/task_il/job_ledger.jsonl"
DEFAULTS = dict(backbone=BACKBONE,clients=5, width=2, curriculum="disjoint", offline=0, syncs=5,
                epochs=2, sharing=0, recurrence=0, outage=0, overlap=0, delay=None,
                seed=42, selection_seed=42, batch=13, replay_batch=13, memory=500,
                lr=0.005, validation_per_class=50, split="test")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def read_entries(path=LEDGER):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 48 or len({r["id"] for r in rows}) != 48:
        raise ValueError("expected exactly 48 distinct experiment entries")
    if Counter(r["family"] for r in rows) != dict(enumerate((8,3,6,3,8,3,6,3,5,3),1)):
        raise ValueError("incorrect family counts")
    return rows

def resolve(row, seed=None, split=None):
    c = {**DEFAULTS, **row["overrides"]}
    if seed is not None:
        c["seed"] = c["selection_seed"] = seed
    if split is not None:
        c["split"] = split
    if c["backbone"]!=BACKBONE:raise ValueError("unsupported Task-IL backbone")
    if c["clients"] < 2 or not 0 <= c["offline"] < c["clients"]:
        raise ValueError("invalid client count")
    if c["syncs"] * c["epochs"] != 10:
        raise ValueError("each episode must have ten local epochs")
    if c["width"] < 1 or c["width"] > 100 or c["batch"] < 1 or c["replay_batch"] < 1 or c["memory"] < 0:
        raise ValueError("invalid workload")
    return c

JOINT_LEDGER=ROOT/"configs/task_il_joint/job_ledger.jsonl"
JOINT_OMIT={"E08-01","E08-02","E08-03","E10-02"}

def read_joint_entries(path=JOINT_LEDGER):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    source=[r for r in read_entries() if r["id"] not in JOINT_OMIT]
    expected=[dict(id=r["id"].replace("E","J",1),family=r["family"],mirror_of=r["id"],
                   overrides={**r["overrides"],"temporal_overlap":True},
                   baseline=r["baseline"].replace("E","J",1)) for r in source]
    if rows!=expected or len(rows)!=44:
        raise ValueError("joint mirror ledger differs from full original catalog")
    return rows

def _joint_pending(c):
    """Insert one-class-at-a-time bridges between original local task anchors.

    The original cross-client joint/disjoint distinction remains orthogonal.
    Sharing and recurrence anchors retain their original client memberships;
    bridge task identities are unique outside the all-client joint condition.
    """
    syncs=c["syncs"];width=c["width"];n=c["clients"]
    if width<2:raise ValueError("temporal overlap needs width >= 2")
    if c["delay"] is not None:
        return _joint_delay_pending(c)
    base=make_schedule({**c,"temporal_overlap":False})
    tasks=base["tasks"]
    used={tuple(t) for t in tasks}
    pending=[]
    for client in range(n):
        es=sorted((e for e in base["episodes"] if e["client"]==client),key=lambda e:e["start"])
        prev=None
        for e in es:
            target=tuple(tasks[e["task"]])
            slot=(e["start"]//syncs)*width
            bridges=[]
            if prev is not None:
                # A smaller original remainder is a full-width sliding window
                # with just the required novel labels (no singleton head).
                if len(target)<width:
                    target=tuple(sorted(list(prev)[len(target):]+list(target)))
                common=set(prev)&set(target)
                remove=sorted(set(prev)-common);add=sorted(set(target)-common)
                if len(remove)!=len(add):raise ValueError("unequal temporal task widths")
                if len(remove)>1 and width==2 and (c["sharing"] or c["recurrence"]):
                    # Four possible bridges for a pair of disjoint two-class
                    # anchors; at most four clients share a target. Avoid
                    # accidentally introducing extra repeated task identities.
                    choices=sorted(tuple(sorted((a,b))) for a in prev for b in target)
                    bridge=next((x for x in choices if x not in used),None)
                    if bridge is None:raise ValueError("cannot allocate unique sharing bridge")
                    bridges=[bridge];used.add(bridge)
                else:
                    current=set(prev)
                    for a,b in zip(remove,add):
                        current.remove(a);current.add(b)
                        window=tuple(sorted(current))
                        if window!=target:bridges.append(window)
            for j,window in enumerate(bridges):
                pending.append((slot-len(bridges)+j,client,window))
            pending.append((slot,client,target));prev=target
    # Compress only globally empty slots; all original simultaneous anchor
    # relationships and the chronological order of sequential targets survive.
    slots={slot:j for j,slot in enumerate(sorted({p[0] for p in pending}))}
    return [(slots[slot]*syncs,i,t) for slot,i,t in pending]

def _joint_delay_pending(c):
    """Exact intervening-round delay with the same occurrence inventory.

    The 1/2-round delays cannot hold a complete five-round intervening
    episode. In those two cases the explicitly repeated target is the sole
    identical adjacent task; all distinct adjacent tasks share one class.
    """
    if c["clients"]!=5 or c["width"]!=2 or c["syncs"]!=5 or c["delay"] not in (1,2,5,10,15):
        raise ValueError("unsupported joint delay")
    d=c["delay"];labels=list(range(100));random.Random(c["seed"]).shuffle(labels)
    groups=[tuple(sorted(labels[i:i+2])) for i in range(0,100,2)]
    pending=[]
    def path(gs):
        windows=[gs[0]]
        for a,b in zip(gs,gs[1:]):
            windows.extend((tuple(sorted((a[1],b[0]))),b))
        return windows
    for i in range(1,5):
        for j,t in enumerate(path(groups[i::5])):
            pending.append((j*5+(30 if i==1 else 0),i,t))
    gs=groups[0::5];target=gs[0]
    # Three unique anchor-sharing fillers, with fixed membership independent
    # of delay. They all share target[1] with the return and main-path bridge.
    fillers=[tuple(sorted((target[1],gs[j][1]))) for j in (2,3,4)]
    pending.append((0,0,target))
    before=d//5
    for j,t in enumerate(fillers[:before]):pending.append((5+5*j,0,t))
    start=5+d;pending.append((start,0,target))
    tail=fillers[before:]+path(gs)[1:]
    for j,t in enumerate(tail):
        pending.append((120 if j==len(tail)-1 else start+5+5*j,0,t))
    return pending

def make_schedule(c):
    n, width, syncs = c["clients"], c["width"], c["syncs"]
    labels = list(range(100))
    random.Random(c["seed"]).shuffle(labels)
    groups = [tuple(sorted(labels[i:i+width])) for i in range(0,100,width)]
    pending = []
    def add(client, classes, start):
        pending.append((start, client, tuple(sorted(classes))))
    def stage(client, classes, s):
        add(client, classes, s*syncs)

    if c.get("temporal_overlap"):
        pending=_joint_pending(c)
    elif c["overlap"]:
        # Equal 16 episodes per client for all overlap variants. Each client has
        # its own 20-class universe; all 100 classes are introduced. The number
        # of novel classes per episode is maximized until the universe is covered.
        if n != 5 or width != 5 or c["overlap"] not in (2,3,4):
            raise ValueError("unsupported overlap configuration")
        o = c["overlap"]
        for i in range(n):
            universe = labels[20*i:20*(i+1)]
            rng = random.Random(c["seed"] + 1009*i)
            previous = tuple(sorted(universe[:5]))
            seen = set(previous)
            stage(i, previous, 0)
            for s in range(1,16):
                retained = rng.sample(list(previous), o)
                outside = [x for x in universe if x not in previous]
                novel = [x for x in outside if x not in seen]
                other = [x for x in outside if x in seen]
                rng.shuffle(novel); rng.shuffle(other)
                current = tuple(sorted(retained + (novel+other)[:5-o]))
                assert len(set(previous)&set(current)) == o
                stage(i,current,s)
                seen.update(current); previous=current
            assert seen == set(universe)
    elif c["delay"] is not None:
        # Same task inventory and sample ownership at every delay. Client 0
        # revisits task 0; whole filler episodes occupy the gap when possible.
        # The other clients keep communication opportunities alive through round 64.
        if n != 5 or width != 2 or syncs != 5 or c["delay"] not in (1,2,5,10,15):
            raise ValueError("unsupported recurrence delay")
        d = c["delay"]
        for i in range(1,n):
            for j,g in enumerate(groups[i::n]):
                add(i,g,j*syncs + (15 if i == 1 else 0))
        own = groups[0::n]
        add(0,own[0],0)
        filler = own[1:]
        before = d//syncs
        for j,g in enumerate(filler[:before]):
            add(0,g,5+j*syncs)
        repeat_start=5+d
        add(0,own[0],repeat_start)
        rest=filler[before:]
        for j,g in enumerate(rest):
            add(0,g,60 if j==len(rest)-1 else repeat_start+5+j*syncs)
    elif c["sharing"] or c["recurrence"]:
        # Five prespecified target groups. Matched 13-stage horizon across
        # 2/3/4-way sharing or recurrence; remaining 45 groups fill free slots.
        m = c["sharing"] or c["recurrence"]
        if n != 5 or width != 2 or m not in (2,3,4):
            raise ValueError("unsupported sharing configuration")
        occupied = set()
        for g in range(5):
            for q in range(m):
                i=(g+q)%n
                s=g if c["sharing"] else q
                stage(i,groups[g],s)
                occupied.add((s,i))
        # Spread the remaining tasks across stages and balance client workloads.
        loads=Counter(i for _,i in occupied)
        for g in groups[5:]:
            s,i=min(((s,i) for s in range(13) for i in range(n) if (s,i) not in occupied),
                    key=lambda si:(sum((si[0],j) in occupied for j in range(n)),loads[si[1]],si))
            stage(i,g,s);occupied.add((s,i));loads[i]+=1
    elif c["curriculum"] == "joint":
        for s,g in enumerate(groups):
            for i in range(n):
                stage(i,g,s)
    elif c["curriculum"] == "disjoint":
        for j,g in enumerate(groups):
            stage(j%n,g,j//n)
    else:
        raise ValueError("unknown curriculum")

    # Canonical class-set and episode identities do not depend on round timing.
    # This keeps the same head initialization, private allocation and SGD order
    # across frequency and recurrence-delay variants.
    keys=sorted({classes for _,_,classes in pending})
    tasks=[list(k) for k in keys];lookup={k:i for i,k in enumerate(keys)};episodes=[]
    for start,i,classes in sorted(pending,key=lambda item:(item[1],item[2],item[0])):
        episodes.append(dict(id=len(episodes),client=i,task=lookup[classes],start=start,
                             end=start+syncs-1))
    rounds=max(e["end"] for e in episodes)+1
    active=[{} for _ in range(rounds)]
    for e in episodes:
        for r in range(e["start"],e["end"]+1):
            key=str(e["client"])
            if key in active[r]:
                raise ValueError("overlapping local episodes")
            active[r][key]=e["id"]
    if not all(active):
        raise ValueError("empty federation round")
    if set(x for task in tasks for x in task)!=set(range(100)):
        raise ValueError("incomplete class coverage")
    if c["outage"] and 1+c["outage"]>=rounds:
        raise ValueError("outage leaves no reconnection")
    # Uniform nested random selection from active clients. No deadlines or
    # forced full training round. Initial model distribution is separate.
    selected=[]
    for r, roster in enumerate(active):
        ids=sorted(map(int,roster))
        if c["outage"] and 1<=r<1+c["outage"]:
            ids=[i for i in ids if i!=0]
        random.Random(f'{c["selection_seed"]}:selection:{r}').shuffle(ids)
        selected.append(sorted(ids[:max(0,min(len(ids),n-c["offline"]))]))
    seen=set(); expected=[]
    for r,roster in enumerate(active):
        seen.update(episodes[e]["task"] for e in roster.values())
        expected.extend([r,t] for t in sorted(seen))
    return dict(schema=1,clients=n,tasks=tasks,episodes=episodes,active=active,
                selected=selected,rounds=rounds,expected=expected)

def data_plan(c, schedule):
    # Participation and sync frequency do not affect sample ownership.
    return dict(schema=1,seed=c["seed"],validation_per_class=c["validation_per_class"],
                clients=c["clients"],tasks=schedule["tasks"],
                episodes=[dict(id=e["id"],client=e["client"],task=e["task"]) for e in schedule["episodes"]])

def validate_schedule(c,s):
    if s!=make_schedule(c):
        raise ValueError("noncanonical or corrupted schedule")
    return True
