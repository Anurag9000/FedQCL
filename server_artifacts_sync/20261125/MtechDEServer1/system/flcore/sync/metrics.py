"""Continual and system metrics computed only from completed evaluation/ledger rows."""
import statistics

def summarize(metrics,rounds,decisions,events,tasks,rounds_per_task,clients):
    final_round=max(x["round"] for x in metrics)
    final={x["task"]:x for x in metrics if x["round"]==final_round}
    lookup={(x["round"],x["task"]):x for x in metrics}
    bwt=[final[t]["accuracy"]-lookup[(t+1)*rounds_per_task-1,t]["accuracy"] for t in range(tasks-1)]
    macro=[]
    for x in final.values():
        macro.extend(c/n for c,n in zip(x["class_correct"],x["class_counts"]) if n)
    old=[x["accuracy"] for x in metrics if x["task"]<x["round"]//rounds_per_task]
    current=[x["accuracy"] for x in metrics if x["task"]==x["round"]//rounds_per_task]
    visits={i:0 for i in range(clients)}
    for row in decisions:
        for i in row["complete"]:visits[i]+=1
    counts=list(visits.values())
    jain=sum(counts)**2/(clients*sum(x*x for x in counts)) if any(counts) else None
    model_ages=[v for row in rounds for v in row["model_ages"].values()]
    by_client={i:0 for i in range(clients)}
    by_round={}
    traffic=set()
    for event in events:
        r=event["round"];n=event["serialized_bytes"]
        by_round[r]=by_round.get(r,0)+n
        for end in ("sender","receiver"):
            i=event[end]
            if type(i) is int and i in by_client:
                by_client[i]+=n;traffic.add((r,i))
    return {"backward_transfer":statistics.mean(bwt) if bwt else 0.,
            "retention_auc_mean":statistics.mean(old) if old else None,
            "acquisition_auc_mean":statistics.mean(current) if current else None,
            "macro_class_accuracy":statistics.mean(macro) if macro else None,
            "worst_class_accuracy":min(macro) if macro else None,
            "complete_syncs_by_client":visits,"jain_sync_fairness":jain,
            "max_model_age":max(model_ages,default=0),
            "mean_model_age":statistics.mean(model_ages) if model_ages else 0.,
            "silent_client_rounds":len(rounds)*clients-len(traffic),
            "deadline_cap_violations":sum(len(row["deadline_cap_violations"]) for row in decisions),
            "optimizer_steps":sum(o["optimizer_steps"] for row in rounds for o in row["outcomes"]),
            "processed_samples":sum(o["samples"] for row in rounds for o in row["outcomes"]),
            "replay_samples":sum(o["replay_samples"] for row in rounds for o in row["outcomes"]),
            "bytes_by_endpoint":by_client,"bytes_by_round":by_round,
            "final_brier":statistics.mean(x["brier"] for x in final.values()) if all("brier" in x for x in final.values()) else None,
            "final_ece":statistics.mean(x["ece"] for x in final.values()) if all("ece" in x for x in final.values()) else None}
