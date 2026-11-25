"""Headless plots for the matched controlled-stream campaign."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot(out,truth,regimes,rounds=(),comparisons=(),results=()):
    with np.load(truth/"distributions.npz") as z:
        fig,axes=plt.subplots(1,2,figsize=(10,5),layout="constrained")
        for ax,key,title in zip(axes,("spatiotemporal_label","spatiotemporal_joint"),("Label similarity","Label-domain similarity")):
            im=ax.imshow(z[key],vmin=0,vmax=1,cmap="viridis");ax.set(title=title,xlabel="Round-major client index",ylabel="Round-major client index")
        fig.colorbar(im,ax=axes,shrink=.8);fig.savefig(out/"ground_truth_heatmap.png",dpi=160);plt.close(fig)
    ids=sorted({x["experiment_id"] for x in regimes});names=sorted({x["regime"] for x in regimes})
    lookup={(x["experiment_id"],x["regime"]):x["mean_joint_coverage_regret"] for x in regimes}
    vals=np.array([[lookup.get((i,j),np.nan) for j in names] for i in ids])
    fig,ax=plt.subplots(figsize=(8,9),layout="constrained");im=ax.imshow(vals,aspect="auto",cmap="magma")
    ax.set_xticks(range(len(names)),names,rotation=30,ha="right");ax.set_yticks(range(len(ids)),ids)
    ax.set_title("Joint distribution coverage regret");fig.colorbar(im,ax=ax);fig.savefig(out/"coverage_by_regime.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(ids),1,figsize=(12,max(5,len(ids)*.6)),layout="constrained",squeeze=False)
    for ax,ident in zip(axes[:,0],ids):
        rows=sorted([x for x in rounds if x["experiment_id"]==ident],key=lambda x:x["round"])
        if rows:
            values=np.array([[i in x["selected"] for x in rows] for i in range(5)])
            ax.imshow(values,aspect="auto",vmin=0,vmax=1,cmap="Greens");ax.set_ylabel(ident,rotation=0,labelpad=30);ax.set_yticks([])
    axes[-1,0].set_xlabel("Round");fig.savefig(out/"participation_heatmap.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    for ident in ids:
        rows=sorted([x for x in rounds if x["experiment_id"]==ident],key=lambda x:x["round"])
        if rows:ax.plot([x["round"] for x in rows],[x["mean_accuracy"] for x in rows],label=ident,alpha=.7)
    ax.set(xlabel="Round",ylabel="Mean seen-task accuracy");ax.legend(ncol=4,fontsize=6);fig.savefig(out/"learning_curves.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
    for x in results:
        ax.scatter(x["total_bytes"]/1e9,x["final_accuracy"]);ax.annotate(x["experiment_id"],(x["total_bytes"]/1e9,x["final_accuracy"]),fontsize=6)
    ax.set(xlabel="Logical communication (GB)",ylabel="Final accuracy");fig.savefig(out/"communication_tradeoff.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout="constrained")
    signals=sorted({x["signal"] for x in comparisons})
    for ax,signal in zip(axes.ravel(),signals):
        for method in ("M1","M2","M3"):
            rows=[x for x in comparisons if x["signal"]==signal and x["mechanism"]==method]
            ax.scatter([x["joint_tv"] for x in rows],[x["estimated_distance"] for x in rows],s=5,alpha=.3,label=method)
        ax.set(title=signal.replace("_"," "),xlabel="Joint TV",ylabel="Estimated",xlim=(0,1),ylim=(0,1));ax.legend(fontsize=6)
    fig.savefig(out/"signal_validity.png",dpi=160);plt.close(fig)
