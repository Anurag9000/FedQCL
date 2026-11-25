"""Headless plots for the matched controlled-stream campaign."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot(out,truth,regimes):
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
