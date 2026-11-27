"""Paired-probe representation statistics. Every row must refer to the same probe.
Uses truncated centered SVD and reports sample-space CCA (rank-limited).
"""
import numpy as np

def _center(a):
    a=np.asarray(a,dtype=float)
    if a.ndim!=2 or len(a)<3 or not np.isfinite(a).all():raise ValueError("invalid probe matrix")
    return a-a.mean(0)

def basis(a,rank,variance=1.):
    u,s,v=np.linalg.svd(a,full_matrices=False)
    positive=int((s>max(a.shape)*np.finfo(float).eps*max(s[0],1e-30)).sum())
    if not positive:return u[:,:0],s[:0],v[:0]
    k=min(positive,rank)
    if variance<1.:
        k=min(k,int(np.searchsorted(np.cumsum(s*s)/np.sum(s*s),variance)+1))
    return u[:,:k],s[:k],v[:k]

def cca(a,b,rank=16,variance=1.):
    a,b=_center(a),_center(b)
    if len(a)!=len(b):raise ValueError("paired observations required")
    # Limit rank to avoid a trivially full sample-space intersection.
    rank=max(1,min(rank,(len(a)-1)//2))
    ua,sa,va=basis(a,rank,variance);ub,sb,vb=basis(b,rank,variance)
    if not len(sa) or not len(sb):return None
    l,rho,_=np.linalg.svd(ua.T@ub,full_matrices=False)
    rho=np.clip(rho,0,1)
    canonical=ua@l
    weights=np.abs(canonical.T@a).sum(1)
    pw=float(np.dot(weights,rho)/weights.sum()) if weights.sum() else None
    return rho,pw

def _corr(x,y):
    x=x-x.mean();y=y-y.mean()
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/den) if den else None

def measures(a,b,rank=16):
    a,b=_center(a),_center(b)
    if len(a)!=len(b):raise ValueError("paired observations required")
    n=len(a);ga=a@a.T;gb=b@b.T
    den=np.linalg.norm(ga)*np.linalg.norm(gb)
    cka=float(np.clip(np.sum(ga*gb)/den,0,1)) if den else None
    hsic=float(np.sum(ga*gb)/(n-1)**2)
    ca=cca(a,b,rank);sv=cca(a,b,rank,.99)
    if ca is None or sv is None or cka is None:return None
    rho,pw=ca;svrho,_=sv
    d1=np.maximum(0,np.diag(ga)[:,None]+np.diag(ga)[None,:]-2*ga)
    d2=np.maximum(0,np.diag(gb)[:,None]+np.diag(gb)[None,:]-2*gb)
    tri=np.triu_indices(n,1);rsa=_corr(d1[tri],d2[tri])
    # Orthogonal Procrustes score for unequal feature counts: nuclear norm
    # of cross-covariance normalized by Frobenius energy.
    _,ra=np.linalg.qr(a.T,mode="reduced");_,rb=np.linalg.qr(b.T,mode="reduced")
    proc=float(np.linalg.svd(ra@rb.T,compute_uv=False).sum()/(np.linalg.norm(a)*np.linalg.norm(b)))
    k=min(5,n-1)
    np.fill_diagonal(d1,np.inf);np.fill_diagonal(d2,np.inf)
    nn1=np.argsort(d1,axis=1,kind="stable")[:,:k];nn2=np.argsort(d2,axis=1,kind="stable")[:,:k]
    neighbors=np.mean([len(set(x)&set(y))/k for x,y in zip(nn1,nn2)])
    angles=np.arccos(rho)
    return np.array([cka,float(rho.mean()),float(svrho.mean()),pw if pw is not None else 0.,
        rsa if rsa is not None else 0.,float(np.log1p(max(0,hsic))),np.clip(proc,0,1),
        float(angles.mean()/np.pi),float(np.mean(rho*rho)),float(neighbors)])
