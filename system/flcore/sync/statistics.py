"""Numerically explicit low-bandwidth statistics; pure NumPy functions."""
import numpy as np

def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or not np.isfinite(x).all() or np.any(x<0) or x.sum()<=0:
        raise ValueError("invalid probability vector")
    return x/x.sum()

def entropy(x):
    p=normalize(x);p=p[p>0]
    return float(-np.sum(p*np.log(p)))

def kl(p,q,smoothing=1e-12):
    if smoothing<=0:raise ValueError("positive smoothing required")
    p=normalize(p);q=normalize(q)
    if p.shape!=q.shape:raise ValueError("incompatible coordinates")
    p=normalize(p+smoothing);q=normalize(q+smoothing)
    return max(0.,float(np.dot(p,np.log(p/q))))

def js(p,q):
    p=normalize(p);q=normalize(q)
    if p.shape!=q.shape:raise ValueError("incompatible coordinates")
    m=(p+q)/2
    return max(0.,min(1.,(entropy(m)-(entropy(p)+entropy(q))/2)/np.log(2)))

def hellinger(p,q):
    p=normalize(p);q=normalize(q)
    return float(np.linalg.norm(np.sqrt(p)-np.sqrt(q))/np.sqrt(2))

def spectral_summary(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not x.size or not np.isfinite(x).all():raise ValueError("invalid matrix")
    s=np.linalg.svd(x,compute_uv=False)
    energy=s*s;total=energy.sum()
    if total==0:return np.zeros(6)
    p=energy/total
    h=entropy(p)
    positive=s[s>max(x.shape)*np.finfo(float).eps*s[0]]
    condition=s[0]/positive[-1] if len(positive) else 1.
    return np.array([np.log1p(s[0]),np.log1p(total),h,np.exp(h),
                     total/(s[0]**2),np.log1p(condition)])

def linear_cka(x,y):
    """Paired, aligned observations only; rows are the same probe examples."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2:
        raise ValueError("CKA requires paired observations")
    x=x-x.mean(0);y=y-y.mean(0)
    den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.clip(np.linalg.norm(x.T@y)**2/den,0,1)) if den else None
