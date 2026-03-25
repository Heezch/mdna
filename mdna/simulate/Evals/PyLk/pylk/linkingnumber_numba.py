import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def _eval_lk_numba(curve1: np.ndarray, curve2: np.ndarray) -> float:
    N1 = len(curve1)
    N2 = len(curve2)
    lk = 0
    for i in range(N1):
        for j in range(N2):
            p1 = curve1[i]
            p2 = curve1[(i+1)%N1]
            q1 = curve2[j]
            q2 = curve2[(j+1)%N2]
            
            v = p2-p1
            w = q2-q1
            nv = np.linalg.norm(v)
            nw = np.linalg.norm(w)
            
            e1 = v/nv
            e2 = w/nw
            r12 = q1-p1
            
            cosbeta = np.dot(e1,e2)
            sinbetasq = 1 - cosbeta**2
            
            a1 = np.dot(r12,(e2*cosbeta-e1)) / sinbetasq
            a2 = np.dot(r12,(e2-e1*cosbeta)) / sinbetasq
            a0 = np.dot(r12,np.cross(e1,e2)) / sinbetasq
            
            dlk  = _F(a1+nv,a2+nw,a0,cosbeta,sinbetasq)
            dlk -= _F(a1+nv,a2,a0,cosbeta,sinbetasq)
            dlk -= _F(a1,a2+nw,a0,cosbeta,sinbetasq)
            dlk += _F(a1,a2,a0,cosbeta,sinbetasq)
            
            lk += dlk
    return lk
            
@jit(nopython=True, cache=True)                 
def _F(t1,t2,a0,cosbeta,sinbetasq):
    minoneover4pi = -0.07957747154594767
    a0sq = a0**2
    num = t1*t2+a0sq*cosbeta
    denom = a0 * np.sqrt( t1**2 + t2**2 - 2*t1*t2*cosbeta + a0sq*sinbetasq )
    return minoneover4pi * np.arctan(num/denom)