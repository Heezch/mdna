cimport numpy as np
import cython
# from libc.math cimport M_PI, asin, 
from libc.math cimport sqrt, atan, isnan

ctypedef np.float32_t DTYPE_t
from cpython.array cimport array, clone
from cython cimport view
from cython.view cimport array as cvarray


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)
cpdef double _eval_lk_cython(double[:, ::1] curve1, double[:, ::1] curve2):
    cdef int N1,N2,i,j
    cdef double lk,dlk,nv,nw,cosbeta,sinbetasq,a0,a1,a2,val
    cdef double[::1] p1,p2,q1,q2,v,w,e1,e2,r12
    cdef double[::1] e2cosbetamine1,e2mine1cosbeta,crosse1e2
    
    cdef double minoneover4pi, a0sq, num, denom
    cdef double t1,t2,a0sqcosbeta
    
    cdef array[double] templatemv
    templatemv = array('d')
    
    p1 = clone(templatemv,3,False)
    p2 = clone(templatemv,3,False)
    q1 = clone(templatemv,3,False)
    q2 = clone(templatemv,3,False)
    v  = clone(templatemv,3,False)
    w  = clone(templatemv,3,False)
    e1 = clone(templatemv,3,False)
    e2 = clone(templatemv,3,False)
    r12 = clone(templatemv,3,False)
    e2cosbetamine1 = clone(templatemv,3,False)
    e2mine1cosbeta = clone(templatemv,3,False)
    crosse1e2 = clone(templatemv,3,False)
    
    N1 = len(curve1)
    N2 = len(curve2)
    minoneover4pi = -0.07957747154594767
    # N1 = curve1.shape[0]
    # N2 = curve2.shape[0]
    
    lk = 0
    for i in range(N1):
        for j in range(N2):
            p1 = curve1[i]
            p2 = curve1[(i+1)%N1]
            q1 = curve2[j]
            q2 = curve2[(j+1)%N2]
            
            v[0] = p2[0]-p1[0]
            v[1] = p2[1]-p1[1]
            v[2] = p2[2]-p1[2]
            
            w[0] = q2[0]-q1[0]
            w[1] = q2[1]-q1[1]
            w[2] = q2[2]-q1[2]
            
            nv = sqrt(v[0]**2+v[1]**2+v[2]**2)
            nw = sqrt(w[0]**2+w[1]**2+w[2]**2)
            
            e1[0] = v[0]/nv
            e1[1] = v[1]/nv
            e1[2] = v[2]/nv
            e2[0] = w[0]/nw
            e2[1] = w[1]/nw
            e2[2] = w[2]/nw
            
            r12[0] = q1[0]-p1[0]
            r12[1] = q1[1]-p1[1]
            r12[2] = q1[2]-p1[2]
            
            cosbeta = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2]
            sinbetasq = 1 - cosbeta**2
            
            e2cosbetamine1[0] = e2[0]*cosbeta - e1[0]
            e2cosbetamine1[1] = e2[1]*cosbeta - e1[1]
            e2cosbetamine1[2] = e2[2]*cosbeta - e1[2]
            
            e2mine1cosbeta[0] = e2[0] - e1[0]*cosbeta 
            e2mine1cosbeta[1] = e2[1] - e1[1]*cosbeta 
            e2mine1cosbeta[2] = e2[2] - e1[2]*cosbeta 
            
            crosse1e2[0] = e1[1]*e2[2] - e1[2]*e2[1] 
            crosse1e2[1] = e1[2]*e2[0] - e1[0]*e2[2] 
            crosse1e2[2] = e1[0]*e2[1] - e1[1]*e2[0] 
            
            a1 = ( r12[0]*e2cosbetamine1[0] + r12[1]*e2cosbetamine1[1] + r12[2]*e2cosbetamine1[2] ) / sinbetasq
            a2 = ( r12[0]*e2mine1cosbeta[0] + r12[1]*e2mine1cosbeta[1] + r12[2]*e2mine1cosbeta[2] ) / sinbetasq
            a0 = ( r12[0]*crosse1e2[0] + r12[1]*crosse1e2[1] + r12[2]*crosse1e2[2] ) / sinbetasq
                        
            # dlk = _F(a1+nv,a2+nw,a0,cosbeta,sinbetasq)
            # dlk = dlk - _F(a1+nv,a2,a0,cosbeta,sinbetasq)
            # dlk = dlk - _F(a1,a2+nw,a0,cosbeta,sinbetasq)
            # dlk = dlk + _F(a1,a2,a0,cosbeta,sinbetasq)
            
            dlk = 0 
            a0sq = a0**2
            a0sqcosbeta = a0sq*cosbeta
            a0sqsinbetasq = a0sq*sinbetasq
            
            t1=a1+nv
            t2=a2+nw
            num = t1*t2+a0sq*cosbeta
            denom = a0 * sqrt( t1**2 + t2**2 - 2*t1*t2*cosbeta + a0sqsinbetasq )
            val = minoneover4pi * atan(num/denom)
            if (not isnan(val)):
                dlk = dlk + val
            
            t1=a1+nv
            t2=a2
            num = t1*t2+a0sq*cosbeta
            denom = a0 * sqrt( t1**2 + t2**2 - 2*t1*t2*cosbeta + a0sqsinbetasq )
            val = minoneover4pi * atan(num/denom)
            if (not isnan(val)):
                dlk = dlk - val        
    
            t1=a1
            t2=a2+nw
            num = t1*t2+a0sq*cosbeta
            denom = a0 * sqrt( t1**2 + t2**2 - 2*t1*t2*cosbeta + a0sqsinbetasq )
            val = minoneover4pi * atan(num/denom)
            if (not isnan(val)):
                dlk = dlk - val  
    
            t1=a1
            t2=a2
            num = t1*t2+a0sq*cosbeta
            denom = a0 * sqrt( t1**2 + t2**2 - 2*t1*t2*cosbeta + a0sqsinbetasq )
            val = minoneover4pi * atan(num/denom)
            if (not isnan(val)):
                dlk = dlk + val  
            
            # dlk = 0 
            # val = _F(a1+nv,a2+nw,a0,cosbeta,sinbetasq)
            # if val == val:
            #     dlk = dlk + val
            # val = _F(a1+nv,a2,a0,cosbeta,sinbetasq)
            # if val == val:
            #     dlk = dlk - val
            # val = _F(a1,a2+nw,a0,cosbeta,sinbetasq)
            # if val == val:
            #     dlk = dlk - val   
            # val = _F(a1,a2,a0,cosbeta,sinbetasq)
            # if val == val:
            #     dlk = dlk + val
            
            if (not isnan(dlk)):
                lk = lk + dlk
    return lk
            

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.profile(True)
# @cython.cdivision(True)             
# cpdef double _F(double t1, double t2, double a0, double cosbeta, double sinbetasq):
#     cdef double minoneover4pi, a0sq, num, denom
#     minoneover4pi = -0.07957747154594767
#     a0sq = a0**2
#     num = t1*t2+a0sq*cosbeta
#     denom = a0 * sqrt( t1**2 + t2**2 - 2*t1*t2*cosbeta + a0sq*sinbetasq )
#     return minoneover4pi * atan(num/denom)