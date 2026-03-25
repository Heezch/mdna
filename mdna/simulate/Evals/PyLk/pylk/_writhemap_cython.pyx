import numpy as np

cimport numpy as np

import cython

from libc.math cimport M_PI, asin, isnan, sqrt

ctypedef np.float32_t DTYPE_t
from cpython.array cimport array, clone
from cython cimport view
from cython.view cimport array as cvarray

"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)
cdef inline double norm(double[::1] vec):
    return sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)
cdef inline double[::1] cross(double[::1] v1,double[::1] v2):
#~     cdef double[::1] v = np.empty(3,dtype=np.double)
    cdef double[::1] v =  cvarray((3,),sizeof(double),'d')
    v[0] = v1[1]*v2[2]-v1[2]*v2[1]
    v[1] = v1[2]*v2[0]-v1[0]*v2[2]
    v[2] = v1[0]*v2[1]-v1[1]*v2[0]
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)
cdef inline double dot(double[::1] v1,double[::1] v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)
cdef inline double[::1] add(double[::1] v1,double[::1] v2):
#~     cdef double[::1] v = np.empty(3,dtype=np.double)
    cdef double[::1] v =  cvarray((3,),sizeof(double),'d')
    v[0] = v1[0]+v2[0]
    v[1] = v1[1]+v2[1]
    v[2] = v1[2]+v2[2]
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)    
cdef inline double[::1] sub(double[::1] v1,double[::1] v2,double[::1] v):
#~     cdef double[::1] v = np.empty(3,dtype=np.double)
#~     cdef double[::1] vn =  cvarray((3,),sizeof(double),'d')
    v[0] = v1[0]-v2[0]
    v[1] = v1[1]-v2[1]
    v[2] = v1[2]-v2[2]
    return v



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)
cdef inline double[::1] smul(double s,double[::1] v):
    v[0] = v[0]*s
    v[1] = v[1]*s
    v[2] = v[2]*s
    return v
""" 
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
@cython.cdivision(True)
cpdef double[:, ::1] wmc_writhemap_klenin1a(double[:, ::1] pos):
    cdef int N,i,j,ip1,jp1
    cdef double fac4pi,vecnorm,Omega,sng
    cdef double[::1] r12,r13,r14,r23,r24,r34,n1,n2,n3,n4,lastpos
    
    cdef array[double] templatemv
    templatemv = array('d')
    
    r12 = clone(templatemv,3,False)
    r13 = clone(templatemv,3,False)
    r14 = clone(templatemv,3,False)
    r23 = clone(templatemv,3,False)
    r24 = clone(templatemv,3,False)
    r34 = clone(templatemv,3,False)
    n1  = clone(templatemv,3,False)
    n2  = clone(templatemv,3,False)
    n3  = clone(templatemv,3,False)
    n4  = clone(templatemv,3,False)
    lastpos = clone(templatemv,3,False)
    
    N = len(pos)
    cdef double[:, ::1] WM  = np.zeros([N,N],dtype=np.double)
#~     cdef double[:, ::1] WM1 = np.zeros([N,N],dtype=np.double)
#~     cdef double[:, ::1] WM2 = np.zeros([N,N],dtype=np.double)
    fac4pi = 1./(4*M_PI)
    
    """
        Calculate Gauss integral segment contributions except those involving the last segment
    """   

    for i in range(N-3):
        ip1 = i+1
        
        r12[0] = pos[ip1,0] - pos[i,0]
        r12[1] = pos[ip1,1] - pos[i,1]
        r12[2] = pos[ip1,2] - pos[i,2]

        for j in range(i+2,N-1):
            jp1 = j+1

            r13[0] = pos[j,0] - pos[i,0]
            r13[1] = pos[j,1] - pos[i,1]
            r13[2] = pos[j,2] - pos[i,2]
            
            r14[0] = pos[jp1,0] - pos[i,0]
            r14[1] = pos[jp1,1] - pos[i,1]
            r14[2] = pos[jp1,2] - pos[i,2]
            
            r23[0] = pos[j,0] - pos[ip1,0]
            r23[1] = pos[j,1] - pos[ip1,1]
            r23[2] = pos[j,2] - pos[ip1,2]
            
            r24[0] = pos[jp1,0] - pos[ip1,0]
            r24[1] = pos[jp1,1] - pos[ip1,1]
            r24[2] = pos[jp1,2] - pos[ip1,2]
            
            r34[0] = pos[jp1,0] - pos[j,0]
            r34[1] = pos[jp1,1] - pos[j,1]
            r34[2] = pos[jp1,2] - pos[j,2]
            
            
            n1[0] = r13[1]*r14[2]-r13[2]*r14[1]
            n1[1] = r13[2]*r14[0]-r13[0]*r14[2]
            n1[2] = r13[0]*r14[1]-r13[1]*r14[0]
            vecnorm = sqrt(n1[0]*n1[0]+n1[1]*n1[1]+n1[2]*n1[2])
            if (vecnorm > 1e-10): 
                n1[0] = n1[0]/vecnorm
                n1[1] = n1[1]/vecnorm
                n1[2] = n1[2]/vecnorm

            n2[0] = r14[1]*r24[2]-r14[2]*r24[1]
            n2[1] = r14[2]*r24[0]-r14[0]*r24[2]
            n2[2] = r14[0]*r24[1]-r14[1]*r24[0]
            vecnorm = sqrt(n2[0]*n2[0]+n2[1]*n2[1]+n2[2]*n2[2])
            if (vecnorm > 1e-10): 
                n2[0] = n2[0]/vecnorm
                n2[1] = n2[1]/vecnorm
                n2[2] = n2[2]/vecnorm
            
            n3[0] = r24[1]*r23[2]-r24[2]*r23[1]
            n3[1] = r24[2]*r23[0]-r24[0]*r23[2]
            n3[2] = r24[0]*r23[1]-r24[1]*r23[0]
            vecnorm = sqrt(n3[0]*n3[0]+n3[1]*n3[1]+n3[2]*n3[2])
            if (vecnorm > 1e-10): 
                n3[0] = n3[0]/vecnorm
                n3[1] = n3[1]/vecnorm
                n3[2] = n3[2]/vecnorm

            n4[0] = r23[1]*r13[2]-r23[2]*r13[1]
            n4[1] = r23[2]*r13[0]-r23[0]*r13[2]
            n4[2] = r23[0]*r13[1]-r23[1]*r13[0]
            vecnorm = sqrt(n4[0]*n4[0]+n4[1]*n4[1]+n4[2]*n4[2])
            if (vecnorm > 1e-10): 
                n4[0] = n4[0]/vecnorm
                n4[1] = n4[1]/vecnorm
                n4[2] = n4[2]/vecnorm

            Omega =  asin(n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2])
            Omega += asin(n2[0]*n3[0]+n2[1]*n3[1]+n2[2]*n3[2])
            Omega += asin(n3[0]*n4[0]+n3[1]*n4[1]+n3[2]*n4[2])
            Omega += asin(n4[0]*n1[0]+n4[1]*n1[1]+n4[2]*n1[2])
            
            sng =1
            n4[0] = r34[1]*r12[2]-r34[2]*r12[1]
            n4[1] = r34[2]*r12[0]-r34[0]*r12[2]
            n4[2] = r34[0]*r12[1]-r34[1]*r12[0]
            if (n4[0]*r13[0]+n4[1]*r13[1]+n4[2]*r13[2]) < 0:
                sng = -1
            Omega = Omega*fac4pi*sng

            if (not isnan(Omega)):
                WM[i,j] = Omega
                WM[j,i] = Omega

    """
        Calculate Gauss integral segment contributions involving the last segment
    """
    
    j = N-1
    lastpos = pos[0]
    for i in range(1,N-2):
        ip1 = i+1
        
        r12[0] = pos[ip1,0] - pos[i,0]
        r12[1] = pos[ip1,1] - pos[i,1]
        r12[2] = pos[ip1,2] - pos[i,2]
        
        r13[0] = pos[j,0] - pos[i,0]
        r13[1] = pos[j,1] - pos[i,1]
        r13[2] = pos[j,2] - pos[i,2]
        
        r14[0] = lastpos[0] - pos[i,0]
        r14[1] = lastpos[1] - pos[i,1]
        r14[2] = lastpos[2] - pos[i,2]
        
        r23[0] = pos[j,0] - pos[ip1,0]
        r23[1] = pos[j,1] - pos[ip1,1]
        r23[2] = pos[j,2] - pos[ip1,2]
        
        r24[0] = lastpos[0] - pos[ip1,0]
        r24[1] = lastpos[1] - pos[ip1,1]
        r24[2] = lastpos[2] - pos[ip1,2]
        
        r34[0] = lastpos[0] - pos[j,0]
        r34[1] = lastpos[1] - pos[j,1]
        r34[2] = lastpos[2] - pos[j,2]
        
        n1[0] = r13[1]*r14[2]-r13[2]*r14[1]
        n1[1] = r13[2]*r14[0]-r13[0]*r14[2]
        n1[2] = r13[0]*r14[1]-r13[1]*r14[0]
        vecnorm = sqrt(n1[0]*n1[0]+n1[1]*n1[1]+n1[2]*n1[2])
        if (vecnorm > 1e-10): 
            n1[0] = n1[0]/vecnorm
            n1[1] = n1[1]/vecnorm
            n1[2] = n1[2]/vecnorm

        n2[0] = r14[1]*r24[2]-r14[2]*r24[1]
        n2[1] = r14[2]*r24[0]-r14[0]*r24[2]
        n2[2] = r14[0]*r24[1]-r14[1]*r24[0]
        vecnorm = sqrt(n2[0]*n2[0]+n2[1]*n2[1]+n2[2]*n2[2])
        if (vecnorm > 1e-10): 
            n2[0] = n2[0]/vecnorm
            n2[1] = n2[1]/vecnorm
            n2[2] = n2[2]/vecnorm
        
        n3[0] = r24[1]*r23[2]-r24[2]*r23[1]
        n3[1] = r24[2]*r23[0]-r24[0]*r23[2]
        n3[2] = r24[0]*r23[1]-r24[1]*r23[0]
        vecnorm = sqrt(n3[0]*n3[0]+n3[1]*n3[1]+n3[2]*n3[2])
        if (vecnorm > 1e-10): 
            n3[0] = n3[0]/vecnorm
            n3[1] = n3[1]/vecnorm
            n3[2] = n3[2]/vecnorm

        n4[0] = r23[1]*r13[2]-r23[2]*r13[1]
        n4[1] = r23[2]*r13[0]-r23[0]*r13[2]
        n4[2] = r23[0]*r13[1]-r23[1]*r13[0]
        vecnorm = sqrt(n4[0]*n4[0]+n4[1]*n4[1]+n4[2]*n4[2])
        if (vecnorm > 1e-10): 
            n4[0] = n4[0]/vecnorm
            n4[1] = n4[1]/vecnorm
            n4[2] = n4[2]/vecnorm

        Omega =  asin(n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2])
        Omega += asin(n2[0]*n3[0]+n2[1]*n3[1]+n2[2]*n3[2])
        Omega += asin(n3[0]*n4[0]+n3[1]*n4[1]+n3[2]*n4[2])
        Omega += asin(n4[0]*n1[0]+n4[1]*n1[1]+n4[2]*n1[2])
        
        sng =1
        n4[0] = r34[1]*r12[2]-r34[2]*r12[1]
        n4[1] = r34[2]*r12[0]-r34[0]*r12[2]
        n4[2] = r34[0]*r12[1]-r34[1]*r12[0]
        if (n4[0]*r13[0]+n4[1]*r13[1]+n4[2]*r13[2]) < 0:
            sng = -1
        Omega = Omega*fac4pi*sng

        if (not isnan(Omega)):
                WM[i,j] = Omega
                WM[j,i] = Omega
    
    return np.array(WM)