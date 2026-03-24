#!/bin/env python3
import sys
import string
import numpy as np
from ..._jit import cond_jit


class TangentCorr:
    
    def __init__(self, mmax: int, disc_len: float = None):
        self.mmax = mmax
        self._disc_len = disc_len
        if disc_len is None: 
            self._disc_lens = np.zeros(2)
        self._tc = np.zeros((mmax,2))
    
    def add_conf(self, pos: np.ndarray) -> None:
        if len(pos.shape) not in [2,3]:
            raise ValueError(f'Position matrix should consist of 2 or 3 dimensions, corresponding to a single and multiple snapshots, respectively.')
        tans = get_tangents(pos,normalized=False)
        if self._disc_len is None:
            disc_len = mean_vector_length(tans)
            if len(pos.shape) == 3:
                # multiple confs
                self._disc_lens[0] += disc_len * len(pos)
                self._disc_lens[1] += len(pos)
            else:
                # single conf
                self._disc_lens[0] += disc_len
                self._disc_lens[1] += 1
        
        self.add_tans(tans,normalized=False)
                
        
    def add_tans(self, tans: np.ndarray, normalized: bool = False) -> None:
        
        if normalized:
            utans = np.copy(tans)
        else:
            utans = normalize(tans)
        
        if self.mmax >= tans.shape[-2]-1:
            mmax = tans.shape[-2]-1
        else:
            mmax = self.mmax
            
        if len(tans.shape) == 3:
            self._tc = _tc_iter_multi(self._tc,utans,mmax)
            # for s in range(len(utans)):
            #     for i in range(len(utans[0])-1):
            #         for m in range(mmax):
            #             j = i+1+m
            #             if j >= len(utans[0]):
            #                 break
            #             self._tc[m,0] += np.dot(utans[s,i],utans[s,j])
            #             self._tc[m,1] += 1
        else:  
            self._tc = _tc_iter_single(self._tc,utans,mmax)
            # for i in range(len(utans)-1):
            #     for m in range(mmax):
            #         j = i+1+m
            #         if j >= len(utans):
            #             break
            #         self._tc[m,0] += np.dot(utans[i],utans[j])
            #         self._tc[m,1] += 1
    
    @property
    def disc_len(self):
        if self._disc_len is None:
            return self._disc_lens[0]/self._disc_lens[1]
        return self._disc_len
    
    @property           
    def tc(self):
        tc = np.copy(self._tc)
        tc[np.where(tc[:,1]==0),1] = 1
        return tc[:,0]/tc[:,1]    

    @property
    def lb(self):    
        tc = self.tc   
        data = np.zeros((3,len(tc)))
        data[0] = np.arange(1,len(tc)+1)*self.disc_len
        data[1] = tc        
        data[2] = -data[0]/np.log(data[1])
        return data
    
@cond_jit
def _tc_iter_single(tc: np.ndarray, utans: np.ndarray, mmax: int) -> np.ndarray:
    for i in range(len(utans)-1):
        for m in range(mmax):
            j = i+1+m
            if j >= len(utans):
                break
            tc[m,0] += np.dot(utans[i],utans[j])
            tc[m,1] += 1
    return tc

@cond_jit
def _tc_iter_multi(tc: np.ndarray, utans: np.ndarray, mmax: int) -> np.ndarray:
    for s in range(len(utans)):
        for i in range(len(utans[0])-1):
            for m in range(mmax):
                j = i+1+m
                if j >= len(utans[0]):
                    break
                tc[m,0] += np.dot(utans[s,i],utans[s,j])
                tc[m,1] += 1
    return tc

########################################################################
########################################################################
########################################################################
# Tangent correlation function

def persistence_length(pos: np.ndarray, mmax: int = 50, disc_len: float = None) -> np.ndarray:
    tans = get_tangents(pos,normalized=False)
    if disc_len is None:
        disc_len = mean_vector_length(tans)
    ntans = normalize(tans)
    tc = tangent_correlators(ntans,mmax=mmax)
    data = np.zeros((3,len(tc)))
    data[0] = np.arange(1,len(tc)+1)*disc_len
    data[1] = tc
    data[2] = -tc[:,0]*disc_len/np.log(tc[:,1])
    return data

########################################################################
########################################################################
########################################################################
# Tangent correlation function

def tangent_correlators(tans: np.ndarray, mmax: int=None, check_normalization: bool = True) -> np.ndarray:
    """ calculates the tangent-tangent correlation functions for all step distances m up to mmax 
    
        assumes tangents to be normalized
    """
    if len(tans.shape) != 3:
        raise ValueError(f'Expected three dimensional array. Encountered ndarray of shape {tans.shape}.')
    
    if check_normalization:
        if not tans_normlized(tans):
            tans = normalize(tans)
    
    # set mmax
    if mmax is None or mmax >= tans.shape[-2]-1:
        mmax = tans.shape[-2]-1
    return _tangent_correlators(tans,mmax)
    
@cond_jit
def _tangent_correlators(tans: np.ndarray, mmax: int) -> np.ndarray:
    tancor = np.zeros(mmax)
    numcor = np.zeros(mmax)
    for s in range(len(tans)):
        for i in range(len(tans[0])-1):
            for m in range(mmax):
                j = i+1+m
                if j >= len(tans[0]):
                    break
                tancor[m] += np.dot(tans[s,i],tans[s,j])
                numcor[m] += 1
    return tancor/numcor

########################################################################
########################################################################
########################################################################
# Vector Methods
 
def get_tangents(pos: np.ndarray, normalized: bool = False) -> np.ndarray:
    """ 
        returns tangents for given configurations. The position vectors for a given configuration must be accessed through the second to last dimension and the components of the vector through the last. 
    """
    ndims = len(pos.shape)
    tans = np.diff(pos,axis=ndims-2)
    if normalized:
        tans = normalize(tans)
    return tans

def normalize(vecs: np.ndarray) -> np.ndarray:
    """ normalizes arrays of vectors of any shape provided that the components of the vectors are stored along the last dimension """
    ndims = len(vecs.shape)
    einsumdims = string.ascii_lowercase[:ndims]
    lens = vector_lengths(vecs)
    # for 2d confs this is ij,i,->ij
    nvecs = np.einsum('%s,%s->%s'%(einsumdims,einsumdims[:-1],einsumdims),vecs,(1./lens))
    return nvecs

def tans_normlized(tans: np.ndarray) -> bool:
    """ checks if tangents are normalized """
    ndims = len(tans.shape)
    if ndims > 2:
        return tans_normlized(tans[0])
    if np.abs(np.linalg.norm(tans[0])-1) > 1e-10:
        return False
    return True
    
def vector_lengths(vecs: np.ndarray) -> np.ndarray:
    """ returns lengths of vectors """
    ndims = len(vecs.shape)
    return np.linalg.norm(vecs,axis=ndims-1)
    
def mean_vector_length(vecs: np.ndarray) -> float:
    """ calculated the mean vector length """
    return np.mean(vector_lengths(vecs))
