import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from ..._jit import cond_jit
from .Constraint import Constraint
from ..chain import Chain

class RepulsionPlane(Constraint):
    
    # debug: bool = False
    # quick_overlap_check: bool = True
    # numba_interval_checks: bool = True
    
    def __init__(
        self, 
        chain: Chain,
        normal: np.ndarray,
        front_plane: bool = True,
        back_plane: bool = True
        ):

        self.chain = chain
        if not self.chain.keep_backup:
            self.chain.activate_backup()
        self.conf = chain.conf
        self.num_bp  = len(self.conf)
        if chain.closed:
            raise ValueError(f'Repulsion plane not valid for closed chains.')
        
        self.normal = normal / np.linalg.norm(normal)
        self.counter = 0
        self.counter_reject = 0
        
        self.front_plane = front_plane
        self.back_plane  = back_plane

    
    def check(self, moved: List = None) -> bool:
        
        # if moved is None:
        #     moved = [[1,self.num_bp-2,1]]
        if self.front_plane:
            lower = np.dot(self.chain.conf[0,:3,3],self.normal)
        else:
            lower = None
        if self.back_plane:
            upper = np.dot(self.chain.conf[-1,:3,3],self.normal)
        else:
            upper = None
        pos = self.chain.conf[:,:3,3]
        
        return check_interval(pos,1,self.num_bp-2,self.normal,lower=lower,upper=upper)
        # for interval in moved:
        #     if interval[2] <= 0:
        #         continue
        #     return check_interval(pos,interval[0],interval[1],self.normal,lower=lower,upper=upper)  
        # return True        
        
@cond_jit
def check_interval(pos: np.ndarray, idfrom: int, idto: int, normal: np.ndarray, lower: float = None, upper: float = None):
    if idfrom == 0:
        idfrom = 1
    if idto  == len(pos) - 1:
        idto = len(pos) - 2
    if idfrom > idto:
        return True
    
    projs = np.dot(pos[idfrom:idto+1],normal)
    if lower is not None:
        if np.min(projs) < lower:
            return False  
    if upper is not None:
        if np.max(projs) > upper:
            return False
    return True                
            
