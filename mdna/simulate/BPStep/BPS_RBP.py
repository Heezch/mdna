from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import scipy as sp
import sys
from ..chain import Chain
from ... import math as so3
from .BPStep import BPStep
from ..BlockMat.bmat import BlockOverlapMatrix

RBP_NUMBA_ENERGY_EVAL = True


class RBP(BPStep):
    def __init__(
        self,
        chain: Chain,
        sequence: str,
        groundstate: np.ndarray,
        stiffmat: Any,
        coupling_range: int,
        closed: bool = False,
        static_group: bool = True,
        temp: float = 300,
        conserve_twist: bool = True
    ):        

        # set reference temperature for dataset
        self._set_gs_vecs(groundstate)
        self._set_stiffness(stiffmat,coupling_range,sequence,closed)
        
        super().__init__(chain, sequence, closed, static_group, temp=temp,conserve_twist=conserve_twist)
                

    #########################################################################################
    #########################################################################################

    def _set_gs_vecs(self,groundstate: np.ndarray) -> None:
        if len(groundstate.shape) == 1:
            if len(groundstate) % 6 != 0:
                raise ValueError(f'Invalid dimension of provided ground state. Shape should be be (N,6) or (6N,)')
            N = len(groundstate) // 6
            groundstate = groundstate.reshape(N,6)
        self.gs_vecs = groundstate        
            
    def _set_stiffness(self, stiffmat: np.ndarray, coupling_range: int, sequence: str, closed: bool) -> None:

        # assign nbps
        nbp = len(sequence)
        if closed:
            nbps = nbp
        else:
            nbps = nbp - 1
        
        self.stiffmat = stiffmat
        self.ncoup = coupling_range
        klim = coupling_range+1
        N = stiffmat.shape[0] // 6
        self.klim = klim
        
        if closed and 2*self.ncoup >= nbps:
            raise ValueError(f'For closed chains the coupling range ({self.ncoup}) needs to be smaller than half the number of base pair steps ({nbps}). Please reduce the coupling range to {(nbps-1)//2}.') 
    
        # check validity of dimensions 
        if N != nbps: 
            raise ValueError(f'RBP._set_stiffness: Dimension of provided stiffness matrix not matching number of base pair steps defined by sequence ({nbps}) base pairs. Matrix contains {N} blocks.')      
                
        self.table_matblocks = np.zeros((nbps,klim,6,6),dtype=float)
        self.table_current_energies  = np.zeros((nbps,klim),dtype=float)
        self.table_proposed_energies = np.zeros((nbps,klim),dtype=float)
        
        for i in range(nbps):
            for k in range(klim):
                j = i+k
                if j >= nbps:
                    j = j % nbps
                
                if sp.sparse.isspmatrix(stiffmat):
                    self.table_matblocks[i,k] = stiffmat[i*6:(i+1)*6,j*6:(j+1)*6].toarray()
                else:
                    self.table_matblocks[i,k] = stiffmat[i*6:(i+1)*6,j*6:(j+1)*6]
                if k == 0 :
                    # diagnonal components pick up the leading factor 1/2, while for off-diagonal components that factor 
                    # cancels due to the double occurance of the corresponding term
                    self.table_matblocks[i,k] *= 0.5

    def _init_params(self) -> None:
        pass

    #########################################################################################
    #########################################################################################

    def _eval_delta_E(self) -> float:
        
        if len(self.proposed_ids) == 0:
            return 0
        
        if RBP_NUMBA_ENERGY_EVAL:
            return rbp_numba_eval_delta_E(
                        self.proposed_ids, 
                        self.table_proposed_energies, 
                        self.table_current_energies,
                        self.proposed_deforms,
                        self.table_matblocks,
                        self.closed
                    )

        dE = 0
        table_selection = np.zeros((self.nbps,self.klim),dtype=bool)
        for i in self.proposed_ids:
            # forward components
            for k in range(self.klim):
                if table_selection[i,k]:
                    continue
                j = i + k
                if j >= self.nbps:
                    if not self.closed:
                        # out of range
                        continue
                    j = j % self.nbps
                
                # the factor 1/2 is included in the matrix
                self.table_proposed_energies[i,k] = self.proposed_deforms[i].T @ self.table_matblocks[i,k] @ self.proposed_deforms[j].T
                table_selection[i,k] = True
                dE += self.table_proposed_energies[i,k] - self.table_current_energies[i,k]
            
            for k in range(self.klim):
                ii = i - k
                ii = ii % self.nbps
                if table_selection[ii,k]:
                    continue
                
                j = ii + k
                if j >= self.nbps:
                    if not self.closed:
                        # out of range
                        continue
                    j = j % self.nbps
                    
                # the factor 1/2 is included in the matrix
                self.table_proposed_energies[ii,k] = self.proposed_deforms[ii].T @ self.table_matblocks[ii,k] @ self.proposed_deforms[j].T
                table_selection[ii,k] = True
                dE += self.table_proposed_energies[ii,k] - self.table_current_energies[ii,k]
                  
        return dE
    
    
    def full_mat_eval(self,current: bool = True):
        if current:
            deforms = self.current_deforms
        else:
            deforms = self.proposed_deforms
        n = len(deforms)
        X = np.zeros((n*6))
        for i in range(n):
            X[i*6:(i+1)*6] = deforms[i]
        return 0.5* X.T @ self.stiffmat @ X


    #########################################################################################
    #########################################################################################

    def set_energy(self, accept: bool = True) -> None:
        if accept:
            self.table_current_energies = np.copy(self.table_proposed_energies)
        else:
            self.table_proposed_energies = np.copy(self.table_current_energies)
    
    #########################################################################################
    #########################################################################################

    def _get_total_energy(self) -> float:
        return np.sum(self.table_current_energies)

    #########################################################################################
    #########################################################################################

    def _set_temperature(self, new_temp: float) -> None:
        self.table_matblocks *= self.temp / new_temp
        self.init_conf()


from ..._jit import cond_jit

@cond_jit
def rbp_numba_eval_delta_E(
    proposed_ids: np.ndarray, 
    table_proposed_energies: np.ndarray, 
    table_current_energies: np.ndarray,
    proposed_deforms: np.ndarray,
    table_matblocks: np.ndarray,
    closed: bool
    ) -> float:

    nbps = len(table_current_energies)
    klim = len(table_current_energies[0])

    dE = 0
    table_selection = np.zeros((nbps,klim))
    for i in proposed_ids:
        # forward components
        for k in range(klim):
            if table_selection[i,k]:
                continue
            j = i + k
            if j >= nbps:
                if not closed:
                    # out of range
                    continue
                j = j % nbps
            table_proposed_energies[i,k] = proposed_deforms[i].T @ table_matblocks[i,k] @ proposed_deforms[j].T
            table_selection[i,k] = True
            dE += table_proposed_energies[i,k] - table_current_energies[i,k]
        
        for k in range(klim):
            ii = i - k
            ii = ii % nbps
            if table_selection[ii,k]:
                continue
            
            j = ii + k
            if j >= nbps:
                if not closed:
                    # out of range
                    continue
                j = j % nbps
            table_proposed_energies[ii,k] = proposed_deforms[ii].T @ table_matblocks[ii,k] @ proposed_deforms[j].T
            table_selection[ii,k] = True
            dE += table_proposed_energies[ii,k] - table_current_energies[ii,k]
    return dE