import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..BPStep.BPStep import BPStep
from ..chain import Chain
from ..ExVol.ExVol import ExVol
from ..Constraints.Constraint import Constraint
from ..._jit import cond_jit
from ... import math as so3
from .mcstep import MCStep
from ..utils import random_unitsphere


class DoublePivot(MCStep):
    def __init__(
        self,
        chain: Chain,
        bpstep: BPStep,
        selrange_min: int,
        selrange_max: int,
        range_id1: int = None,
        range_id2: int = None,
        full_trial_conf: bool = False,
        exvol: ExVol = None,
        constraints: List[Constraint] = [],
        use_numba: bool = True,
    ):
        """
        Initiate:
            - configuration
            - energy model (separate object passed to chain)

        """
        super().__init__(chain, bpstep, full_trial_conf, exvol=exvol,constraints=constraints)
        self.name = "DoublePivot"
        self.numba_segment_rotation = use_numba

        if self.closed:
            raise ValueError(f"Pivot: Pivot move not allowed for closed chains.")

        MCS_MST_MAX_THETA = 0.09
        self.max_theta = MCS_MST_MAX_THETA * np.sqrt(self.chain.temp / 300)

        self.selrange_min = selrange_min
        self.selrange_max = selrange_max

        # set the range of the random generators selecting the hinges
        if self.closed:
            if self.selrange_max >= self.nbp / 2:
                self.selrange_max = self.nbp // 2

        if self.selrange_min < 2:
            self.selrange_min = 2
        if self.selrange_max < self.selrange_min:
            self.selrange_max = self.selrange_min

        self.restricted_range = False
        if range_id1 is not None or range_id2 is not None:
            if range_id1 is None or range_id2 is None:
                raise ValueError(
                    f"Crankshaft: range limit requires both range_id1 and range_id2 to be set."
                )
            if not self.closed and range_id1 >= range_id2:
                raise ValueError(
                    f"Crankshaft: requires range_id1 < range_id2 if chain is not closed."
                )

            if not self.closed and range_id1 == 0:
                range_id1 = 1

            self.range_id1 = range_id1
            self.range_id2 = range_id2
            if self.range_id2 < self.range_id1:
                self.range_id2 = (
                    self.range_id1 + (self.range_id2 - self.range_id1) % self.nbp
                )

            self.restricted_range = True
            self.range_id1_upper = self.range_id2 - self.selrange_min
            if self.range_id1_upper <= self.range_id1:
                raise ValueError(
                    f"Crankshaft: range {self.range_id1} - {self.range_id2} too close for min range {self.selrange_min}."
                )

        self.requires_ev_check = True
        # self.moved_intervals = np.zeros((1, 3),dtype=int)
        # self.moved_intervals[0, 2] = 1000

    #########################################################################################
    #########################################################################################

    def mc_move(self) -> bool:
        #############################
        # select hinges
        if self.restricted_range:
            idA = np.random.randint(self.range_id1, self.range_id1_upper)
            dist = self.range_id2 - idA
            if dist > self.selrange_max:
                hingedist = np.random.randint(self.selrange_min, self.selrange_max)
            else:
                hingedist = np.random.randint(self.selrange_min, dist)
            idB = idA + hingedist
        else:
            idA = np.random.randint(1, self.nbp - self.selrange_min)
            dist = self.nbp - idA
            if dist > self.selrange_max:
                hingedist = np.random.randint(self.selrange_min, self.selrange_max)
            else:
                hingedist = np.random.randint(self.selrange_min, dist)
            idB = idA + hingedist

        idAn = idA - 1
        idBn = idB - 1

        #############################
        # Trial Move
        s = random_unitsphere()
        theta = np.random.uniform(0, self.max_theta)
        Theta = s * theta
        G = so3.euler2rotmat(Theta)
        TA_rot = np.copy(self.chain.conf[idA])
        TA_rot[:3, :3] = G @ self.chain.conf[idA, :3, :3]
        
        if idB == self.nbp-1:
            TBn_rot = np.copy(self.chain.conf[idBn])
            TB_rot  = np.copy(self.chain.conf[idB])
            # rotate B
            TBn_rot[:3,:3] = G @ self.chain.conf[idBn, :3, :3]
            # translate Bn
            v = G @ (self.chain.conf[idBn,:3,3] - self.chain.conf[idA,:3,3])
            TBn_rot[:3,3] = self.chain.conf[idA,:3,3] + v
            # translate B
            v = G @ (self.chain.conf[idB,:3,3] - self.chain.conf[idA,:3,3])
            TB_rot[:3,3]  = self.chain.conf[idA,:3,3] + v

            # propose moves
            self.bpstep.propose_move(idAn, self.chain.conf[idAn], TA_rot)
            self.bpstep.propose_move(idBn, TBn_rot, TB_rot)

        else:
            TB_rot = np.copy(self.chain.conf[idB])
            TB_rot[:3, :3] = G.T @ TB_rot[:3,:3]
            
            # propose moves
            self.bpstep.propose_move(idAn, self.chain.conf[idAn], TA_rot)
            self.bpstep.propose_move(idBn, self.chain.conf[idBn], TB_rot)
            
        # check change of termini
        if self.bpstep.trace_termini:
            v = self.chain.conf[idB,:3,3] - self.chain.conf[idA,:3,3]
            vp = G @ v
            dr = vp - v
            tau_last = np.copy(self.chain.conf[-1])
            tau_last[:3,3] += dr
            self.bpstep.propose_move_last(tau_last)
        

        # calculate energy
        dE = self.bpstep.eval_delta_E()

        # metropolis step
        if np.random.uniform() >= np.exp(-dE):
            return False
        

        ##########################
        # rotate inbetween segment

        # rotate triad at idA
        self.chain.conf[idA] = TA_rot
        for i in range(idA+1,idB):
            #rotation
            self.chain.conf[i,:3,:3] = G @ self.chain.conf[i,:3,:3]
            #translation
            v = G @ (self.chain.conf[i,:3,3] - self.chain.conf[idA,:3,3])
            self.chain.conf[i,:3,3] = self.chain.conf[idA,:3,3] + v
        
        if self.bpstep.trace_termini:
            v = self.chain.conf[idB,:3,3] - self.chain.conf[idA,:3,3]
            vp = G @ v
            dr = vp - v
        
        for i in range(idB,self.nbp):
            self.chain.conf[i,:3,3] += dr
         
        
        # if np.abs(np.sum(self.chain.conf[idBn,:3,:3]-TBn_rot[:3,:3])) > 1e-10:
        #     print('triad Bn is wrong!')
        #     print(TBn_rot)
        #     print(self.chain.conf[idBn])
        #     sys.exit()
        
        # if np.abs(np.sum(self.chain.conf[-1,:3,:3]-tau_last[:3,:3])) > 1e-10:
        #     print('last triad inconsistent')
        #     print(tau_last)
        #     print(self.chain.conf[-1])
        #     sys.exit()
   
        # self.bpstep.set_move(True)
        # if not self.bpstep.check_deform_consistency():
        #     print(idA,idB)
        #     sys.exit()
               
        self.moved_intervals = [[0,idA,0],[idA+1,idBn,1000],[idB,self.nbp-1,1]]  
        return True

    #########################################################################################
    #########################################################################################


# @cond_jit
# def segment_rotation(
#     conf: np.ndarray,
#     Rlab: np.ndarray,
#     nbp: int,
#     idA: int,
#     hingedist: int,
# ) -> np.ndarray:
#     # move positions
#     rotated_vecs = np.zeros((hingedist - 2, 3))
#     id = idA % nbp
#     for i in range(hingedist - 2):
#         idm1 = id
#         id = (id + 1) % nbp
#         rotated_vecs[i] = Rlab @ (conf[id, :3, 3] - conf[idm1, :3, 3])

#     id = idA % nbp
#     for i in range(hingedist - 2):
#         idm1 = id
#         id = (id + 1) % nbp
#         conf[id, :3, 3] = conf[idm1, :3, 3] + rotated_vecs[i]
#         conf[id, :3, :3] = Rlab @ conf[id, :3, :3]
#     return conf

