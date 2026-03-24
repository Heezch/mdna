import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..aux import random_unitsphere
from ..BPStep.BPStep import BPStep
from ..chain import Chain
from ..ExVol.ExVol import ExVol
from ..Constraints.Constraint import Constraint
from ..._jit import cond_jit
from ... import math as so3
from .mcstep import MCStep


class ClusterTrans(MCStep):
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
        constraints: List[Constraint] = []
    ):
        """
        Initiate:
            - configuration
            - energy model (separate object passed to chain)

        """
        super().__init__(chain, bpstep, full_trial_conf, exvol=exvol,constraints=constraints)
        self.name = "ClusterTrans"
        self.USE_NUMBA_SEGMENT_ROTATION = True

        MCS_MST_MAX_TRANS = 0.1
        self.max_trans = MCS_MST_MAX_TRANS * np.sqrt(self.chain.temp / 300)

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
                    f"ClusterTrans: range limit requires both range_id1 and range_id2 to be set."
                )
            if not self.closed and range_id1 >= range_id2:
                raise ValueError(
                    f"ClusterTrans: requires range_id1 < range_id2 if chain is not closed."
                )
            range_id2 += 1
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
                    f"ClusterTrans: range {self.range_id1} - {self.range_id2} too close for min range {self.selrange_min}."
                )

        self.requires_ev_check = True
        self.moved_intervals = np.zeros((1, 3),dtype=int)
        self.moved_intervals[0, 2] = 1000

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
            idB = (idA + hingedist) % self.nbp
            idA = idA % self.nbp
        else:
            if self.closed:
                idA = np.random.randint(0, self.nbp)
                hingedist = np.random.randint(self.selrange_min, self.selrange_max)
                idB = (idA + hingedist) % self.nbp
            else:
                idA = np.random.randint(1, self.nbp - self.selrange_min)
                dist = self.nbp - idA
                if dist > self.selrange_max:
                    hingedist = np.random.randint(self.selrange_min, self.selrange_max)
                else:
                    hingedist = np.random.randint(self.selrange_min, dist)
                idB = idA + hingedist

        idAn = (idA - 1) % self.nbp
        idBn = (idB - 1) % self.nbp

        #############################
        # random translation
        s = random_unitsphere()
        dr = np.random.uniform(0, self.max_trans)
        dv = s * dr

        tau1 = np.copy(self.chain.conf[idA])
        tau2 = np.copy(self.chain.conf[idBn])
        tau1[:3, 3] += dv
        tau2[:3, 3] += dv

        # propose moves
        self.bpstep.propose_move(idAn, self.chain.conf[idAn], tau1)
        self.bpstep.propose_move(idBn, tau2, self.chain.conf[idB])

        # calculate energy
        dE = self.bpstep.eval_delta_E()

        # metropolis step
        if np.random.uniform() >= np.exp(-dE):
            return False

        ##########################
        # rotate inbetween segment
        self.chain.conf[np.arange(idA, idA + hingedist) % self.nbp, :3, 3] += dv

        ##########################
        # set moved intervals for excluded volume check
        # self.moved_intervals[0, 0] = idA
        # self.moved_intervals[0, 1] = idBn

        # idAp1 = (idA + 1) % self.nbp
        if idA < idBn:
            self.moved_intervals = [[0,idA-1,0],[idA,idBn,1000],[idBn+1,self.nbp-1,0]]
        else:
            self.moved_intervals = [[0,idBn,1000],[idA,self.nbp-1,1000],[idBn+1,idA-1,0]]

        return True


if __name__ == "__main__":
    from ..BPStep.BPS_RBP import RBP

    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    npb = 1000
    closed = False
    conf = np.zeros((npb, 4, 4))
    gs = np.array([0, 0, 0.6, 0, 0, 0.34])
    g = so3.se3_euler2rotmat(gs)
    conf[0] = np.eye(4)
    for i in range(1, npb):
        g = so3.se3_euler2rotmat(gs + np.random.normal(0, 0.1, 6))
        conf[i] = conf[i - 1] @ g

    seq = "".join(["ATCG"[np.random.randint(4)] for i in range(npb)])
    specs = {"method": "MD", "gs_key": "group_gs", "stiff_key": "group_stiff"}

    ch = Chain(conf, keep_backup=True, closed=closed)
    bps = RBP(ch, seq, specs, closed=closed, static_group=True)

    # tau1 = np.copy(ch.conf[9])
    # tau2 = np.copy(ch.conf[10])
    # tau3 = np.copy(ch.conf[18])
    # tau4 = np.copy(ch.conf[19])
    # tau5 = np.copy(ch.conf[20])

    # cs = Crankshaft(ch,bps,2,16,range_id1=18,range_id2=4)
    ct = ClusterTrans(ch, bps, 2, 200, range_id1=10, range_id2=20)
    Es = []

    for i in range(100000):
        ct.mc()
        if i % 1000 == 0:
            print(f"{i}: {ct.acceptance_rate()}")
            Es.append(ct.bpstep.get_total_energy())
            # print(np.sum(tau1-ch.conf[9]))
            # print(np.sum(tau2-ch.conf[10]))
            # print(np.sum(tau3-ch.conf[18]))
            # print(np.sum(tau4-ch.conf[19]))
            # print(np.sum(tau5-ch.conf[20]))

    print(f"<E> / DoFs = {np.mean(Es)/(len(ct.chain.conf)-1)/6}")
    print(np.mean(Es))
