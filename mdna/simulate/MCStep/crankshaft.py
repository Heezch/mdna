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


class Crankshaft(MCStep):
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
        self.name = "Crankshaft"
        self.numba_segment_rotation = use_numba

        MCS_CSROT_FAC = 0.25 
        self.sigma = MCS_CSROT_FAC * np.sqrt(
            self.chain.temp / 300
        )  # *np.sqrt(disc_len/0.34);

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
        # Trial Move
        theta = np.random.normal(0, self.sigma)
        Theta = self.chain.conf[idB, :3, 3] - self.chain.conf[idA, :3, 3]
        Theta = Theta / np.linalg.norm(Theta) * theta
        Rlab = so3.euler2rotmat(Theta)
        TA_rot = np.copy(self.chain.conf[idA])
        TA_rot[:3, :3] = Rlab @ self.chain.conf[idA, :3, :3]

        TBn_rot = np.copy(self.chain.conf[idBn])
        TBn_rot[:3, :3] = Rlab @ self.chain.conf[idBn, :3, :3]
        TBn_rot[:3, 3] = self.chain.conf[idB, :3, 3] - Rlab @ (
            self.chain.conf[idB, :3, 3] - self.chain.conf[idBn, :3, 3]
        )

        # propose moves
        self.bpstep.propose_move(idAn, self.chain.conf[idAn], TA_rot)
        self.bpstep.propose_move(idBn, TBn_rot, self.chain.conf[idB])

        # calculate energy
        dE = self.bpstep.eval_delta_E()

        # metropolis step
        if np.random.uniform() >= np.exp(-dE):
            return False

        ##########################
        # rotate inbetween segment

        # rotate triad at idA
        self.chain.conf[idA] = TA_rot
        self.chain.conf[idBn] = TBn_rot

        if self.numba_segment_rotation:
            self.chain.conf = segment_rotation(
                self.chain.conf, Rlab, self.nbp, idA, hingedist
            )
        else:
            # rotate triads
            self.chain.conf[
                np.arange(idA + 1, idA + hingedist - 1) % self.nbp, :3, :3
            ] = (
                Rlab
                @ self.chain.conf[
                    np.arange(idA + 1, idA + hingedist - 1) % self.nbp, :3, :3
                ]
            )
            # move positions
            vecs = (
                self.chain.conf[
                    np.arange(idA + 1, idA + hingedist - 1) % self.nbp, :3, 3
                ]
                - self.chain.conf[np.arange(idA, idA + hingedist - 2) % self.nbp, :3, 3]
            )
            rotated_vecs = (Rlab @ vecs.T).T
            id = idA % self.nbp
            for i in range(hingedist - 2):
                idm1 = id
                id = (id + 1) % self.nbp
                self.chain.conf[id, :3, 3] = (
                    self.chain.conf[idm1, idAp1:3, 3] + rotated_vecs[i]
                )

        # self.moved_intervals[0, 0] = (idA + 1) % self.nbp
        # self.moved_intervals[0, 1] = idBn

        idAp1 = (idA + 1) % self.nbp
        if idAp1 < idBn:
            self.moved_intervals = [[0,idAp1-1,0],[idAp1,idBn,1000],[idBn+1,self.nbp-1,0]]
        else:
            self.moved_intervals = [[0,idBn,1000],[idAp1,self.nbp-1,1000],[idBn+1,idAp1-1,0]]
            
        return True

    #########################################################################################
    #########################################################################################


@cond_jit
def segment_rotation(
    conf: np.ndarray,
    Rlab: np.ndarray,
    nbp: int,
    idA: int,
    hingedist: int,
) -> np.ndarray:
    # move positions
    rotated_vecs = np.zeros((hingedist - 2, 3))
    id = idA % nbp
    for i in range(hingedist - 2):
        idm1 = id
        id = (id + 1) % nbp
        rotated_vecs[i] = Rlab @ (conf[id, :3, 3] - conf[idm1, :3, 3])

    id = idA % nbp
    for i in range(hingedist - 2):
        idm1 = id
        id = (id + 1) % nbp
        conf[id, :3, 3] = conf[idm1, :3, 3] + rotated_vecs[i]
        conf[id, :3, :3] = Rlab @ conf[id, :3, :3]
    return conf


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

    # cs = Crankshaft(ch,bps,2,16,range_id1=18,range_id2=4)
    cs = Crankshaft(ch, bps, 2, 200)

    Es = []

    for i in range(100000):
        cs.mc()
        if i % 1000 == 0:
            print(f"{i}: {cs.acceptance_rate()}")
            Es.append(cs.bpstep.get_total_energy())

    print(f"<E> / DoFs = {np.mean(Es)/(len(cs.chain.conf)-1)/6}")
