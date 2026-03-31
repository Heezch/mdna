import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numba import njit

from ..utils import random_unitsphere
from ..BPStep.BPStep import BPStep
from ..chain import Chain
from ..ExVol.ExVol import ExVol
from ..Constraints.Constraint import Constraint
from ... import math as so3
from .mcstep import MCStep


class MidstepMove(MCStep):
    def __init__(
        self,
        chain: Chain,
        bpstep: BPStep,
        midstep_triad_ids: List,
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
        self.name = "MidstepMove"

        self.midstep_triad_ids = sorted(midstep_triad_ids)
        for i in range(len(midstep_triad_ids)):
            if (
                i < len(midstep_triad_ids) - 1
                and self.midstep_triad_ids[i + 1] - self.midstep_triad_ids[i] <= 1
            ):
                raise ValueError(
                    f"MidstepMove: Current restrained midstep triads need to be at least 2 triads apart."
                )
            if (
                not self.closed
                and self.midstep_triad_ids[i] < 0
                or self.midstep_triad_ids[i] > self.nbps
            ):
                raise ValueError(
                    f"MidstepMove: Midstep triad assignment out of bounds ({self.midstep_triad_ids[i]})"
                )
            if not self.closed and self.midstep_triad_ids[i] == self.nbps:
                raise ValueError(
                    f"MidstepMove: Cannot assign midstep triad fix to last triad for open configuration. (nbp: {self.nbp}, triad assignment: {self.midstep_triad_ids[i]})"
                )

        MCS_MST_MAX_THETA = 0.15
        MCS_MST_MAX_TRANS = 0.015
        self.max_theta = MCS_MST_MAX_THETA * np.sqrt(self.chain.temp / 300)
        self.max_trans = MCS_MST_MAX_TRANS * np.sqrt(self.chain.temp / 300)

        self.requires_ev_check = True

        # self.moved_intervals = np.zeros((3, 3),dtype=int)
        # self.moved_intervals[0, 2] = 0
        # self.moved_intervals[1, 2] = 1000
        # self.moved_intervals[2, 2] = 0
        # self.moved_intervals[0, 0] = 0
        # self.moved_intervals[2, 1] = self.nbp-1

    #########################################################################################
    #########################################################################################

    def mc_move(self) -> bool:
        #############################
        # select midstep triads
        id1 = self.midstep_triad_ids[np.random.randint(len(self.midstep_triad_ids))]
        id1m1 = (id1 - 1) % self.nbp
        id2 = (id1 + 1) % self.nbp
        id2p1 = (id2 + 1) % self.nbp

        # random translation
        s = random_unitsphere()
        dr = np.random.uniform(0, self.max_trans)
        dv = s * dr

        # random rotation
        s = random_unitsphere()
        theta = np.random.uniform(0, self.max_theta)
        Theta = s * theta
        G = so3.euler2rotmat(Theta)

        # select triads
        T1 = self.chain.conf[id1, :3, :3]
        T2 = self.chain.conf[id2, :3, :3]

        # rotate first triad
        T1p = G @ T1

        # calculate rotation of second triad
        H = so3.midstep(T1, T2)
        sqrtRp = T1p.T @ H
        T2p = H @ sqrtRp

        # assign new full SE3 triads
        tau1p = np.copy(self.chain.conf[id1])
        tau2p = np.copy(self.chain.conf[id2])
        tau1p[:3, :3] = T1p
        tau2p[:3, :3] = T2p
        tau1p[:3, 3] += dv
        tau2p[:3, 3] -= dv

        # propose moves
        if id1 > 0 or self.closed:
            self.bpstep.propose_move(id1m1, self.chain.conf[id1m1], tau1p)
        self.bpstep.propose_move(id1, tau1p, tau2p)
        self.bpstep.propose_move(id2, tau2p, self.chain.conf[id2p1])

        # calculate energy
        dE = self.bpstep.eval_delta_E()
        # print(dE)

        # metropolis step
        if np.random.uniform() >= np.exp(-dE):
            return False

        ##########################
        # assign changes
        self.chain.conf[id1] = tau1p
        self.chain.conf[id2] = tau2p

        if id1 < id2:
            self.moved_intervals = [[0,id1-1,0],[id1,id2,1000],[id2+1,self.nbp-1,0]]
        else:
            self.moved_intervals = [[0,id2,1000],[id2+1,id1-1,0], [id1,self.nbp-1,1000]]
        return True


if __name__ == "__main__":
    from ..BPStep.BPS_RBP import RBP

    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    npb = 12
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
    msm = MidstepMove(ch, bps, [4, 8])

    Es = []

    import time

    t1 = time.time()

    for i in range(100000):
        msm.mc()
        if i % 1000 == 0:
            print(f"{i}: {msm.acceptance_rate()}")
            Es.append(bps.get_total_energy())

    t2 = time.time()

    print(f"dt = {t2-t1}")

    print(f"<E> / DoFs = {np.mean(Es)/(len(msm.chain.conf)-1)/6}")
