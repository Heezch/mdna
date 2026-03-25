import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..BPStep.BPStep import BPStep
from ..chain import Chain
from ..ExVol.ExVol import ExVol
from ..Constraints.Constraint import Constraint
from ... import math as so3


class MCStep(ABC):
    def __init__(
        self,
        chain: Chain,
        bpstep: BPStep,
        full_trial_conf: bool = False,
        exvol: ExVol = None,
        constraints: List[Constraint] = []
    ):
        """
        Initiate:
            - configuration
            - energy model (separate object passed to chain)

        """
        self.name = "MCStep"

        self.chain = chain
        self.bpstep = bpstep

        self.nbp = self.chain.nbp
        self.nbps = self.chain.nbps
        self.closed = self.chain.closed

        self.full_trial_conf = full_trial_conf

        self.count_steps = 0
        self.count_accept = 0

        # moved list
        self.moved_intervals = []

        # backup setting
        self.backup_required = False

        # excluded volume
        self.exvol = exvol
        self.requires_ev_check = False
        if exvol is None:
            self.ev_active = False
        else:
            self.ev_active = True

        # constraints
        self.constraints = constraints
        self.constraint_active = (not len(constraints) == 0)

    #########################################################################################
    #########################################################################################

    def mc(self) -> bool:
        self.count_steps += 1

        if self.full_trial_conf:
            raise NotImplementedError(
                f"Generation of full trial configurations has not been implemented yet."
            )
        else:
            # these conditions can probably be improved!
            accepted = self.mc_move()
            if accepted:
                if self.constraint_active:
                    accepted = self.check_constraints()

                if accepted and self.ev_active and self.requires_ev_check:
                    accepted = self.exvol.check(self.moved_intervals)
                    
                if not accepted:
                    self.chain.revert_to_backup()

            if accepted:
                self.count_accept += 1
                if self.backup_required or self.ev_active:
                    self.chain.set_backup()
                      
        self.bpstep.set_move(accepted)
        
        # diff = np.abs(np.sum(self.bpstep.current_last_triad[:3,:3]-self.chain.conf[-1,:3,:3]))
        # if diff > 1e-10:
        #     print('change of last triad not detected!')
        #     print(self.bpstep.current_last_triad[:3,:3])
        #     print(self.chain.conf[-1,:3,:3])
        #     print('proposed:')
        #     print(self.bpstep.proposed_last_triad)
        #     print(diff)
        #     print(f'accepted = {accepted}')
        #     print(self.name)
        #     if self.name == 'Pivot':
        #         print(f'{self.preserve_termini=}')
        #     for moved in self.moved_intervals:
        #         print(moved)
        #     sys.exit()
            
        # call chain maintainance methods
        # conf = np.copy(self.chain.conf)
        if self.chain.check_realign():
            self.bpstep.init_conf()
      
        return accepted

    #########################################################################################
    #########################################################################################

    def check_constraints(self) -> bool:
        for constraint in self.constraints:
            if not constraint.check(moved=self.moved_intervals):
                return False
        return True

    #########################################################################################
    #########################################################################################

    def acceptance_rate(self) -> float:
        return self.count_accept / self.count_steps

    #########################################################################################
    #########################################################################################

    @abstractmethod
    def mc_move(self) -> bool:
        pass

    # @abstractmethod
    # def gen_trial_conf(self) -> float:
    #     pass


if __name__ == "__main__":
    rng = np.random.normal(0, 1, 1)
    print(rng)

    ch = Chain(rng)

    # mcs = MCStep(ch)

    T1 = so3.se3_euler2rotmat(np.array([0.1, 0.2, 0.6, 0.2, 0.3, 0.1]))
    T2 = so3.se3_euler2rotmat(np.array([-0.1, 0.21, -0.35, 0.24, -0.124, 0.143]))

    import time

    T1i = so3.se3_inverse(T1)

    t1 = time.time()
    for i in range(10000):
        T1i = so3.se3_inverse(T1)

    t2 = time.time()
    for i in range(10000):
        T1i = np.linalg.inv(T1)

    t3 = time.time()

    print(f"dt1 = {t2-t1}")
    print(f"dt2 = {t3-t2}")
