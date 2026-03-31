from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..chain import Chain
from ... import math as so3
from .BPStep import BPStep
from .RBPStiff import GenStiffness

# local RBP model

class LRBP(BPStep):
    def __init__(
        self,
        chain: Chain,
        sequence: str,
        specs: Dict,
        closed: bool = False,
        static_group: bool = True,
        temp: float = 300,
        conserve_twist: bool = True
    ):        
        # set reference temperature for dataset
        self.specs = specs
        super().__init__(chain, sequence, closed, static_group, temp=temp,conserve_twist=conserve_twist)

    #########################################################################################
    #########################################################################################

    def _init_params(self) -> None:
        self.temp = 300
        genstiff = GenStiffness(method=self.specs["method"])
        self.stiffmats = np.zeros((self.nbps, 6, 6))
        self.gs_vecs = np.zeros((self.nbps, 6))
        if self.closed:
            extseq = self.sequence + self.sequence[0]
        else:
            extseq = self.sequence
        for i in range(self.nbps):
            self.gs_vecs[i] = genstiff.dimers[extseq[i : i + 2]][self.specs["gs_key"]]
            self.stiffmats[i] = genstiff.dimers[extseq[i : i + 2]][
                self.specs["stiff_key"]
            ]
        self.current_energies = np.zeros(self.nbps)
        self.proposed_energies = np.zeros(self.nbps)

    #########################################################################################
    #########################################################################################

    def _eval_delta_E(self) -> float:
        dE = 0
        for id in self.proposed_ids:
            # beta E = 0.5 Om.T @ M @ Om
            self.proposed_energies[id] = (
                0.5
                * self.proposed_deforms[id].T
                @ self.stiffmats[id]
                @ self.proposed_deforms[id]
            )
            dE += self.proposed_energies[id] - self.current_energies[id]
        return dE

    #########################################################################################
    #########################################################################################

    def set_energy(self, accept: bool = True) -> None:
        if accept:
            self.current_energies = np.copy(self.proposed_energies)
        else:
            self.proposed_energies = np.copy(self.current_energies)

    #########################################################################################
    #########################################################################################

    def _get_total_energy(self) -> float:
        return np.sum(self.current_energies)

    #########################################################################################
    #########################################################################################

    def _set_temperature(self, temp: float) -> None:
        self.stiffmats *= self.temp / temp
        self.init_conf()

