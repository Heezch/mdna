import glob
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from .... import math as so3


class GenStiffness:
    def __init__(self, method: str = "md"):
        self.method = method
        self.load_from_file()

    def load_from_file(self):
        if self.method.lower() == "md":
            path = os.path.join(
                os.path.dirname(__file__), "Parametrization/MolecularDynamics"
            )
        elif "crystal" in self.method.lower():
            path = os.path.join(
                os.path.dirname(__file__), "Parametrization/Crystallography"
            )
        else:
            raise ValueError(f'Unknown method "{self.method}".')
        bases = "ATCG"
        self.dimers = {}
        for b1 in bases:
            for b2 in bases:
                seq = b1 + b2
                self.dimers[seq] = self.read_dimer(seq, path)

    def read_dimer(self, seq: str, path: str):
        fnstiff = glob.glob(path + "/Stiffness*" + seq + "*")[0]
        fnequi = glob.glob(path + "/Equilibrium*" + seq + "*")[0]

        equi = np.loadtxt(fnequi)
        stiff = np.loadtxt(fnstiff)
        equi_triad = so3.se3_midstep2triad(equi)
        stiff_group = so3.se3_algebra2group_stiffmat(
            equi, stiff, translation_as_midstep=True
        )
        dimer = {
            "seq": seq,
            "group_gs": equi_triad,
            "group_stiff": stiff_group,
            "equi": equi,
            "stiff": stiff,
        }
        return dimer

    def gen_params(self, seq: str, use_group: bool = False, sparse: bool = True):
        N = len(seq) - 1
        if sparse:
            stiff = lil_matrix((6 * N, 6 * N))
        else:
            stiff = np.zeros((6 * N, 6 * N))
        gs = np.zeros((N, 6))
        for i in range(N):
            bp = seq[i : i + 2].upper()
            if use_group:
                pstiff = self.dimers[bp]["group_stiff"]
                pgs = self.dimers[bp]["group_gs"]
            else:
                pstiff = self.dimers[bp]["stiff"]
                pgs = self.dimers[bp]["equi"]

            stiff[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] = pstiff
            gs[i] = pgs
        
        if sparse:
            stiff = stiff.tocsc()
        return stiff,gs