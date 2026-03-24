import os
import sys

import numpy as np

from .... import math as so3


def read_nucleosome_triads(fn: str) -> np.ndarray:
    data = np.loadtxt(fn)
    N = len(data) // 12
    nuctriads = np.zeros((N, 4, 4))
    for i in range(N):
        tau = np.eye(4)
        pos = data[i * 12 : i * 12 + 3] / 10
        triad = data[i * 12 + 3 : i * 12 + 12].reshape((3, 3))
        triad = so3.euler2rotmat(so3.rotmat2euler(triad))
        tau[:3, :3] = triad
        tau[:3, 3] = pos
        nuctriads[i] = tau
    return nuctriads


if __name__ == "__main__":
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    triadfn = os.path.join(os.path.dirname(__file__), "State/Nucleosome.state")
    nuctriads = read_nucleosome_triads(triadfn)
    for nuct in nuctriads:
        print(nuct)
