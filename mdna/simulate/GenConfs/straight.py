from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from ... import math as so3
import sys

def gen_straight(gs: np.ndarray,dlk: float = 0) -> np.ndarray:
    if len(gs.shape) == 1:
        gs = gs.reshape((len(gs)//6,6))
    tw_per_step = dlk / len(gs) * 2 * np.pi
    gs = np.copy(gs)
    nbp = len(gs) + 1
    conf = np.zeros((nbp, 4, 4))
    conf[0] = np.eye(4)
    for i in range(1, nbp):
        strgs = gs[i-1]
        strgs[:2] = 0
        strgs[2] += tw_per_step
        g = so3.se3_euler2rotmat(strgs)
        conf[i] = conf[i - 1] @ g
    return conf