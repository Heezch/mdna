import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .._jit import cond_jit
from .. import math as so3


@cond_jit
def random_unitsphere():
    a = 2 * np.pi * np.random.uniform()
    x = np.random.uniform()
    b = np.arccos(1 - 2 * x)
    vec = np.ones(3)
    vec[:2] *= np.sin(b)
    vec[2] *= np.cos(b)
    vec[0] *= np.cos(a)
    vec[1] *= np.sin(a)
    return vec

@cond_jit
def params2conf(params: np.ndarray) -> np.ndarray:
    conf = np.zeros((len(params), 4, 4), dtype=np.float64)
    conf[0] = np.eye(4)
    for i in range(1, len(params)):
        conf[i] = conf[i - 1] @ so3.se3_euler2rotmat(params[i])
    return conf

@cond_jit
def triad_realign(triad: np.ndarray) -> np.ndarray:
    triad = np.copy(triad)
    e1 = triad[:,0]
    e2 = triad[:,1]
    e3 = triad[:,2]
    e3 = e3 / np.linalg.norm(e3)
    e2 = e2 - np.dot(e3,e2)*e3
    e2 = e2 / np.linalg.norm(e2)
    e1 = np.cross(e2,e3)
    e1 = e1 / np.linalg.norm(e1)
    triad = np.empty((3,3), dtype=np.float64)
    triad[:,0] = e1
    triad[:,1] = e2
    triad[:,2] = e3
    return triad