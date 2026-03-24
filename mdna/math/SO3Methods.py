#!/bin/env python3

import numpy as np

from .._jit import cond_jit

"""
This is a legacy module and will be phased out in the near future
"""


DEF_SO3M_EPSILON = 1e-12
DEF_S03M_CLOSE_TO_ONE = 0.999999999999
DEF_S03M_CLOSE_TO_MINUS_ONE = -0.999999999999


@cond_jit
def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
    """
    Returns the matrix version of the Euler-Rodrigues formula
    """
    Om = np.linalg.norm(Omega)
    R = np.zeros((3, 3), dtype=np.double)

    # if norm is zero, return identity matrix
    if Om < 1e-12:
        R[0, 0] = 1
        R[1, 1] = 1
        R[2, 2] = 1
        return R

    cosOm = np.cos(Om)
    sinOm = np.sin(Om)
    Omsq = Om * Om
    cosOm = np.cos(Om)
    sinOm = np.sin(Om)
    Omsq = Om * Om
    fac1 = (1 - cosOm) / Omsq
    fac2 = sinOm / Om

    R[0, 0] = cosOm + Omega[0] ** 2 * fac1
    R[1, 1] = cosOm + Omega[1] ** 2 * fac1
    R[2, 2] = cosOm + Omega[2] ** 2 * fac1
    A = Omega[0] * Omega[1] * fac1
    B = Omega[2] * fac2
    R[0, 1] = A - B
    R[1, 0] = A + B
    A = Omega[0] * Omega[2] * fac1
    B = Omega[1] * fac2
    R[0, 2] = A + B
    R[2, 0] = A - B
    A = Omega[1] * Omega[2] * fac1
    B = Omega[0] * fac2
    R[1, 2] = A - B
    R[2, 1] = A + B
    return R


@cond_jit
def phi2rotz(phi: float) -> np.ndarray:
    """
    rotation matrix for rotation over z-axis
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    R = np.zeros((3, 3), dtype=np.double)
    R[0, 0] = cp
    R[1, 1] = cp
    R[0, 1] = -sp
    R[1, 0] = sp
    R[2, 2] = 1
    return R


@cond_jit
def phi2rotx(phi: float) -> np.ndarray:
    """
    rotation matrix for rotation over x-axis
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    R = np.zeros((3, 3), dtype=np.double)
    R[1, 1] = cp
    R[2, 2] = cp
    R[1, 2] = -sp
    R[2, 1] = sp
    R[0, 0] = 1
    return R


@cond_jit
def phi2roty(phi: float) -> np.ndarray:
    """
    rotation matrix for rotation over y-axis
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    R = np.zeros((3, 3), dtype=np.double)
    R[2, 2] = cp
    R[0, 0] = cp
    R[2, 0] = -sp
    R[0, 2] = sp
    R[1, 1] = 1
    return R


@cond_jit
def rotmat2euler(R: np.ndarray) -> np.ndarray:
    """
    Inversion of Euler Rodriguez Formula
    """
    val = 0.5 * (np.trace(R) - 1)
    if val > DEF_S03M_CLOSE_TO_ONE:
        return np.zeros(3)
    if val < DEF_S03M_CLOSE_TO_MINUS_ONE:
        if R[0, 0] > DEF_S03M_CLOSE_TO_ONE:
            return np.array([np.pi, 0, 0])
        if R[1, 1] > DEF_S03M_CLOSE_TO_ONE:
            return np.array([0, np.pi, 0])
        return np.array([0, 0, np.pi])
    Th = np.arccos(val)
    Theta = np.array([(R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])])
    Theta = Th * 0.5 / np.sin(Th) * Theta
    return Theta
