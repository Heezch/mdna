#!/bin/env python3

import numpy as np

from .._jit import cond_jit

DEF_EULER_EPSILON = 1e-12
DEF_EULER_CLOSE_TO_ONE = 0.999999999999
DEF_EULER_CLOSE_TO_MINUS_ONE = -0.999999999999

##########################################################################################################
############### SO3 Methods ##############################################################################
##########################################################################################################


@cond_jit
def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
    """Returns the matrix version of the Euler-Rodrigues formula

    Args:
        Omega (np.ndarray): Euler vector / Rotation vector (3-vector)

    Returns:
        np.ndarray: Rotation matrix (element of SO(3))
    """
    Om = np.linalg.norm(Omega)
    R = np.zeros((3, 3), dtype=np.double)

    # if norm is zero, return identity matrix
    if Om < DEF_EULER_EPSILON:
        R[0, 0] = 1
        R[1, 1] = 1
        R[2, 2] = 1
        return R

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


# @cond_jit
# def rotmat2euler(R: np.ndarray) -> np.ndarray:
#     """Inversion of Euler Rodriguez Formula

#     Args:
#         R (np.ndarray): Rotation matrix (element of SO(3))

#     Returns:
#         np.ndarray: Euler vector / Rotation vector (3-vector)
#     """
#     val = 0.5 * (np.trace(R) - 1)
#     if val > DEF_EULER_CLOSE_TO_ONE:
#         return np.zeros(3)
#     if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
#         if R[0, 0] > DEF_EULER_CLOSE_TO_ONE:
#             return np.array([np.pi, 0, 0])
#         if R[1, 1] > DEF_EULER_CLOSE_TO_ONE:
#             return np.array([0, np.pi, 0])
#         return np.array([0, 0, np.pi])
#     Th = np.arccos(val)
#     Theta = np.array([(R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])])
#     Theta = Th * 0.5 / np.sin(Th) * Theta
#     return Theta

@cond_jit
def rotmat2euler(R: np.ndarray) -> np.ndarray:
    """Inversion of Euler Rodriguez Formula

    Args:
        R (np.ndarray): Rotation matrix (element of SO(3))

    Returns:
        np.ndarray: Euler vector / Rotation vector (3-vector)
    """
    val = 0.5 * (np.trace(R) - 1)
    if val > DEF_EULER_CLOSE_TO_ONE:
        return np.zeros(3)
    if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
        # rotation around first axis by angle pi
        if R[0, 0] > DEF_EULER_CLOSE_TO_ONE:
            return np.array([np.pi, 0, 0])
        # rotation around second axis by angle pi
        if R[1, 1] > DEF_EULER_CLOSE_TO_ONE:
            return np.array([0, np.pi, 0])
        # rotation around third axis by angle pi
        if R[2, 2] > DEF_EULER_CLOSE_TO_ONE:
            return np.array([0, 0, np.pi])
        # rotation around arbitrary axis by angle pi
        A = R - np.eye(3)       
        b = np.cross(A[0],A[1])
        th = b - np.dot(b,A[2])*A[2]
        th = th / np.linalg.norm(th) * np.pi
        return th
    Th = np.arccos(val)
    Theta = np.array([(R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])])
    Theta = Th * 0.5 / np.sin(Th) * Theta
    return Theta


#########################################################################################################
############## sqrt of rotation matrix ##################################################################
#########################################################################################################


@cond_jit
def sqrt_rot(R: np.ndarray) -> np.ndarray:
    """generates rotation matrix that corresponds to a rotation over the same axis, but over half the angle."""
    return euler2rotmat(0.5 * rotmat2euler(R))


@cond_jit
def midstep(triad1: np.ndarray, triad2: np.ndarray) -> np.ndarray:
    return triad1 @ sqrt_rot(triad1.T @ triad2)


##########################################################################################################
############### SE3 Methods ##############################################################################
##########################################################################################################


@cond_jit
def se3_euler2rotmat(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    # if Omega.shape != (6,):
    #     raise ValueError(f'Expected shape (6,) array, but encountered {Omega.shape}.')
    if rotation_first:
        vrot = Omega[:3]
        vtrans = Omega[3:]
    else:
        vrot = Omega[3:]
        vtrans = Omega[:3]
    rotmat = np.zeros((4, 4))
    rotmat[:3, :3] = euler2rotmat(vrot)
    rotmat[:3, 3] = vtrans
    rotmat[3, 3] = 1
    return rotmat


@cond_jit
def se3_rotmat2euler(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    # if R.shape != (4,4):
    #     raise ValueError(f'Expected shape (4,4) array, but encountered {R.shape}.')
    vrot = rotmat2euler(R[:3, :3])
    vtrans = R[:3, 3]
    if rotation_first:
        return np.concatenate((vrot, vtrans))
    else:
        return np.concatenate((vtrans, vrot))
