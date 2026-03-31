#!/bin/env python3

import numpy as np

from .generators import hat_map, vec_map  # , generator1, generator2, generator3
from .._jit import cond_jit

##########################################################################################################
############### SO3 Methods ##############################################################################
##########################################################################################################


@cond_jit
def cayley2rotmat(cayley: np.ndarray) -> np.ndarray:
    """Transforms cayley vector to corresponding rotation matrix

    Args:
        cayley (np.ndarray): Cayley vector

    Returns:
        np.ndarray: rotation matrix
    """
    hat = hat_map(cayley)
    return np.eye(3) + 4.0 / (4 + np.dot(cayley, cayley)) * (
        hat + 0.5 * np.dot(hat, hat)
    )


@cond_jit
def rotmat2cayley(rotmat: np.ndarray) -> np.ndarray:
    """Transforms rotation matrix to corresponding Cayley vector

    Args:
        rotmat (np.ndarray): element of SO(3)

    Returns:
        np.ndarray: returns 3-vector
    """
    return 2.0 / (1 + np.trace(rotmat)) * vec_map(rotmat - rotmat.T)


##########################################################################################################
############### SE3 Methods ##############################################################################
##########################################################################################################


def se3_cayley2rotmat(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    if Omega.shape != (6,):
        raise ValueError(f"Expected shape (6,) array, but encountered {Omega.shape}.")
    if rotation_first:
        vrot = Omega[:3]
        vtrans = Omega[3:]
    else:
        vrot = Omega[3:]
        vtrans = Omega[:3]
    rotmat = np.zeros((4, 4))
    rotmat[:3, :3] = cayley2rotmat(vrot)
    rotmat[:3, 3] = vtrans
    rotmat[3, 3] = 1
    return rotmat


def se3_rotmat2cayley(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    if R.shape != (4, 4):
        raise ValueError(f"Expected shape (4,4) array, but encountered {R.shape}.")
    vrot = rotmat2cayley(R[:3, :3])
    vtrans = R[:3, 3]
    if rotation_first:
        return np.concatenate((vrot, vtrans))
    else:
        return np.concatenate((vtrans, vrot))
