#!/bin/env python3

import numpy as np

from .._jit import cond_jit


@cond_jit
def hat_map(x: np.ndarray) -> np.ndarray:
    """Maps rotation vectors (Euler vectors) onto the corresponding elements of so(3).

    Args:
        x (np.ndarray): euler vector (3-vector)

    Returns:
        np.ndarray: rotation amtrix (element of SO(3))
    """
    X = np.zeros((3, 3))
    X[0, 1] = -x[2]
    X[1, 0] = x[2]
    X[0, 2] = x[1]
    X[2, 0] = -x[1]
    X[1, 2] = -x[0]
    X[2, 1] = x[0]
    return X


@cond_jit
def vec_map(X: np.ndarray) -> np.ndarray:
    """Inverse of the hat map. Maps elements of so(3) onto the corresponding Euler vectors.

    Args:
        X (np.ndarray): generator of SO(3) (element of so(3)). This should be a skewsymmetric matrix

    Returns:
        np.ndarray: rotation vector (3-vector)
    """
    return np.array([X[2, 1], X[0, 2], X[1, 0]])


@cond_jit
def generator1() -> np.ndarray:
    """first generator of SO(3)

    Returns:
        np.ndarray: L1
    """
    X = np.zeros((3, 3))
    X[1, 2] = -1
    X[2, 1] = 1
    return X


@cond_jit
def generator2() -> np.ndarray:
    """first generator of SO(3)

    Returns:
        np.ndarray: L1
    """
    X = np.zeros((3, 3))
    X[0, 2] = 1
    X[2, 0] = -1
    return X


@cond_jit
def generator3() -> np.ndarray:
    """first generator of SO(3)

    Returns:
        np.ndarray: L1
    """
    X = np.zeros((3, 3))
    X[0, 1] = -1
    X[1, 0] = 1
    return X
