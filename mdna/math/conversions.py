#!/bin/env python3

import numpy as np

from .generators import hat_map
from .._jit import cond_jit

# from .generators import hat_map, vec_map, generator1, generator2, generator3


@cond_jit
def cayley2euler(cayley: np.ndarray) -> np.ndarray:
    """Transforms Cayley vector to corresponding Euler vector

    Args:
        cayley (np.ndarray): Cayley vector (3-vector)

    Returns:
        np.ndarray: Euler vector (3-vector)
    """
    norm = np.linalg.norm(cayley)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.arctan(0.5 * norm) / norm * cayley


@cond_jit
def cayley2euler_factor(cayley: np.ndarray) -> float:
    """Transforms Cayley vector to corresponding Euler vector

    Args:
        cayley (np.ndarray): Cayley vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(cayley)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.arctan(0.5 * norm) / norm


@cond_jit
def euler2cayley(euler: np.ndarray) -> np.ndarray:
    """Transforms Euler vector to corresponding Cayley vector

    Args:
        euler (np.ndarray): Euler vector (3-vector)

    Returns:
        np.ndarray: Cayley vector (3-vector)
    """
    norm = np.linalg.norm(euler)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.tan(0.5 * norm) / norm * euler


@cond_jit
def euler2cayley_factor(euler: np.ndarray) -> float:
    """Transforms Euler vector to corresponding Cayley vector

    Args:
        euler (np.ndarray): Euler vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(euler)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.tan(0.5 * norm) / norm


##########################################################################################################
############### Linear Transformations based on linear expansion #########################################
##########################################################################################################


@cond_jit
def cayley2euler_linearexpansion(cayley_gs: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Cayley to Euler vector around a given groundstate vector

    Args:
        cayley_gs (np.ndarray): The Cayley vector around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    cayley_norm = np.linalg.norm(cayley_gs)
    cayley_norm_sq = cayley_norm**2
    ratio_euler_cayley = 2 * np.arctan(0.5 * cayley_norm) / cayley_norm
    fac = (4.0 / (4 + cayley_norm_sq) - ratio_euler_cayley) / cayley_norm_sq
    return np.eye(3) * ratio_euler_cayley + np.outer(cayley_gs, cayley_gs) * fac

    # cnorm = np.linalg.norm(cayley_gs)
    # csq = cnorm**2
    # fac = 2./csq*(2./(4+csq) - np.arctan(cnorm/2)/cnorm)
    # mat = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         mat[i,j] = fac * cayley_gs[i] * cayley_gs[j]
    #     mat[i,i] += 2*np.arctan(0.5*cnorm)/cnorm
    # return mat


@cond_jit
def euler2cayley_linearexpansion(euler_gs: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Euler to Cayley vector around a given groundstate vector

    Args:
        cayley_gs (np.ndarray): The Euler vector around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    euler_norm = np.linalg.norm(euler_gs)
    euler_norm_sq = euler_norm**2
    ratio_cayley_euler = 2 * np.tan(0.5 * euler_norm) / euler_norm
    fac = (1.0 / (np.cos(0.5 * euler_norm)) ** 2 - ratio_cayley_euler) / euler_norm_sq
    return np.eye(3) * ratio_cayley_euler + np.outer(euler_gs, euler_gs) * fac

    # enorm = np.linalg.norm(euler_gs)
    # esq = enorm**2
    # fac = 1./esq*( 1./np.cos(0.5*enorm) - 2*np.tan(0.5*enorm)/enorm )
    # mat = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         mat[i,j] = fac * euler_gs[i] * euler_gs[j]
    #     mat[i,i] += 2*np.tan(0.5*enorm)/enorm
    # return mat


##########################################################################################################
############### Change Splitting between static and dynamic components ###################################
##########################################################################################################


@cond_jit
def splittransform_group2algebra(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) to lie algebra splitting
    representation R = exp(hat(Theta_0) + hat(Delta')). Linear transformation T transforms Delta
    into Delta' as T*Delta = Delta'.

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians

    Returns:
        float: Linear transformation matrix T (3x3) that transforms Delta into Delta': T*Delta = Delta'
    """
    htheta = hat_map(Theta_0)
    hthetasq = np.dot(htheta, htheta)

    accutheta = np.copy(htheta)
    # zeroth order
    T = np.eye(3)
    # first order
    T += 0.5 * accutheta
    # second order
    accutheta = np.dot(accutheta, htheta)
    T += 1.0 / 12 * accutheta
    # fourth order
    accutheta = np.dot(accutheta, hthetasq)
    T += -1.0 / 720 * accutheta
    # sixth order
    accutheta = np.dot(accutheta, hthetasq)
    T += 1.0 / 30240 * accutheta
    # eighth order
    accutheta = np.dot(accutheta, hthetasq)
    T += -1.0 / 1209600 * accutheta
    # tenth order
    accutheta = np.dot(accutheta, hthetasq)
    T += 1.0 / 47900160 * accutheta
    # twelth order
    accutheta = np.dot(accutheta, hthetasq)
    T += -691.0 / 1307674368000 * accutheta
    return T

    # htheta = hat_map(Theta_0)
    # # hthetasq = np.matmul(htheta,htheta)
    # hthetasq = np.dot(htheta,htheta)

    # accutheta = np.copy(htheta)
    # # first order
    # T = np.eye(3)
    # # seconds order
    # T += 0.5 * accutheta
    # # third order
    # # accutheta = np.matmul(accutheta,htheta)
    # accutheta = np.dot(accutheta,htheta)
    # T += 1./12 * accutheta
    # # fifth order
    # # accutheta = np.matmul(accutheta,htheta)
    # accutheta = np.dot(accutheta,htheta)
    # T += -1./720 * accutheta
    # # seventh order
    # # accutheta = np.matmul(accutheta,htheta)
    # accutheta = np.dot(accutheta,htheta)
    # T += 1./30240 * accutheta
    # # ninth order
    # # accutheta = np.matmul(accutheta,htheta)
    # accutheta = np.dot(accutheta,htheta)
    # T += -1./1209600 * accutheta
    # # eleventh order
    # # accutheta = np.matmul(accutheta,htheta)
    # accutheta = np.dot(accutheta,htheta)
    # T += 1./47900160 * accutheta
    # # thirteenth order
    # # accutheta = np.matmul(accutheta,htheta)
    # accutheta = np.dot(accutheta,htheta)
    # T += -691./1307674368000 * accutheta
    # return T


@cond_jit
def splittransform_algebra2group(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in lie algebra splitting representation R = exp(hat(Theta_0) + hat(Delta')) to group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) t. Linear transformation T transforms Delta
    into Delta' as T'*Delta' = Delta. Currently this is defined as the inverse of the transformation
    defined in the method splittransform_group2algebra

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians

    Returns:
        float: Linear transformation matrix T' (3x3) that transforms Delta into Delta': T'*Delta' = Delta
    """
    return np.linalg.inv(splittransform_group2algebra(Theta_0))
