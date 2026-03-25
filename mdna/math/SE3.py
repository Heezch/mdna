#!/bin/env python3

from typing import List

import numpy as np

from .conversions import splittransform_algebra2group, splittransform_group2algebra
from .Euler import euler2rotmat, rotmat2euler, se3_rotmat2euler, sqrt_rot
from .generators import hat_map
from .._jit import cond_jit


@cond_jit
def se3_inverse(g: np.ndarray) -> np.ndarray:
    """Inverse of element of SE3"""
    inv = np.zeros(g.shape)
    inv[:3, :3] = g[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ g[:3, 3]
    inv[3, 3] = 1
    return inv


@cond_jit
def se3_triads2rotmat(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1"""
    return se3_inverse(tau1) @ tau2


@cond_jit
def se3_triads2euler(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    return se3_rotmat2euler(se3_triads2rotmat(tau1, tau2))


@cond_jit
def se3_midstep2triad(triad_euler: np.ndarray) -> np.ndarray:
    midstep_euler = np.copy(triad_euler)
    vrot = triad_euler[:3]
    vtrans = triad_euler[3:]
    sqrt_rotmat = euler2rotmat(0.5 * vrot)
    midstep_euler[3:] = sqrt_rotmat @ vtrans
    return midstep_euler


@cond_jit
def se3_triad2midstep(midstep_euler: np.ndarray) -> np.ndarray:
    triad_euler = np.copy(midstep_euler)
    vrot = midstep_euler[:3]
    vtrans = midstep_euler[3:]
    sqrt_rotmat = euler2rotmat(0.5 * vrot)
    triad_euler[3:] = sqrt_rotmat.T @ vtrans
    return triad_euler


@cond_jit
def se3_triadxrotmat_midsteptrans(tau1: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Multiplication of triad with rotation matrix g (in SE3) assuming that the translation of g is defined with respect to the midstep triad."""
    R = g[:3, :3]
    T1 = tau1[:3, :3]
    tau2 = np.eye(4)
    tau2[:3, :3] = T1 @ R
    tau2[:3, 3] = tau1[:3, 3] + T1 @ sqrt_rot(R) @ g[:3, 3]
    return tau2


@cond_jit
def se3_triads2rotmat_midsteptrans(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1, assuming that the translation of g is defined with respect to the midstep triad."""
    T1 = tau1[:3, :3]
    T2 = tau2[:3, :3]
    R = T1.T @ T2
    Tmid = T1 @ sqrt_rot(R)
    zeta = Tmid.T @ (tau2[:3, 3] - tau1[:3, 3])
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = zeta
    return g


@cond_jit
def se3_transformation_triad2midstep(g: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from canonical definition to mid-step triad definition."""
    midg = np.copy(g)
    midg[:3, 3] = np.transpose(sqrt_rot(g[:3, :3])) @ g[:3, 3]
    return midg


@cond_jit
def se3_transformation_midstep2triad(midg: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from mid-step triad definition to canonical definition."""
    g = np.copy(midg)
    g[:3, 3] = sqrt_rot(midg[:3, :3]) @ midg[:3, 3]
    return g


##########################################################################################################
##########################################################################################################
##########################################################################################################


@cond_jit
def se3_algebra2group_lintrans(
    groundstate_algebra: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    Trans = np.eye(6)
    Omega_0 = groundstate_algebra[:3]
    zeta_0 = groundstate_algebra[3:]

    Trans[:3, :3] = splittransform_algebra2group(Omega_0)
    if translation_as_midstep:
        sqrtS_transp = euler2rotmat(-0.5 * Omega_0)
        zeta_0_hat_transp = hat_map(-zeta_0)
        H_half = splittransform_algebra2group(0.5 * Omega_0)
        Trans[3:, :3] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H_half
        Trans[3:, 3:] = sqrtS_transp
    else:
        Trans[3:, 3:] = euler2rotmat(-Omega_0)
    return Trans


@cond_jit
def se3_group2algebra_lintrans(
    groundstate_group: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    Trans = np.eye(6)
    Phi_0 = groundstate_group[:3]
    s = groundstate_group[3:]

    H_inv = splittransform_group2algebra(Phi_0)
    Trans[:3, :3] = H_inv
    if translation_as_midstep:
        sqrtS = euler2rotmat(0.5 * Phi_0)
        zeta_0 = sqrtS.T @ s
        zeta_0_hat_transp = hat_map(-zeta_0)
        H_half = splittransform_algebra2group(0.5 * Phi_0)
        Trans[3:, :3] = -0.5 * zeta_0_hat_transp @ H_half @ H_inv
        Trans[3:, 3:] = sqrtS
    else:
        Trans[3:, 3:] = euler2rotmat(Phi_0)
    return Trans


##########################################################################################################
##########################################################################################################
##########################################################################################################


@cond_jit
def se3_algebra2group_stiffmat(
    groundstate_algebra: np.ndarray,
    stiff_algebra: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    """Converts stiffness matrix from algebra-level (vector) splitting between static and dynamic component to group-level (matrix) splitting. Optionally, the transformations from midstep triad definition to triad definition of the translational component may also be included."""
    HX = se3_algebra2group_lintrans(
        groundstate_algebra, translation_as_midstep=translation_as_midstep
    )
    HX_inv = np.linalg.inv(HX)
    stiff_group = HX_inv.T @ stiff_algebra @ HX_inv
    return stiff_group


@cond_jit
def se3_group2algebra_stiffmat(
    groundstate_group: np.ndarray,
    stiff_group: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    """Converts stiffness matrix from group-level (matrix) splitting between static and dynamic component to algebra-level (vector) splitting. Optionally, the transformations from midstep triad definition to triad definition of the translational component may also be included. I.e. the final
    definition will assume a midstep triad definition of the translational component.
    """
    HX_inv = se3_group2algebra_lintrans(
        groundstate_group, translation_as_midstep=translation_as_midstep
    )
    HX = np.linalg.inv(HX_inv)
    stiff_algebra = HX.T @ stiff_group @ HX
    return stiff_algebra
