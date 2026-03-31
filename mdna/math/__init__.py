"""
SO3
=====

A package for various rotation operations

"""

from .Cayley import cayley2rotmat, rotmat2cayley, se3_cayley2rotmat, se3_rotmat2cayley
from .conversions import (
    cayley2euler,
    cayley2euler_factor,
    cayley2euler_linearexpansion,
    euler2cayley,
    euler2cayley_factor,
    euler2cayley_linearexpansion,
    splittransform_algebra2group,
    splittransform_group2algebra,
)
from .Euler import (
    euler2rotmat,
    midstep,
    rotmat2euler,
    se3_euler2rotmat,
    se3_rotmat2euler,
    sqrt_rot,
)
from .extend_euler import extend_euler
from .generators import generator1, generator2, generator3, hat_map, vec_map
from .matrices import dots
from .._jit import cond_jit
from .SE3 import (
    se3_algebra2group_lintrans,
    se3_algebra2group_stiffmat,
    se3_group2algebra_lintrans,
    se3_group2algebra_stiffmat,
    se3_inverse,
    se3_midstep2triad,
    se3_transformation_midstep2triad,
    se3_transformation_triad2midstep,
    se3_triad2midstep,
    se3_triads2euler,
    se3_triads2rotmat,
    se3_triads2rotmat_midsteptrans,
    se3_triadxrotmat_midsteptrans,
)

# legacy method
from .SO3Methods import phi2rotx, phi2roty, phi2rotz
