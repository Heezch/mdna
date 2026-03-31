import numpy as np
import warnings
from .linkingnumber_python import _eval_lk_python

LK_METHOD = 0
try:
    from .linkingnumber_cython import _eval_lk_cython
    LK_METHOD = 1
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        "Cython version of linkingnumber (PyLk) not compiled. Defaulting to numba implementation. Consider compiling the cython version."
    )
    try:
        from .linkingnumber_numba import _eval_lk_numba
        LK_METHOD = 2
    except (ImportError, ModuleNotFoundError):
        warnings.warn(
            "PyLK: Numba not installed. Defaulting to python implementation. Consider installing numba or compiling cython implementation."
        )
         
def linkingnumber(chain1: np.ndarray, chain2: np.ndarray) -> np.ndarray:
    if LK_METHOD == 1:
        print('using cython')
        return _eval_lk_cython(chain1,chain2)
    elif LK_METHOD == 2:
        print('using numba')
        return _eval_lk_numba(chain1,chain2)
    return _eval_lk_python(chain1,chain2)

