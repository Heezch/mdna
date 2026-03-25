import warnings

import numpy as np

from ._writhemap_numba import wmn_writhemap_klenin1a
from ._writhemap_python import wmp_writhemap_klenin1a

try:
    from ._writhemap_cython import wmc_writhemap_klenin1a

    WM_CYTHON_INCLUDED = True
    WM_DEFAULT_METHOD = "cython"
except (ImportError, ModuleNotFoundError):
    WM_CYTHON_INCLUDED = False
    WM_DEFAULT_METHOD = "numba"
    warnings.warn(
        "Cython version of writhemap (PyLk) not compiled. Defaulting to numba implementation. Consider compiling the cython version."
    )

WM_METHODS = ["klenin1a"]

def writhemap(config, method="klenin1a", implementation=WM_DEFAULT_METHOD):
    if method not in WM_METHODS:
        raise ValueError(f"Method '{method}' not implemented")

    if implementation == "cython" and not WM_CYTHON_INCLUDED:
        raise ModuleNotFoundError("No module named 'pylk._writhemap_cython'")

    if method == "klenin1a":
        if implementation == "cython":
            return np.asarray(wmc_writhemap_klenin1a(config))
        elif implementation == "numba":
            return wmn_writhemap_klenin1a(config)
        elif implementation == "python":
            return wmp_writhemap_klenin1a(config)
        else:
            raise ValueError(
                f"Invalid implementation '{implementation}' for method '{method}'"
            )
