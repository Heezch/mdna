"""
Conditional Numba JIT decorators.

Provides decorators that apply Numba JIT compilation when available,
falling back to plain Python if Numba is not installed.  Inlined from
the former ``pyConDec`` submodule (github.com/eskoruppa/pyConDec).
"""


def cond_jit(function):
    """Apply ``@numba.jit(nopython=True)`` if Numba is installed, otherwise return *function* unchanged."""
    try:
        from numba import jit
        return jit(nopython=True)(function)
    except ModuleNotFoundError:
        print(f"Warning: {function.__name__}: numba not installed. For speedup please install numba: pip install numba")
        return function


def cond_jitclass(origclass):
    """Apply ``@numba.experimental.jitclass()`` if Numba is installed, otherwise return *origclass* unchanged."""
    try:
        from numba.experimental import jitclass
        return jitclass()(origclass)
    except ModuleNotFoundError:
        print(f"Warning: {origclass.__name__}: numba not installed. For speedup please install numba: pip install numba")
        return origclass
