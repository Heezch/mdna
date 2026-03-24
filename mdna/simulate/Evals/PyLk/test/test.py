import time

import iopolymc
import numpy as np

import pylk

# compile
R = np.zeros([100, 3])
R[:, 0] = 1
wm = pylk.writhemap(R, implementation="numba")

statefn = "test.state"

state = iopolymc.load_state(statefn)
configs = state["pos"]

print("#####################")
print("python implementation")
N = 2
print(f"running {N} calculations")
t1 = time.time()
for i in range(N):
    wm = pylk.writhemap(configs[-1], implementation="python")
t2 = time.time()
print(f"Wr = {np.sum(wm)}")
print(f"elapsed time: {(t2-t1)/N}")


print("#####################")
print("numba implementation")
N = 100
print(f"running {N} calculations")
t1 = time.time()
for i in range(N):
    wm = pylk.writhemap(configs[-1], implementation="numba")
t2 = time.time()
print(f"Wr = {np.sum(wm)}")
print(f"elapsed time: {(t2-t1)/N}")

print("#####################")
print("cython implementation")
N = 100
print(f"running {N} calculations")
t1 = time.time()
for i in range(N):
    wm = pylk.writhemap(configs[-1], implementation="cython")
t2 = time.time()
print(f"Wr = {np.sum(wm)}")
print(f"elapsed time: {(t2-t1)/N}")
