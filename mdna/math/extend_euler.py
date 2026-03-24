#!/bin/env python3
import numpy as np

def extend_euler(eulers: np.ndarray,copy: bool = True):
    if copy:
        exteulers = np.copy(eulers)
    else:
        exteulers = eulers   
    for s in range(1,len(exteulers)):
        Om1 = exteulers[s-1]
        Om2 = exteulers[s]
        nOm2 = np.linalg.norm(Om2)
        uOm2 = Om2/nOm2
        shift = np.round (1./(2*np.pi) * (np.dot(Om1,uOm2) - nOm2))
        if np.abs(shift) > 0:
            nOm2p = nOm2 + shift * 2*np.pi
            Om2p = nOm2p * uOm2
            exteulers[s] = Om2p
    return exteulers