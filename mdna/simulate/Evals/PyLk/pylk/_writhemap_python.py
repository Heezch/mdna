import math

import numpy as np


def wmp_writhemap_klenin1a(pos):
    N = len(pos)
    WM = np.zeros([N, N])
    fac4pi = 1.0 / (4 * np.pi)

    """
        Calculate Gauss integral segment contributions except those involving the last segment
    """

    for i in range(N - 3):
        ip1 = i + 1
        r12 = pos[ip1] - pos[i]
        for j in range(i + 2, N - 1):
            jp1 = j + 1

            r13 = pos[j] - pos[i]
            r14 = pos[jp1] - pos[i]
            r23 = pos[j] - pos[ip1]
            r24 = pos[jp1] - pos[ip1]
            r34 = pos[jp1] - pos[j]

            n1 = np.cross(r13, r14)
            vecnorm = np.linalg.norm(n1)
            if vecnorm > 1e-10:
                n1 = n1 / vecnorm

            n2 = np.cross(r14, r24)
            vecnorm = np.linalg.norm(n2)
            if vecnorm > 1e-10:
                n2 = n2 / vecnorm

            n3 = np.cross(r24, r23)
            vecnorm = np.linalg.norm(n3)
            if vecnorm > 1e-10:
                n3 = n3 / vecnorm

            n4 = np.cross(r23, r13)
            vecnorm = np.linalg.norm(n4)
            if vecnorm > 1e-10:
                n4 = n4 / vecnorm

            Omega = np.arcsin(np.dot(n1, n2))
            Omega += np.arcsin(np.dot(n2, n3))
            Omega += np.arcsin(np.dot(n3, n4))
            Omega += np.arcsin(np.dot(n4, n1))

            Omega = Omega * fac4pi * np.sign(np.dot(np.cross(r34, r12), r13))

            if not math.isnan(Omega):
                WM[i, j] = Omega
                WM[j, i] = Omega

    """
        Calculate Gauss integral segment contributions involving the last segment
    """

    j = N - 1
    lastpos = pos[0]
    for i in range(1, N - 2):
        ip1 = i + 1

        r12 = pos[ip1] - pos[i]
        r13 = pos[j] - pos[i]
        r14 = lastpos - pos[i]
        r23 = pos[j] - pos[ip1]
        r24 = lastpos - pos[ip1]
        r34 = lastpos - pos[j]

        n1 = np.cross(r13, r14)
        vecnorm = np.linalg.norm(n1)
        if vecnorm > 1e-10:
            n1 = n1 / vecnorm

        n2 = np.cross(r14, r24)
        vecnorm = np.linalg.norm(n2)
        if vecnorm > 1e-10:
            n2 = n2 / vecnorm

        n3 = np.cross(r24, r23)
        vecnorm = np.linalg.norm(n3)
        if vecnorm > 1e-10:
            n3 = n3 / vecnorm

        n4 = np.cross(r23, r13)
        vecnorm = np.linalg.norm(n4)
        if vecnorm > 1e-10:
            n4 = n4 / vecnorm

        Omega = np.arcsin(np.dot(n1, n2))
        Omega += np.arcsin(np.dot(n2, n3))
        Omega += np.arcsin(np.dot(n3, n4))
        Omega += np.arcsin(np.dot(n4, n1))

        Omega = Omega * fac4pi * np.sign(np.dot(np.cross(r34, r12), r13))

        if not math.isnan(Omega):
            WM[i, j] = Omega
            WM[j, i] = Omega
    return WM
