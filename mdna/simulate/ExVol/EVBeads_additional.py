import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from ..._jit import cond_jit
from .ExVol import ExVol
from ..chain import Chain

    
#########################################################################################
########## Double Move ##################################################################
#########################################################################################

@cond_jit
def doubleMove(id1: int, id2: int, bp_pos: np.ndarray, bp_pos_backup: np.ndarray, EV_dist: float) -> float:
    p1  = bp_pos_backup[id1]
    p1p = bp_pos[id1]
    p2  = bp_pos_backup[id2]
    p2p = bp_pos[id2]
    
    dist_primes = np.linalg.norm(p2p-p1p)
    if (dist_primes<EV_dist):
        return dist_primes

    Delta_p = p1-p2
    Delta_v = p1p-p2p-Delta_p

    # check if both were translated by the same vector
    dist = np.dot(Delta_v,Delta_v)
    if (dist < 1e-10):
        return dist_primes
    
    lamb = -np.dot(Delta_p,Delta_v)/dist
    # check if the closest approach is outside the moved interval
    if (lamb < 0):
        return np.linalg.norm(Delta_p)
    
    if (lamb > 1):
        return dist_primes
    
    dvec = Delta_p+lamb*Delta_v
    dist = np.linalg.norm(dvec)
    return dist


#########################################################################################
########## Single Move ##################################################################
#########################################################################################

@cond_jit
def singleMove(id1: int, id2: int, bp_pos: np.ndarray, bp_pos_backup: np.ndarray)-> float:
    p1  = bp_pos_backup[id1]
    p1p = bp_pos[id1]
    p2  = bp_pos[id2]
    
    v = p1p-p1
    nv = np.linalg.norm(v)
    w1 = p2-p1
    if (nv<1e-12):
        return np.linalg.norm(p2-p1)
    v = v/nv
    w2 = p1p-p2
    n_com = np.dot(v,w1)
    if (n_com < 0):
        return np.linalg.norm(p2-p1)
    if (np.dot(v,w2) < 0):
        return np.linalg.norm(p2-p1p)
    return np.linalg.norm( w1-n_com*v )

#########################################################################################
########## Numba Single Move Interval Check #############################################
#########################################################################################


@cond_jit
def singleMove_intervals(
    A1: int, A2: int, B1: int, B2: int, closed: bool, 
    bp_pos: np.ndarray, bp_pos_backup: np.ndarray,
    EV_dist: float, num_EV: int, EV_beads: np.ndarray,
    curr_size_EV_bead: float, 
    neighbour_skip_boundary_plus_one: int,
    neighbour_skip_plus_one: int
    ) -> bool:
    """
        Checks for projected overlap between EV beads in the two intervals limited by
        A1,A2 and B1,B2 respectively. This method assumes that the beads in the B
        interval have not moved, such that only the linear displacement of the beads
        in the A interval have to be considered. The checking is done by the
        singleMove method.
    """
    
    # #ifdef DEBUG_EXVOL
    # if (!(A2<B1 || B2 < A1)) {
    #     std::cout << "Check Intervals " << A1 << " " << A2 << " " << B1 << " " << B2 << std::endl;
    #     throw std::invalid_argument("ExVol::check_interval_singleMove(): Invalid Intervals! The invervals are not allowed to cross the periodic boundary!");
    # }
    # #endif

    if (closed):
                
    ##############################################################
    # PERIODIC BOUNDARY CONDITION

        # Left to right order: A - B
        if (A2<B1):
            a1 = max([A1,B2-num_EV+neighbour_skip_boundary_plus_one])
            a2 = min([A2,B1-neighbour_skip_plus_one])

            # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
            a = int(A1)
            while (a<=A2):
                if ( a1<=a and a<=a2):
                    a=a2+1
                    continue
                
                b  = max ([B1,a+neighbour_skip_plus_one])
                b2 = min ([B2,a+num_EV-neighbour_skip_boundary_plus_one])
                while (b<=b2):
                    dist = singleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
                a += 1
        
        # Left to right order: B - A
        else:
            a1 = max([A1,B2+neighbour_skip_plus_one])
            a2 = min([A2,B1+num_EV-neighbour_skip_boundary_plus_one])
            # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
            a = int(A1)
            while (a<=A2):
                if ( a1<=a and a<=a2):
                    a=a2+1
                    continue
                
                b  = max ([B1,a-num_EV+neighbour_skip_boundary_plus_one])
                b2 = min ([B2,a-neighbour_skip_plus_one])
                while (b<=b2):
                    dist = singleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
                a+=1
    else:
    ##############################################################
    # NON-PERIODIC BOUNDARY CONDITION
    
        # Left to right order: A - B
        if (A2<B1):
            # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
            # In this case this region is only at the right boundary of the interval A.
            a1 = int(A1)
            a2 = min([A2,B1-neighbour_skip_plus_one])
            
            # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
            for a in range(a2+1,A2+1):
                b = max([B1,a+neighbour_skip_plus_one])
                while (b<=B2):
                    dist = singleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        # print(f'violation: {EV_beads[a]}-{EV_beads[b]}')
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
        
        # Left to right order: B - A
        else:
            # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
            # In this case this region is only at the left boundary of the interval A.
            a1 = max([A1,B2+neighbour_skip_plus_one])
            a2 = int(A2)
            
            # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
            for a in range(A1,a1):      
                b  = int(B1)
                b2 = min([B2,a-neighbour_skip_plus_one])
                while (b<=b2):
                    dist = singleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        # print(f'violation: {EV_beads[a]}-{EV_beads[b]}')
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
                    
    ##############################################################
    # Finally check all pairs outside the range of potential neighbour skips. I.e. the bulk of the intervals.
    for a in range(a1,a2+1):
        b = int(B1)
        while (b<=B2):
            dist = singleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup)
            if (dist < EV_dist):
                # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                # print(f'violation: {EV_beads[a]}-{EV_beads[b]}')
                return False
            #b+=1
            b += int((dist-EV_dist)//curr_size_EV_bead+1)
    return True


@cond_jit
def doubleMove_intervals(
    A1: int, A2: int, B1: int, B2: int, closed: bool, 
    bp_pos: np.ndarray, bp_pos_backup: np.ndarray,
    EV_dist: float, num_EV: int, EV_beads: np.ndarray,
    curr_size_EV_bead: float, 
    neighbour_skip_boundary_plus_one: int,
    neighbour_skip_plus_one: int
    ) -> bool:
    """
        Checks for projected overlap between EV beads in the two intervals limited by
        A1,A2 and B1,B2 respectively. This method assumes that the beads in the B
        interval have not moved, such that only the linear displacement of the beads
        in the A interval have to be considered. The checking is done by the
        singleMove method.
    """
    
    # #ifdef DEBUG_EXVOL
    # if (!(A2<B1 || B2 < A1)) {
    #     std::cout << "Check Intervals " << A1 << " " << A2 << " " << B1 << " " << B2 << std::endl;
    #     throw std::invalid_argument("ExVol::check_interval_singleMove(): Invalid Intervals! The invervals are not allowed to cross the periodic boundary!");
    # }
    # #endif

    if (closed):
                
    ##############################################################
    # PERIODIC BOUNDARY CONDITION

        # Left to right order: A - B
        if (A2<B1):
            a1 = max([A1,B2-num_EV+neighbour_skip_boundary_plus_one])
            a2 = min([A2,B1-neighbour_skip_plus_one])

            # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
            a = int(A1)
            while (a<=A2):
                if ( a1<=a and a<=a2):
                    a=a2+1
                    continue
                
                b  = max ([B1,a+neighbour_skip_plus_one])
                b2 = min ([B2,a+num_EV-neighbour_skip_boundary_plus_one])
                while (b<=b2):
                    dist = doubleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup,EV_dist)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
                a += 1
        
        # Left to right order: B - A
        else:
            a1 = max([A1,B2+neighbour_skip_plus_one])
            a2 = min([A2,B1+num_EV-neighbour_skip_boundary_plus_one])
            # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
            a = int(A1)
            while (a<=A2):
                if ( a1<=a and a<=a2):
                    a=a2+1
                    continue
                
                b  = max ([B1,a-num_EV+neighbour_skip_boundary_plus_one])
                b2 = min ([B2,a-neighbour_skip_plus_one])
                while (b<=b2):
                    dist = doubleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup,EV_dist)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
                a+=1
    else:
    ##############################################################
    # NON-PERIODIC BOUNDARY CONDITION
    
        # Left to right order: A - B
        if (A2<B1):
            # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
            # In this case this region is only at the right boundary of the interval A.
            a1 = int(A1)
            a2 = min([A2,B1-neighbour_skip_plus_one])
            
            # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
            for a in range(a2+1,A2+1):
                b = max([B1,a+neighbour_skip_plus_one])
                while (b<=B2):
                    dist = doubleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup,EV_dist)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
        
        # Left to right order: B - A
        else:
            # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
            # In this case this region is only at the left boundary of the interval A.
            a1 = max([A1,B2+neighbour_skip_plus_one])
            a2 = int(A2)
            
            # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
            for a in range(A1,a1):      
                b  = int(B1)
                b2 = min([B2,a-neighbour_skip_plus_one])
                while (b<=b2):
                    dist = doubleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup,EV_dist)
                    if (dist < EV_dist):
                        # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                        return False
                    #b+=1
                    b += int((dist-EV_dist)//curr_size_EV_bead+1)
                    
    ##############################################################
    # Finally check all pairs outside the range of potential neighbour skips. I.e. the bulk of the intervals.
    for a in range(a1,a2+1):
        b = int(B1)
        while (b<=B2):
            dist = doubleMove(EV_beads[a],EV_beads[b],bp_pos,bp_pos_backup,EV_dist)
            if (dist < EV_dist):
                # debug_plot(bp_pos,bp_pos_backup,EV_beads,a,b)
                return False
            #b+=1
            b += int((dist-EV_dist)//curr_size_EV_bead+1)
    return True


#########################################################################################
########## Current Max Distance #########################################################
#########################################################################################

@cond_jit
def largest_dist(    
    evpos: np.ndarray, 
    closed: bool,
    ) -> np.ndarray:
    largest = np.max(norms(evpos[1:]-evpos[:-1]))
    if closed:
        dist = np.linalg.norm(evpos[0]-evpos[-1])
        if dist > largest:
            largest = dist
    return largest

@cond_jit
def norms(vecs: np.ndarray) -> np.ndarray:
    dists = np.empty(len(vecs))
    for i in range(len(vecs)):
        dists[i] = np.sqrt(vecs[i,0]**2+vecs[i,1]**2+vecs[i,2]**2)
    return dists

#########################################################################################
########## Plotting for Debug ###########################################################
#########################################################################################

import matplotlib.pyplot as plt
def debug_plot(pos, backup, EV_beads, a, b):
    debug = False
    if not debug: return
    
    print('####################')
    print(f'overlap {a} - {b}')
    ida = EV_beads[a]
    idb = EV_beads[b]
    print(f'bp ids: {ida} - {idb}')
    print(pos[ida])
    print(pos[idb])
    print(np.linalg.norm(pos[ida]-pos[idb]))
        
        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax = plt.gca(projection='3d')
    
    ida = EV_beads[a]
    idb = EV_beads[b]
    
    segpos = pos[ida:idb+1]
    rempos = np.array([pos[i] for i in range(len(pos)) if i not in np.arange(ida,idb+1)])
    
    segpos_backup = backup[ida:idb+1]
    rempos_backup = np.array([backup[i] for i in range(len(pos)) if i not in np.arange(ida,idb+1)])
    
    ax.plot(*backup.T,color='black',label='Control Points',lw=1)
    ax.scatter(*rempos_backup.T,color='green',s=16,edgecolor='black',alpha=0.2,zorder=2)
    ax.scatter(*segpos_backup.T,color='red',s=16,edgecolor='black',zorder=3,alpha=0.5)
    evpos_backup = np.array([backup[id] for id in EV_beads])
    ax.scatter(*evpos_backup.T,color='blue',s=500,edgecolor='black',alpha=0.2)
    
    ax.plot(*pos.T,color='black',label='Control Points',lw=1)
    ax.scatter(*rempos.T,color='green',s=16,edgecolor='black',alpha=0.4,zorder=2)
    ax.scatter(*segpos.T,color='red',s=16,edgecolor='black',zorder=3)
    
    evpos = np.array([pos[id] for id in EV_beads])
    ax.scatter(*evpos.T,color='blue',s=500,edgecolor='black',alpha=0.4)
    
    
    com = np.mean(pos,axis=0)
    maxrge = np.max([np.max(pos[:,i])-np.min(pos[:,i]) for i in range(3)])
    margin = maxrge*0.01
    halfrge = maxrge*0.5+margin
    rges = []
    for i in range(3):
        rges.append([com[i]-halfrge,com[i]+halfrge])
    for x in range(2):
        for y in range(2):
            for z in range(2):
                ax.scatter([rges[0][x]],[rges[1][y]],[rges[2][z]],alpha=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()