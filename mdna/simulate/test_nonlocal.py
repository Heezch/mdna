import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import time 
import numpy as np

from .BPStep.BPStep import BPStep
from .BPStep.BPS_RBP import RBP
from .BPStep.BPS_LRBP import LRBP
from .chain import Chain
from .ExVol.ExVol import ExVol
from .MCStep.clustertranslation import ClusterTrans
from .MCStep.crankshaft import Crankshaft
from .MCStep.midstepmove import MidstepMove
from .MCStep.pivot import Pivot
from .MCStep.singletriad import SingleTriad
from .MCStep.doublepivot import DoublePivot

from .. import math as so3

from .ExVol.EVBeads import EVBeads
from .Constraints.RepulsionPlane import RepulsionPlane
from .Dumps.xyz import write_xyz
from .BPStep.RBPStiff import GenStiffness



if __name__ == "__main__":
    
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    
    nbp = 4000
    closed = False
    conf = np.zeros((nbp, 4, 4))
    gs = np.array([0, 0, 0.61, 0, 0, 0.34])
    g = so3.se3_euler2rotmat(gs)
    conf[0] = np.eye(4)
    for i in range(1, nbp):
        # g = so3.se3_euler2rotmat(gs + np.random.normal(0, 0.1, 6))
        g = so3.se3_euler2rotmat(gs)
        conf[i] = conf[i - 1] @ g
    
    seq = "".join(["ATCG"[np.random.randint(4)] for i in range(nbp)])
    
    specs = {"method": "MD", "gs_key": "group_gs", "stiff_key": "group_stiff"}

    ch = Chain(conf, keep_backup=True, closed=closed)
    
    ncoup = 4
    genstiff = GenStiffness()
    stiff,gs = genstiff.gen_params(seq,use_group=True)
    
    

    if closed:
        stiff,gs = genstiff.gen_params(seq+seq[:1+ncoup],use_group=True)
    

    N = len(stiff) // 6
    # ncoup = 1
    for i in range(N-1):
        for k in range(1,ncoup+1):
            j = i+k
            if j > N-1:
                continue
            stiff[i*6:(i+1)*6,j*6:(j+1)*6] = 1./((1+j)) * stiff[i*6:(i+1)*6,i*6:(i+1)*6]
    stiff = 0.5*(stiff+stiff.T)
                
    if closed:
        M = 6*nbp
        for i in range(len(stiff)):
            ii = i % M
            for j in range(len(stiff)):
                jj = j % M
                if stiff[i,j] != 0:
                    stiff[ii,jj] = stiff[i,j]
        stiff = stiff[:M,:M]
        gs    = gs[:M]
    
    
    ss = stiff[::6,::6]
    print(ss)
    
    bps = RBP(ch,seq,gs,stiff,ncoup,closed=closed,static_group=True)


    # bps = LRBP(ch, seq, specs, closed=closed, static_group=True)
    
    bps.trace_angle_last_triad()
    bps.trace_angle_first_triad()
    
    force = 0.5
    beta_force = np.array([0,0,1])* force / 4.114
    bps.set_stretching_force(beta_force)
    
    
    ev_distance = 4
    check_crossings=True
    EV = EVBeads(ch,ev_distance=ev_distance,max_distance=0.46,check_crossings=check_crossings)
    # EV = None
    
    constraints = []
    repplane = RepulsionPlane(ch,np.array([0,0,1]))
    constraints.append(repplane)


    moves = list()
    moves.append(Crankshaft(ch, bps, 2, nbp//2,exvol=EV))
    # moves.append(Pivot(ch, bps,exvol=EV,constraints=constraints,rotate_end=True,preserve_termini=False))
    
    moves.append(DoublePivot(ch,bps,2,nbp//2,exvol=EV,constraints=constraints))
    moves.append(Pivot(ch, bps,exvol=EV,constraints=constraints,rotate_end=True,preserve_termini=True))
    moves.append(SingleTriad(ch,bps,exvol=EV,excluded_triad_ids=[0,-1]))
    moves.append(SingleTriad(ch,bps,exvol=EV,excluded_triad_ids=[0,-1]))
    # moves.append(SingleTriad(ch,bps,exvol=EV))


    # moves.append(SingleTriad(ch,bps))
    # moves.append(ClusterTrans(ch,bps,2,25))

    # for i in range(10000):
    #     for move in moves:
    #         move.mc()
    
    Es = []
    confs = []
    confs.append(np.copy(ch.conf[:, :3, 3]))
    t1 = time.time()
    
    angles_first = []
    angles_last  = []
    
    for i in range(1000000):
        for move in moves:
            move.mc()
            angles_first.append(bps.get_angle_first())
            angles_last.append(bps.get_angle_last())

        if i%1000 == 0:
            confs.append(np.copy(ch.conf[:, :3, 3]))

        if i % 1000 == 0:
            print('####################################')
            print(f"step {i}: ")
            print(f'first turns = {bps.get_angle_first()/(2*np.pi)}')
            print(f'last turns  = {bps.get_angle_last()/(2*np.pi)}')
            t2 = time.time()
            print(f'dt = {t2-t1}')
            t1 = t2
            for move in moves:
                print(f"{move.name}: {move.acceptance_rate()}")
            Es.append(bps.get_total_energy())
            
            if not bps.check_deform_consistency():
                sys.exit()
        
        if i % 5000 == 0 and i != 0:
            types = ["C" for i in range(ch.nbp)]
            data = {"pos": confs, "types": types}
            write_xyz("conf.xyz", data)
            
    print(f"<E> / DoFs = {np.mean(Es)/(len(ch.conf)-1)/6}")
    print(f'mean angle first = {np.mean(angles_first)}')
    print(f'mean angle last  = {np.mean(angles_last)}')
    
    types = ["C" for i in range(ch.nbp)]
    data = {"pos": confs, "types": types}
    write_xyz("conf.xyz", data)
