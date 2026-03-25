import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import time 
import numpy as np

from .BPStep.BPStep import BPStep
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


if __name__ == "__main__":
    from .BPStep.BPS_RBP import RBP

    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    nbp = 1000
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
    bps = RBP(ch, seq, specs, closed=closed, static_group=True)
    
    bps.trace_angle_last_triad()
    bps.trace_angle_first_triad()
    
    force = 0.5
    
    beta_force = np.array([0,0,1])* force / 4.114
    # bps.set_stretching_force(beta_force)
    
    
    ev_distance = 3
    check_crossings=True
    EV = EVBeads(ch,ev_distance=ev_distance,max_distance=0.46,check_crossings=check_crossings)
    EV = None
    
    constraints = []
    # repplane = RepulsionPlane(ch,np.array([0,0,1]))
    # constraints.append(repplane)


    moves = list()
    # moves.append(Crankshaft(ch, bps, 2, 50,exvol=EV))
    # moves.append(Pivot(ch, bps,exvol=EV,constraints=constraints,rotate_end=True,preserve_termini=False))
    
    moves.append(DoublePivot(ch,bps,2,nbp//2,exvol=EV,constraints=constraints))
    moves.append(Pivot(ch, bps,exvol=EV,constraints=constraints,rotate_end=True,preserve_termini=True))


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
