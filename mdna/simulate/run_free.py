import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import time 
import numpy as np
import scipy as sp
import argparse

try:
    from numba import jit
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
except ModuleNotFoundError:
    print('ModuleNotFoundError: numba')

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

from .GenConfs.straight import gen_straight
from .GenConfs.load_seq import load_seq

from .Evals.tangent_correlation import TangentCorr


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate PolyMC input files")
    parser.add_argument('-in',    '--basename', type=str, required=True) 
    parser.add_argument('-o',     '--outbasename', type=str, required=True) 
    parser.add_argument('-s',     '--steps', type=int,required=True)
    parser.add_argument('-f',     '--force', type=float,default=0)
    parser.add_argument('-nc',     '--ncoup', type=int, default=2)
    parser.add_argument('-ev',     '--evdist', type=float, default=0.)
    parser.add_argument('-equi',   '--equilibration_steps', type=int, default=0)
    parser.add_argument('-nprint', '--print_every', type=int, default=10000)
    parser.add_argument('-ndump',  '--dump_every', type=int, default=0)
    parser.add_argument('-lbn',    '--lbn', type=int, default=0)
    parser.add_argument('-lbmmax', '--lbmmax', type=int, default=50)
    # parser.add_argument('-trace',  '--trace', action='store_true')
    args = parser.parse_args()
    
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    
    #############################
    # load files
    basename = args.basename
    basename_params = basename + '_params'
    stifffn  = basename_params + '_stiff.npz'
    gsfn     = basename_params + '_gs.npy'
    seq      = load_seq(basename)
    
    # stiffmat and groundstate
    stiff = sp.sparse.load_npz(stifffn)
    # stiff = sp.sparse.lil_matrix(stiff)
    gs    = np.load(gsfn)

    nbp = 1001
    nbps = nbp - 1
    seq = seq[:nbp]
    stiff = stiff[:6*nbps,:6*nbps]
    gs    = gs[:6*nbps]
    
    # intial configuration
    print('####################################')
    print('Generating Configuration...')
    conf = gen_straight(gs)
    nbp  = len(conf)
    nbps = nbp-1
    closed = False
    
    # intiate chain
    print('####################################')
    print('Initiating Chain...')
    chain = Chain(conf, keep_backup=True, closed=closed)
    
    # initiate BPStep
    print('####################################')
    print('Initiating Elastic Energy...')
    ncoup = args.ncoup
    
    bps = RBP(chain,seq,gs,stiff,ncoup,closed=closed,static_group=True)

    # # set angle tracing
    # bps.trace_angle_last_triad()
    # bps.trace_angle_first_triad()

    # set stretching force
    force = args.force
    force_dir = np.array([0,0,1])
    beta_force = force_dir * force / 4.114
    bps.set_stretching_force(beta_force)
    
    # set excluded volume
    evdist = args.evdist
    maxdist = 0.46
    check_crossings=True
    if evdist > 0:
        print('####################################')
        print('Initiating Excluded Volume...')
        EV = EVBeads(chain,ev_distance=evdist,max_distance=maxdist,check_crossings=check_crossings)
    else:
        EV = None
    
    # set repulsion plane 
    constraints = []
    if EV is not None:
        repplane = RepulsionPlane(chain,np.array([0,0,1]))
        constraints.append(repplane)
        
    ################################################
    # set monte carlo moves
    print('####################################')
    print('Initiating MC moves...')
    
    moves = list()
    max_cluster = np.min([nbp//2,400])
    
    moves.append(DoublePivot(chain,bps,2,nbp//2,exvol=EV,constraints=constraints))
    moves.append(Pivot(chain, bps,exvol=EV,constraints=constraints,rotate_end=True,preserve_termini=False))
    moves.append(Crankshaft(chain, bps, 2, max_cluster,exvol=EV))
    moves.append(Crankshaft(chain, bps, 2, 20,exvol=EV))
    moves.append(Crankshaft(chain, bps, 2, 20,exvol=EV))
    moves.append(Crankshaft(chain, bps, 2, 20,exvol=EV))
    moves.append(Crankshaft(chain, bps, 2, 20,exvol=EV))
    moves.append(ClusterTrans(chain, bps, 2, max_cluster,exvol=EV))
    st = SingleTriad(chain,bps,exvol=EV,excluded_triad_ids=[0,-1])
    moves += [st]*5

    
    ################################################
    # set simulation specs
    print_every = args.print_every
    equi = args.print_every
    steps = args.steps
    dump_every = args.dump_every
    if print_every == 0:
        print_every = steps*2+equi*2

    
    ################################################
    # Equilibrate
    equi = args.equilibration_steps
    
    if equi > 0:
        print('####################################')
        print('Equilibrate...')
        t1 = time.time()
        for step in range(1,equi+1):
            for move in moves:
                move.mc()
            if step % print_every == 0:
                print('########################')
                print(f"Equi step {step}: ")
                t2 = time.time()
                print(f'dt = %.3f'%(t2-t1))
                t1 = t2
                
                # longest move name
                maxlen = np.max([len(move.name) for move in moves])
                for move in set(moves):
                    print(f"{move.name}:{' '*(maxlen-len(move.name))} %.3f"%(move.acceptance_rate()))

    ################################################
    # Main Run
    
    lbn = args.lbn 
    tancor = TangentCorr(args.lbmmax, disc_len=0.34)
    lbfn = args.outbasename + '_lb'
    
    outdir =  os.path.dirname(args.outbasename)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print('####################################')
    print('Main Run...')
    t1 = time.time()
    for step in range(steps):
        for move in moves:
            move.mc()
            
        if step % print_every == 0 and step != 0:
            print('########################')
            print(f"Step {step}: ")
            t2 = time.time()
            print(f'dt = %.3f'%(t2-t1))
            t1 = t2
            # print('Elastic energy = %.3f'%bps.get_total_energy())
            
            # longest move name
            maxlen = np.max([len(move.name) for move in moves])
            for move in set(moves):
                print(f"{move.name}:{' '*(maxlen-len(move.name))} %.3f"%(move.acceptance_rate()))  
                
            # dump to lb file
            lbdata = tancor.lb
            iter = args.lbmmax // 5
            if iter < 1:
                iter = 1
            print(lbdata[:,::iter])
            print(f'<disc_len> = {tancor.disc_len}')
            np.save(lbfn,lbdata)
                 
        if lbn > 0 and step % lbn == 0:
            tancor.add_tans(chain.triads[:,:,2],normalized=True)
            
    
    
    # dump to lb file
    lbdata = tancor.lb
    iter = args.lbmmax // 5
    if iter < 1:
        iter = 1
    print(lbdata[:,::iter])
    print(f'<disc_len> = {tancor.disc_len}')
    np.save(lbfn,lbdata) 
    

          
