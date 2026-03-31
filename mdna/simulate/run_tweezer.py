import os, sys
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

from .ExVol.EVBeads import EVBeads
from .Constraints.RepulsionPlane import RepulsionPlane
from .Dumps.xyz import write_xyz
from .BPStep.RBPStiff import GenStiffness

from .GenConfs.straight import gen_straight
from .GenConfs.load_seq import load_seq

from .Evals.PyLk.pylk import writhemap
from . import Evals as evals

if __name__ == "__main__":
    
    ########################################################################################
    ########################### SETUP ###################################################### 
    ########################################################################################
    # writhe configuration extension
    num_ext = 10
    
    parser = argparse.ArgumentParser(description="Generate PolyMC input files")
    parser.add_argument('-in',    '--basename', type=str, required=True) 
    parser.add_argument('-o',     '--outbasename', type=str, required=True) 
    parser.add_argument('-s',     '--steps', type=int,required=True)
    parser.add_argument('-f',     '--force', type=float,default=0)
    parser.add_argument('-nc',     '--ncoup', type=int, default=2)
    parser.add_argument('-ev',     '--evdist', type=float, default=0.)
    parser.add_argument('-equi',   '--equilibration_steps', type=int, default=0)
    parser.add_argument('-nprint', '--print_every', type=int, default=10000)
    parser.add_argument('-fomtn',  '--fomtn', type=int, default=0)
    parser.add_argument('-fomtwr', '--include_writhe', action='store_true')
    parser.add_argument('-xyzn',   '--xyzn', type=int, default=0)
    parser.add_argument('-dlk',    '--linking_number', type=float, default=0)
    parser.add_argument('-print_link', '--print_link', action='store_true')
    parser.add_argument('-fl',     '--fix_link', action='store_true')
    # parser.add_argument('-trace',  '--trace', action='store_true')
    args = parser.parse_args()
    
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    
    #############################
    # load files
    basename = args.basename
    basename_params = basename + '_params'
    stifffn  = basename_params + '_stiff.npz'
    gsfn     = basename_params + '_gs.npy'
    
    # stiffmat and groundstate
    stiff = sp.sparse.load_npz(stifffn)
    # stiff = sp.sparse.lil_matrix(stiff)
    gs    = np.load(gsfn)
              
    if os.path.isfile(basename):
        seq = load_seq(basename)
    else:
        seq = 'X'*(len(gs)+1)
        
    # ####################
    # print('TESTING! REMOVE THIS!')
    # # REMOVE THIS!
    # # gs[:,:2] = 0
    # # gs[:,3:5] = 0
    # N = 100
    # gs = gs[:N]
    # stiff = stiff[:N*6,:N*6]
    # seq = seq[:N+1]
    # ####################
        
    # intial configuration
    print('####################################')
    print('Generating Configuration...')
    conf = gen_straight(gs,dlk=args.linking_number)
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
        
    # set angle tracing
    if not args.fix_link:
        bps.trace_angle_last_triad()
        bps.trace_angle_first_triad()

    # set stretching force
    force = args.force
    force_dir = np.array([0,0,1])
    beta_force = force_dir * force / 4.114
    bps.set_stretching_force(beta_force)
        
    # set excluded volume
    evdist = args.evdist
    check_crossings=True
    if evdist > 0:
        print('####################################')
        print('Initiating Excluded Volume...')
        avgdist = np.mean(np.linalg.norm(gs[:,3:],axis=1))
        maxdist = 0.46
        maxdist = np.min([1.5*avgdist,evdist])
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
    moves.append(Crankshaft(chain, bps, 2, max_cluster,exvol=EV,constraints=constraints))
    moves.append(ClusterTrans(chain, bps, 2, max_cluster,exvol=EV,constraints=constraints))
    st_full = SingleTriad(chain,bps,exvol=EV,excluded_triad_ids=[0,-1],constraints=constraints)
    moves += [st_full]*2
    
    # link changing moves
    if not args.fix_link:
        moves.append(Pivot(chain, bps,exvol=EV,constraints=constraints,rotate_end=True,preserve_termini=True))
        sel_limit = nbp - 20
        if sel_limit < 1: sel_limit = 1     
        moves.append(Pivot(chain, bps,exvol=EV,constraints=constraints,rotate_end=True,preserve_termini=True,selection_limit_id=sel_limit))
        st_trans = SingleTriad(chain,bps,exvol=EV,rotate=False,selected_triad_ids=[id for id in range(nbp-20,nbp) if id > 0],constraints=constraints)
        moves += [st_trans]

    ################################################
    # set simulation specs
    print_every = args.print_every
    equi = args.print_every
    steps = args.steps
    if print_every == 0:
        print_every = steps*2+equi*2


    ################################################
    # set turns to current dlk
    tw = np.sum(bps.current_deforms[:,2]) / (2*np.pi)
    wr = evals.writhe(chain.conf[:,:3,3],closed=False,num_ext=num_ext,ext_dir=force_dir)    
    bps.angle_tracing_last_proposed_angle = tw + wr
    bps.angle_tracing_last_current_angle  = tw + wr

    ###################################################################################
    # Equilibrate
    ###################################################################################
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
                print('first turns = %.3f'%(bps.get_angle_first()/(2*np.pi)))
                print('last turns  = %.3f'%(bps.get_angle_last()/(2*np.pi)))
                t2 = time.time()
                print(f'dt = %.3f'%(t2-t1))
                t1 = t2
                
                # longest move name
                maxlen = np.max([len(move.name) for move in moves])
                for move in set(moves):
                    print(f"{move.name}:{' '*(maxlen-len(move.name))} %.3f"%(move.acceptance_rate()))

    ###################################################################################
    # Init dumps
    ###################################################################################
    
    fomtn = args.fomtn
    xyzn  = args.xyzn
    
    fomt_fn = args.outbasename + '.fomt'
    xyz_fn  = args.outbasename + '.xyz'

    stored_fomt = 100
    fomt_data = []
    if fomtn > 0:
        with open(fomt_fn, 'w') as f:
            f.write('')
    if xyzn > 0:
        with open(xyz_fn, 'w') as f:
            f.write('')     
    
    ###################################################################################
    # Main Loop
    ###################################################################################

    print('####################################')
    print('Main Run...')
    t1 = time.time()
    for step in range(1,steps+1):
        
        ###################################
        # Moves
        ###################################
        for move in moves:
            move.mc()
        
        ###################################
        # Print
        ###################################
        if step % print_every == 0:
            print('########################')
            print(f"Step {step}: ")
            
            if args.print_link:
                tw = np.sum(bps.current_deforms[:,2]) / (2*np.pi)
                wr = evals.writhe(chain.conf[:,:3,3],closed=False,num_ext=num_ext,ext_dir=force_dir)            
                print(f'tw = {tw}')
                print(f'wr = {wr}')
                print(f'lk = {tw+wr}')
            if not args.fix_link:
                print('first turns = %.3f'%(bps.get_angle_first()/(2*np.pi)))
                print('last turns  = %.3f'%(bps.get_angle_last()/(2*np.pi)))
            
            
            t2 = time.time()
            print(f'dt = %.3f'%(t2-t1))
            t1 = t2
            # print('Elastic energy = %.3f'%bps.get_total_energy())
            
            # longest move name
            maxlen = np.max([len(move.name) for move in moves])
            for move in set(moves):
                print(f"{move.name}:{' '*(maxlen-len(move.name))} %.3f"%(move.acceptance_rate()))   

        ###################################
        # Dump FOMT
        ###################################
        if fomtn > 0 and step % fomtn == 0:
            # end rotation
            rot = bps.get_angle_last()/(2*np.pi)
            # z direction
            z = np.dot(chain.conf[-1,:3,3]-chain.conf[0,:3,3] , force_dir)
            # twist
            tw = np.sum(bps.current_deforms[:,2]) / (2*np.pi)
            
            # store as string
            dumpstr = '%.4f %.4f %.4f'%(z,rot,tw)
            if args.include_writhe:
                wr = evals.writhe(chain.conf[:,:3,3],closed=False,num_ext=num_ext,ext_dir=force_dir)
                dumpstr = dumpstr + ' %.4f'%wr
            
            fomt_data.append(dumpstr)
            # print to file
            if len(fomt_data) == stored_fomt:
                with open(fomt_fn, 'a') as f:
                    for d in fomt_data:
                        f.write(d+'\n')
                fomt_data = []
        
        ###################################
        # Dump XYZ
        ###################################
        if xyzn > 0 and step % xyzn == 0:
            confs = [chain.conf[:,:3,3]]
            types = [x for x in seq]
            data = {"pos": confs, "types": types}
            write_xyz(xyz_fn, data,append=True)
               
    # write remaining dumps  
    with open(fomt_fn, 'a') as f:
        for d in fomt_data:
            f.write(d+'\n')