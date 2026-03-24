import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import numpy as np

try:
    from numba import jit
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
except ModuleNotFoundError:
    print('ModuleNotFoundError: numba')

from ..BPStep.BPStep import BPStep
from ..chain import Chain, se3_triads
from ..ExVol.ExVol import ExVol
from ..MCStep.clustertranslation import ClusterTrans
from ..MCStep.crankshaft import Crankshaft
from ..MCStep.midstepmove import MidstepMove
from ..MCStep.pivot import Pivot
from ..MCStep.singletriad import SingleTriad
from ... import math as so3
from .model_selection import init_bps
from ..ExVol.EVBeads import EVBeads
from ..BPStep.BPS_LRBP import LRBP
from ..BPStep.RBPStiff import GenStiffness

def equilibrate(
    triads: np.ndarray,
    positions: np.ndarray,
    sequence: str,
    closed: bool = False,
    endpoints_fixed: bool = True,
    fixed: List[int] = [],
    temp: float = 300,
    num_cycles: int = None,
    # fixed_positions: List[int] = None,
    exvol_rad: float = 0,
    model: str = "lankas",
    dump_every: int = None,
    cycles_per_eval: int = 20,
    evals_per_average: int = 25,
) -> Dict:
    """_summary_

    Args:
        triads (np.ndarray): _description_
        positions (np.ndarray): _description_
        sequence (str): _description_
        closed (bool, optional): _description_. Defaults to False.
        endpoints_fixed (bool, optional): _description_. Defaults to True.
        fixed (List[int], optional): _description_. Defaults to [].
        temp (float, optional): _description_. Defaults to 300.
        num_cycles (int, optional): _description_. Defaults to None.
        exvol_rad (float, optional): _description_. Defaults to 0.
        model (str, optional): _description_. Defaults to 'lankas'.
        dump_every (int, optional): _description_. Defaults to None.
        cycles_per_eval (int, optional): _description_. Defaults to 20.
        evals_per_average (int, optional): _description_. Defaults to 25.

    Returns:
        Dict: _description_

    TODO:
        - assess convergence of energy
    """

    ################################
    # FIX THIS

    exvol_active = False
    keep_backup = False
    if exvol_rad > 0:
        keep_backup=True
        exvol_active=True
        

    ################################


    # # set excluded volume
    # evdist = args.evdist
    # check_crossings=True
    # if evdist > 0:
    #     print('####################################')
    #     print('Initiating Excluded Volume...')
    #     avgdist = np.mean(np.linalg.norm(gs[:,3:],axis=1))
    #     maxdist = 0.46
    #     maxdist = np.min([1.5*avgdist,evdist])
    #     EV = EVBeads(chain,ev_distance=evdist,max_distance=maxdist,check_crossings=check_crossings)
    # else:
    #     EV = None



    #############################
    # init configuration chain and energy
    conf = se3_triads(triads, positions)
    chain = Chain(conf, closed=closed, keep_backup=keep_backup)
    
    
    
    check_crossings=True
    exvol = None
    if exvol_active:
        evdist = exvol_rad * 2
        # exvol_active = True
        keep_backup = True
        print('####################################')
        print('Initiating Excluded Volume...')
        avgdist = np.mean(np.linalg.norm(positions[1:]-positions[:-1],axis=1))
        maxdist = 0.46
        maxdist = np.min([1.5*avgdist,evdist])
        exvol = EVBeads(chain,ev_distance=evdist,max_distance=maxdist,check_crossings=check_crossings)
        # print("Warning: Excluded volume interactions not yet implemented!")
    
    
    # if closed:
    #     seq = sequence + sequence[0]
    # else:
    #     seq = str(sequence)  
    # genstiff = GenStiffness(method='md')
    # stiff,gs = genstiff.gen_params(seq, use_group = True, sparse = True)
    
    # print(stiff.shape)
    # print(conf.shape)
    # sys.exit()
    
    # ncoup = 0
    # bps = RBP(chain,seq,gs,stiff,ncoup,closed=closed,static_group=True)
    
    specs = {"method": "MD", "gs_key": "group_gs", "stiff_key": "group_stiff"}
    bps = LRBP(chain, sequence, specs, closed=closed, static_group=True, temp=temp)
    
    
    
    # bps = init_bps(model, chain, sequence, closed=closed, temp=temp)
    N = len(conf)

    #############################
    # set fixed and free points
    if endpoints_fixed:
        if 0 not in fixed:
            fixed = [0] + fixed
        if N - 1 not in fixed:
            fixed += [N - 1]
    bpids = [i for i in range(N)]
    free = [i for i in bpids if i not in fixed]

    #############################
    # init moves
    moves = list()

    # add single triad moves:
    single = SingleTriad(chain, bps, selected_triad_ids=free, exvol=exvol)
    Nsingle = len(free) // 4
    if Nsingle == 0 and len(free) > 0:
        Nsingle = 1
    moves += [single for i in range(Nsingle)]

    if closed:
        if len(fixed) == 0:
            # add crankshaft
            cs = Crankshaft(chain, bps, 2, N // 2, exvol=exvol)
            ct = ClusterTrans(chain, bps, 2, N // 2, exvol=exvol)
            moves += [cs, ct]
        else:
            # add crankshaft moves on intervals
            #  -> this will be replaced by a single move with multiple interval assignments
            for fid in range(1, len(fixed)):
                f1 = fixed[fid - 1] + 1
                f2 = fixed[fid]
                diff = f2 - f1
                if diff > 4:
                    rge = np.min([N // 2, diff])
                    cs = Crankshaft(
                        chain, bps, 2, rge, range_id1=f1, range_id2=f2, exvol=exvol
                    )
                    ct = ClusterTrans(
                        chain, bps, 2, rge, range_id1=f1, range_id2=f2, exvol=exvol
                    )
                    moves += [cs, ct]

            # between last and first fix
            f1 = fixed[-1] + 1
            f2 = fixed[0]
            diff = f2 - f1 + N
            if diff > 4:
                rge = np.min([N // 2, diff])
                cs = Crankshaft(
                    chain, bps, 2, rge, range_id1=f1, range_id2=f2, exvol=exvol
                )
                ct = ClusterTrans(
                    chain, bps, 2, rge, range_id1=f1, range_id2=f2, exvol=exvol
                )
                moves += [cs, ct]

    else:
        if not endpoints_fixed:
            # endpoints open
            if len(fixed) == 0:
                pv1 = Pivot(chain, bps, rotate_end=False, exvol=exvol)
                pv2 = Pivot(chain, bps, rotate_end=True, exvol=exvol)
                moves += [pv1, pv2]
            else:
                if fixed[0] > 4:
                    pv1 = Pivot(
                        chain,
                        bps,
                        rotate_end=False,
                        exvol=exvol,
                        selection_limit_id=fixed[0],
                    )
                    moves.append(pv1)
                if fixed[-1] < N - 5:
                    pv2 = Pivot(
                        chain,
                        bps,
                        rotate_end=True,
                        exvol=exvol,
                        selection_limit_id=fixed[-1] + 1,
                    )
                    moves.append(pv2)

        else:
            for fid in range(1, len(fixed)):
                f1 = fixed[fid - 1] + 1
                f2 = fixed[fid]
                diff = f2 - f1
                if diff > 4:
                    rge = np.min([N // 2, diff])
                    cs = Crankshaft(
                        chain, bps, 2, rge, range_id1=f1, range_id2=f2, exvol=exvol
                    )
                    ct = ClusterTrans(
                        chain, bps, 2, rge, range_id1=f1, range_id2=f2, exvol=exvol
                    )
                    moves += [cs, ct]

    #############################
    # simulation loop
    confs = []
    confs.append(np.copy(chain.conf[:, :3, 3]))
    energies = []
    energies.append(bps.get_total_energy())

    if num_cycles is not None:
        for cyc in range(num_cycles):
            for move in moves:
                move.mc()

            if cyc % 10 == 0:
                energies.append(bps.get_total_energy())
            if dump_every is not None and cyc % dump_every == 0:
                confs.append(np.copy(chain.conf[:, :3, 3]))
            if cyc % 1000 == 0:
                print(f"cycle {cyc}: ")

    else:
        for cyc in range(cycles_per_eval * evals_per_average * 2):
            for move in moves:
                move.mc()
            if cyc % cycles_per_eval == 0:
                energies.append(bps.get_total_energy())
            if dump_every is not None and cyc % dump_every == 0:
                confs.append(np.copy(chain.conf[:, :3, 3]))

        Em1 = np.mean(energies[:evals_per_average])
        Em2 = np.mean(energies[evals_per_average:])
        equi_down = Em1 > Em2
        print(f"E = {Em1} kT")
        print(f"E = {Em2} kT")
        cyc = 0
        equilibrated = False
        while not equilibrated:
            # equilibration check
            for eval in range(evals_per_average):
                for c in range(evals_per_average):
                    cyc += 1
                    for move in moves:
                        move.mc()
                    if cyc % 100 == 0:
                        confs.append(np.copy(chain.conf[:, :3, 3]))
                energies.append(bps.get_total_energy())

            Em1 = Em2
            Em2 = np.mean(energies[-evals_per_average:])
            print(f"E = {Em2} kT")
            if equi_down:
                if Em2 > Em1:
                    equilibrated = True
                    break
            else:
                if Em2 < Em1:
                    equilibrated = True
                    break

    out = {
        "positions": chain.conf[:, :3, 3],
        "triads": chain.conf[:, :3, :3],
        "elastic": bps.get_total_energy(),
        "confs":    confs,
    }
    return out


if __name__ == "__main__":
    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    # npb = 25
    # closed = False
    # endpoints_fixed = False
    # fixed = []
    # temp = 300

    # conf = np.zeros((npb, 4, 4))
    # gs = np.array([0, 0, 0.6, 0, 0, 0.34])
    # g = so3.se3_euler2rotmat(gs)
    # conf[0] = np.eye(4)
    # for i in range(1, npb):
    #     g = so3.se3_euler2rotmat(gs + np.random.normal(0, 0.1, 6))
    #     conf[i] = conf[i - 1] @ g

    # seq = "".join(["ATCG"[np.random.randint(4)] for i in range(npb)])

    # triads = conf[:, :3, :3]
    # pos = conf[:, :3, 3]

    # print("first")
    # out = equilibrate(
    #     triads,
    #     pos,
    #     seq,
    #     closed=False,
    #     endpoints_fixed=False,
    #     fixed=[],
    #     temp=100000,
    #     num_cycles=100,
    # )

    # # fixed = [10,20,30,70]
    # # fixed_pos1 = np.copy(out['positions'][fixed])

    # print("second")
    # out = equilibrate(
    #     out["triads"],
    #     out["positions"],
    #     seq,
    #     closed=closed,
    #     endpoints_fixed=endpoints_fixed,
    #     fixed=fixed,
    #     temp=300,
    # )

    # # fixed_pos2 = np.copy(out['positions'][fixed])
    # # print(fixed_pos2-fixed_pos1)

    # from ..Dumps.xyz import write_xyz

    # types = ["C" for i in range(len(conf))]
    # data = {"pos": out["confs"], "types": types}
    # write_xyz("test_equi.xyz", data)
    
    
    # import matplotlib.pyplot as plt
    # def plot(pos, 
    #     triads):
    
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111,projection='3d')
    #     ax = plt.gca(projection='3d')
    #     for position, frame in zip(pos,triads):
    #         right, up, forward = frame.T
            
    #         ax.quiver(*position,*right,length=0.2,color='g')
    #         ax.quiver(*position,*up,length=0.2,color='b')
    #         ax.quiver(*position,*forward,length=0.2,color='r')
        
    #     ax.plot(*pos.T,color='black',label='Control Points',lw=1)
        
    #     com = np.mean(pos,axis=0)
    #     maxrge = np.max([np.max(pos[:,i])-np.min(pos[:,i]) for i in range(3)])
    #     margin = maxrge*0.01
    #     halfrge = maxrge*0.5+margin
    #     rges = []
    #     for i in range(3):
    #         rges.append([com[i]-halfrge,com[i]+halfrge])
    #     for x in range(2):
    #         for y in range(2):
    #             for z in range(2):
    #                 ax.scatter([rges[0][x]],[rges[1][y]],[rges[2][z]],alpha=0.01)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.legend()
    #     plt.show()
        
    # plot(out['positions'],out['triads'])
    
    
    def gen_circular(nbp: int, lk: float, disc_len: float):
        conf = np.zeros((nbp,4,4))
        conf[:,3,3] = 1
        dtheta = 2*np.pi / nbp
        conf[:,:3,3] = [[np.cos(i*dtheta),np.sin(i*dtheta),0] for i in range(nbp)]
        conf[:,:3,3] *= disc_len / np.linalg.norm(conf[1,:3,3]-conf[0,:3,3])
        for i in range(nbp):
            t = conf[(i+1)%nbp,:3,3] - conf[i,:3,3] 
            t = t / np.linalg.norm(t)
            b = np.array([0,0,1])
            n = np.cross(b,t)
            conf[i,:3,0] = n
            conf[i,:3,1] = b
            conf[i,:3,2] = t
        
        thpbp = lk*2*np.pi / nbp
        for i in range(nbp):
            theta = i*thpbp
            R = so3.euler2rotmat(np.array([0,0,theta]))
            conf[i,:3,:3] = conf[i,:3,:3] @ R 
        # plot(conf[:,:3,3],conf[:,:3,:3]=
        return conf 
    
    endpoints_fixed = False
    fixed = []
    
    nbp = 300
    dlk = 5
    disc_len = 0.34
    closed = True
    temp = 300
    exvol_rad = 2
    
    lk = nbp//10.5 + dlk    
    conf = gen_circular(nbp,lk,disc_len)
    seq = "".join(["ATCG"[np.random.randint(4)] for i in range(nbp)])

    triads = conf[:, :3, :3]
    pos = conf[:, :3, 3]

    print("first")
    out = equilibrate(
        triads,
        pos,
        seq,
        closed=closed,
        exvol_rad=exvol_rad,
        endpoints_fixed=False,
        fixed=[],
        temp=300,
        num_cycles=25000,
        dump_every=100
    )

    # fixed = [10,20,30,70]
    # fixed_pos1 = np.copy(out['positions'][fixed])

    # print("second")
    # out = equilibrate(
    #     out["triads"],
    #     out["positions"],
    #     seq,
    #     closed=closed,
    #     endpoints_fixed=endpoints_fixed,
    #     fixed=fixed,
    #     temp=300,
    # )

    # fixed_pos2 = np.copy(out['positions'][fixed])
    # print(fixed_pos2-fixed_pos1)

    from ..Dumps.xyz import write_xyz

    types = ["C" for i in range(len(conf))]
    data = {"pos": out["confs"], "types": types}
    write_xyz("test_equi.xyz", data)
    
    
    taus = np.zeros((len(out["positions"]),4,4))
    taus[:,:3,:3] = out["triads"]
    taus[:,3,:3] = out["positions"]
    np.save('taus.npy',taus)