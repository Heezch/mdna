from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from scipy.optimize import curve_fit
import warnings

try:
    from numba import jit
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
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
from ..Evals.PyLk.pylk import writhe, triads2link


RUN_MOVES_PER_FREE = 0.1

class Run:
    
    def __init__(
        self,
        triads: np.ndarray,
        positions: np.ndarray,
        sequence: str,
        closed: bool = False,
        endpoints_fixed: bool = True,
        fixed: List[int] = [],
        temp: float = 300,
        exvol_rad: float = 0,
        check_crossings: bool = True,
        parameter_set: str = "md",
    ):
        """_summary_

        Args:
            triads (np.ndarray): _description_
            positions (np.ndarray): _description_
            sequence (str): _description_
            closed (bool, optional): _description_. Defaults to False.
            endpoints_fixed (bool, optional): _description_. Defaults to True.
            fixed (List[int], optional): _description_. Defaults to [].
            temp (float, optional): _description_. Defaults to 300.
            exvol_rad (float, optional): _description_. Defaults to 0.
            model (str, optional): _description_. Defaults to 'lankas'.
        """
               
        #####################################
        # init configuration chain and energy
        conf = se3_triads(triads, positions)
        self.nbp = len(conf)
        self.chain = Chain(conf, closed=closed, keep_backup=exvol_rad > 0)
        self.closed = closed
        
        #####################################
        # init energy
        specs = {"method": parameter_set, "gs_key": "group_gs", "stiff_key": "group_stiff"}
        self.bps = LRBP(self.chain, sequence, specs, closed=closed, static_group=True, temp=temp)
        
        #####################################
        # init excluded volume
        self.exvol = None
        if exvol_rad > 0:
            print('####################################')
            print('Initiating Excluded Volume...')
            self.evdist = exvol_rad * 2
            self.keep_backup = True
            avgdist = np.mean(np.linalg.norm(positions[1:]-positions[:-1],axis=1))
            maxdist = 0.46
            maxdist = np.min([1.5*avgdist,self.evdist])
            self.exvol = EVBeads(self.chain,ev_distance=self.evdist,max_distance=maxdist,check_crossings=check_crossings)
        
        #####################################
        # set fixed and free points
        if endpoints_fixed:
            fixed = list(set([0,self.nbp - 1]+fixed))   
        self.endpoints_fixed = endpoints_fixed
        self.fixed = sorted(fixed)            
        self.free = [i for i in range(self.nbp) if i not in fixed]
        
        #####################################
        # init moves
        self.init_moves()
        

    def init_moves(self, moves_per_free: float = RUN_MOVES_PER_FREE):
        
        self.moves = list()
        self.singles = list()
        
        #####################################
        # add single triad moves:
        single = SingleTriad(self.chain, self.bps, selected_triad_ids=self.free, exvol=self.exvol)
        Nsingle = int(np.ceil(len(self.free) * moves_per_free))
        self.singles += [single]*Nsingle

        #####################################
        # if closed
        if self.closed:
            if len(self.fixed) == 0:
                # add crankshaft
                cs = Crankshaft(
                    self.chain, self.bps, 2, self.nbp // 2, exvol=self.exvol
                )
                ct = ClusterTrans(
                    self.chain, self.bps, 2, self.nbp // 2, exvol=self.exvol
                )
                self.moves += [cs, ct]
            else:
                # add crankshaft moves on intervals
                #  -> this will be replaced by a single move with multiple interval assignments
                for fid in range(1, len(self.fixed)):
                    f1 = self.fixed[fid - 1] + 1
                    f2 = self.fixed[fid]
                    diff = f2 - f1
                    if diff > 4:
                        rge = np.min([self.nbp // 2, diff])
                        cs = Crankshaft(
                            self.chain, self.bps, 2, rge, range_id1=f1, range_id2=f2, exvol=self.exvol
                        )
                        ct = ClusterTrans(
                            self.chain, self.bps, 2, rge, range_id1=f1, range_id2=f2, exvol=self.exvol
                        )
                        self.moves += [cs, ct]

                # between last and first fix
                f1 = self.fixed[-1] + 1
                f2 = self.fixed[0]
                diff = f2 - f1 + self.nbp
                if diff > 4:
                    rge = np.min([self.nbp // 2, diff])
                    cs = Crankshaft(
                        self.chain, self.bps, 2, rge, range_id1=f1, range_id2=f2, exvol=self.exvol
                    )
                    ct = ClusterTrans(
                        self.chain, self.bps, 2, rge, range_id1=f1, range_id2=f2, exvol=self.exvol
                    )
                    self.moves += [cs, ct]
                            
        #####################################
        # if open
        else:
            
            ###########################################
            # crankshaft and cluster translation moves
            if len(self.fixed) == 0:
                cs = Crankshaft(
                    self.chain, self.bps, 2, self.nbp // 2, exvol=self.exvol
                )
                ct = ClusterTrans(
                    self.chain, self.bps, 2, self.nbp // 2, exvol=self.exvol
                )
                self.moves += [cs, ct]
            else:
                # add moves on intervals
                for fid in range(1, len(self.fixed)):
                    f1 = self.fixed[fid - 1] + 1
                    f2 = self.fixed[fid]
                    diff = f2 - f1
                    if diff > 4:
                        rge = np.min([self.nbp // 2, diff])
                        cs = Crankshaft(
                            self.chain, self.bps, 2, rge, range_id1=f1, range_id2=f2, exvol=self.exvol
                        )
                        ct = ClusterTrans(
                            self.chain, self.bps, 2, rge, range_id1=f1, range_id2=f2, exvol=self.exvol
                        )
                        self.moves += [cs, ct] 
                        
            ###########################################
            # pivot moves
            if not self.endpoints_fixed:
                
                # fully open
                if len(self.fixed) == 0:
                    # pivot moves in both directions
                    pv1 = Pivot(
                        self.chain, self.bps, rotate_end=False, exvol=self.exvol
                    )
                    pv2 = Pivot(
                        self.chain, self.bps, rotate_end=True, exvol=self.exvol
                    )
                    self.moves += [pv1, pv2]
  
                else:
                    # pivot moves on tail segments
                    if self.fixed[0] > 4:
                        pv1 = Pivot(
                            self.chain,
                            self.bps,
                            rotate_end=False,
                            exvol=self.exvol,
                            selection_limit_id=self.fixed[0],
                        )
                        self.moves.append(pv1)
                    if self.fixed[-1] < self.nbp - 5:
                        pv2 = Pivot(
                            self.chain,
                            self.bps,
                            rotate_end=True,
                            exvol=self.exvol,
                            selection_limit_id=self.fixed[-1] + 1,
                        )
                        self.moves.append(pv2)
                        
        moves_factor = int( moves_per_free * len(self.free) / len(self.moves) )
        if moves_factor > 0:
            self.moves *= moves_factor
         
    def run(self, cycles: int, dump_every: int = 0, start_id: int = 0) -> np.ndarray:
        if dump_every > 0:
            dump_confs = []
        for c in range(start_id,cycles+start_id):
            if dump_every > 0 and c%dump_every==0:
                dump_confs.append(np.copy(self.chain.conf))  
            for move in self.moves:
                move.mc()
            for single in self.singles:
                single.mc()
        if dump_every <= 0:
            return np.copy(self.chain.conf)
        else:
            return dump_confs
           
    
    def equilibrate_simple(self, 
        equilibrate_writhe: bool = True, 
        dump_every: int = 0,
        cycles_per_eval: int = 3,
        evals_per_average: int = 100,
        init_cycle_multiplier: int = 4,
        num_below_max: int = 3,
    ):
        
        n_cycle = 0
        dump_confs = []
        dump_energies = []
        
        equi_energies = []
        equi_writhe   = []

        E_equi = False
        wr_equi = not equilibrate_writhe
        
        ###################################################
        # calculate baseline 
        for eval in range(evals_per_average*init_cycle_multiplier*2):
            for c in range(cycles_per_eval):
                if n_cycle%dump_every==0:
                    dump_confs.append(np.copy(self.chain.conf))
                    dump_energies.append(self.bps.get_total_energy())
                self.run(1,dump_every=-1)
                n_cycle+=1
            equi_energies.append(self.bps.get_total_energy())
            if not wr_equi:
                equi_writhe.append(writhe(np.copy(self.chain.conf[:,:3,3]),closed=self.closed))

        ###################################################
        # evaluate baseline 
        mid = len(equi_energies)//2
        E1 = np.mean(equi_energies[:mid])
        E2 = np.mean(equi_energies[mid:])
        sign_dE = np.sign(E2-E1)
        # equi_E_down = E1 > E2
        print(f"E1 = %.2f kT"%(E1))
        print(f"E2 = %.2f kT"%(E2))
        E_curr = E2
        
        print(f'{wr_equi=}')
        if not wr_equi:
            wr1 = np.mean(equi_writhe[:mid])
            wr2 = np.mean(equi_writhe[mid:])
            sign_dwr = np.sign(wr2-wr1)
            print(f"wr1 = %.2f"%(wr1))
            print(f"wr2 = %.2f"%(wr2))
            wr_curr = wr2
        
        E_num_below = 0
        wr_num_below = 0
        
        ###################################################
        # main loop
        while (not E_equi) or (not wr_equi):
            equi_energies = []
            equi_writhe = []
            for eval in range(evals_per_average):
                for c in range(cycles_per_eval):
                    if n_cycle%dump_every==0:
                        dump_confs.append(np.copy(self.chain.conf))
                        dump_energies.append(self.bps.get_total_energy())
                    self.run(1,dump_every=-1)
                    n_cycle+=1
                if not E_equi:
                    equi_energies.append(self.bps.get_total_energy())
                if not wr_equi:
                    equi_writhe.append(writhe(np.copy(self.chain.conf[:,:3,3]),closed=self.closed))
            
            # evaluate energy equilibation
            if not E_equi:
                E_new = np.mean(equi_energies)
                print(f"E = %.2f kT"%(E_new))
                if E_new*sign_dE < E_curr*sign_dE:
                    E_num_below +=1
                    if E_num_below >= num_below_max:
                        E_equi = True
                else:
                    E_curr = E_new
                    E_num_below = 0
                print(f'{E_num_below=}')
            
            # evaluate writhe equilibation
            if not wr_equi:
                wr_new = np.mean(equi_writhe)
                print(f"wr = %.2f"%(wr_new))
                if wr_new*sign_dwr < wr_curr*sign_dwr:
                    wr_num_below +=1
                    if wr_num_below >= num_below_max:
                        wr_equi = True
                else:
                    wr_curr = wr_new
                    wr_num_below = 0
                print(f'{wr_num_below=}')
        
        # add last snapshot to output if applicable
        if n_cycle%dump_every==0:
            dump_confs.append(np.copy(self.chain.conf))
            dump_energies.append(self.bps.get_total_energy())
        
        # generate output dictionary
        out = {
            "last" :  np.copy(self.chain.conf),
            "energy": equi_energies,
            "confs":  np.array(dump_confs),
        }
        return out
    
    
    def equilibrate(self, 
        dump_every: int = 0,
        cycles_per_eval: int = 5,
        min_evals: int = 100,
        plot_equi: bool = False,
        num_taus: int = 6
    ):
        
        dump_confs = []
        energies = []
        energies.append(self.bps.get_total_energy())
        n_cycle = 0
        
        # run minimal equilibration
        for eval in range(min_evals):
            dump_confs += self.run(cycles_per_eval,dump_every=dump_every,start_id=n_cycle)
            n_cycle += cycles_per_eval
            energies.append(self.bps.get_total_energy())
            
        equi = False
        popt = None
        while not equi:
            dump_confs += self.run(cycles_per_eval,dump_every=dump_every,start_id=n_cycle)
            n_cycle += cycles_per_eval
            energies.append(self.bps.get_total_energy())
            popt = self._check_equilibration(energies,p0=popt)
            tau = popt[-1]
            if len(energies) > tau*num_taus:
                equi = True
        en_popt = popt
           
        ens = np.zeros((len(energies),3)) 
        steps = np.arange(len(energies))
        ens[:,0] = steps*cycles_per_eval
        ens[:,1] = energies
        ens[:,2] = expdecay(steps,*en_popt)
                               
        if plot_equi: 
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(3, 2), dpi=300,facecolor='w',edgecolor='k') 
            ax1 = plt.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1,rowspan=1)
            steps = np.arange(len(energies))
            ax1.plot(ens[:,0], ens[:,1], lw=1,color='black')
            ax1.plot(ens[:,0], ens[:,2],color='red',ls='--',lw=1)
            
            #### Set Labels
            ax1.tick_params(axis="both",which='major',direction="in",labelsize=6)
            ax1.set_xlabel('MC cycles',fontsize=7)
            ax1.set_ylabel(r'Energy ($k_B T$)',fontsize=7)
            ax1.xaxis.set_label_coords(0.5,-0.1)
            ax1.yaxis.set_label_coords(-0.1,0.5)
            
            plt.subplots_adjust(
                left=0.14,
                right=0.97,
                bottom=0.12,
                top=0.88,
                wspace=3.5,
                hspace=1
            )
            plt.show()
            
        # add last snapshot to output if applicable
        if n_cycle%dump_every==0:
            dump_confs.append(np.copy(self.chain.conf))
        
        # generate output dictionary
        out = {
            "last" :  np.copy(self.chain.conf),
            "confs":  np.array(dump_confs),
            "energies" : ens
        }
        return out
            

    def _check_equilibration(self,series,maxfev=25000,p0=None):
        steps = np.arange(len(series))
        if p0 is None:
            A = series[-1]
            B = series[0]-series[-1]
            if B < 0:
                tau = np.argmax(series)
            else:
                tau = np.argmin(series)
            p0 = [A,B,tau]
        popt, pcov = curve_fit(expdecay, steps, series ,maxfev=maxfev,p0=p0)
        return popt
           
            
def expdecay(steps,A,B,tau):
    return A+B*np.exp(-steps/tau)
            

if __name__ == "__main__":
    np.set_printoptions(linewidth=250, precision=3, suppress=True)


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
    
    nbp = 315
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

    print(f'link = {triads2link(pos,triads)}')

    run = Run(
        triads,
        pos,
        seq,
        closed=closed,
        endpoints_fixed=True,
        fixed=[],
        temp=300,
        exvol_rad=exvol_rad,
        check_crossings=True,
        parameter_set='md'
    )
    
    out = run.equilibrate_simple(equilibrate_writhe=True,dump_every=10)
    # out = run.equilibrate(dump_every=10,plot_equi=True)

    from ..Dumps.xyz import write_xyz
    types = ["C" for i in range(len(conf))]
    data = {"pos": out["confs"][:,:,:3,3], "types": types}
    write_xyz("test_equi.xyz", data)
