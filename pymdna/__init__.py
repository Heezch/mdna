from .utils import  *
from .spline import SplineFrames, Twister
from .geometry import ReferenceBase, NucleicFrames, NucleicFrames_quaternion
from .generators import SequenceGenerator, StructureGenerator
from .modifications import Mutate, Hoogsteen, Methylate
from .analysis import GrooveAnalysis, TorsionAnalysis, ContactCount
from .build import Build
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


def _check_input(sequence=None, n_bp=None):
    """Check the input sequence and number of base pairs"""

    if sequence is None and n_bp is not None:
        sequence = ''.join(np.random.choice(list('ACGT'), n_bp))
        print('Random sequence:', sequence,'\n')

    elif sequence is not None and n_bp is None:
        n_bp = len(sequence)

    elif sequence is None and n_bp is None:
        sequence = 'CGCGAATTCGCG'
        n_bp = len(sequence)
        print('Default sequence:', sequence)
        print('Number of base pairs:', n_bp,'\n')

    elif sequence is not None and n_bp is not None:
        if n_bp != len(sequence):
            raise ValueError('Sequence length and n_bp do not match','\n')
    return sequence, n_bp


def load(traj=None, frames=None, sequence=None, chainids=[0,1]):
    """Load DNA representation from:
        - base step mean reference frames/spline frames
        - or MDtraj trajectory
    Args:
        frames: np.array 
            base step mean reference frames of shape (n_bp, n_timesteps, 4, 3)
            or (n_bp, 4, 3)
        traj: object
            mdtraj trajectory
        sequence: str
            DNA sequence corresponding to the frames
        chainids: list
            chain ids of the DNA structure
    Returns:
        Nucleic object
    Raises:
        ValueError: If the input arguments are not consistent or missing required information.
    """
    return Nucleic(sequence=sequence, n_bp=None, traj=traj, frames=frames, chainids=chainids)

def make(control_points: np.ndarray, sequence: str = None,closed: bool = False, n_bp : int = None, dLk : int = None):
    """Generate DNA structure from sequence and control points
    Args:
        sequence: DNA sequence
        control_points: control points of shape (n,3) with n > 3
        closed: is the DNA structure circular
        n_bp: number of base pairs to scale shape with
        dLk: Change in twist in terms of Linking number of DNA structure to output
    Returns:
        Nucleic object"""


    sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)
    spline = SplineFrames(control_points=control_points, n_bp=n_bp, closed=closed, dLk=dLk)
    generator = StructureGenerator(sequence=sequence, spline=spline, circular=closed)
    frames = generator.frames

    return Nucleic(sequence=sequence, n_bp=n_bp, frames=frames, chainids=[0,1])


def compute_rigid_parameters(traj, chainids=[0,1]):
    """Compute the rigid base parameters of the DNA structure
    Args:
        traj: trajectory
        chainids: chain ids
    Returns:
        rigid base object"""

    return NucleicFrames(traj, chainids)


def sequence_to_pdb(sequence='CGCGAATTCGCG', filename='my_dna', save=True, output='GROMACS',shape=None,n_bp=None,circular=False,dLk=None):
    """_summary_

    Parameters
    ----------
    sequence : str, optional
        DNA sequence code, by default 'CGCGAATTCGCG'
    filename : str, optional
        filename for pdb output, by default 'my_dna'
    save : bool, optional
        boolean to save pdb or not, by default True
    output : str, optional
        Type of pdb DNA format, by default 'GROMACS'
    shape : ndarray, optional
        control_points of shape (n,3) with n > 3 that is used for spline interpolation to determine DNA shape, by default None which is a straight line
    n_bp : _type_, optional
        Number of base pairs to scale shape with, by default None then sequence is used to determine n_bp
    circular : bool, optional
        Key that tells if structure is circular/closed, by default False
    dLk : int, optional
        Change in twist in terms of Linking number of DNA structure to output, by default None (neutral twist base on bp_per_turn = 10.5)

    Returns
    -------
    MDtraj object
        returns MDtraj trajectory object of DNA structure (containing only a single frame)
    """

    sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)

    # Linear strand of control points 
    if shape is None:
        shape = Shapes.line(length=1)
        # Convert the control points to a spline
    spline = SplineFrames(control_points=shape, n_bp=n_bp,closed=circular,dLk=dLk)
    # Generate the DNA structure
    generator = StructureGenerator(sequence=sequence,spline=spline, circular=circular)

    # Edit the DNA structure to make it compatible with the AMBER force field
    traj = generator.traj
    if output == 'GROMACS':
        phosphor_termini = traj.top.select(f'name P OP1 OP2 and resid 0 {traj.top.chain(0).n_residues}')
        all_atoms = traj.top.select('all')
        traj = traj.atom_slice([at for at in all_atoms if at not in phosphor_termini])

    # Save the DNA structure as pdb file
    if save:
        traj.save(f'./{filename}.pdb')

    return traj

def sequence_to_md(sequence=None, time=10, time_unit='picoseconds',temperature=310, solvated=False,  filename='my_dna', save=True, output='GROMACS',shape=None,n_bp=None,circular=False,dLk=None):
    """Simulate DNA sequence using OpenMM
        Args:
            sequence: DNA sequence
            time: simulation time
            time_unit: time unit
            temperature: temperature
            solvated: solvate DNA
        Returns:
            mdtraj trajectory"""
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        from mdtraj.reporters import HDF5Reporter
        import mdtraj as md
        openmm_available = True
    except ImportError:
        openmm_available = False
        print("Openmm is not installed. You shall not pass.")

    pdb = sequence_to_pdb(sequence=sequence, filename='my_dna', save=True, output='GROMACS',shape=None,n_bp=None,circular=False,dLk=None)
    
    if not openmm_available:
        print('But here is your DNA structure')
        return pdb
    else:
        if time_unit == 'picoseconds':
            time_unit = time * unit.picoseconds
        elif time_unit == 'nanoseconds':
            time_unit = time * unit.nanoseconds

        time = time * time_unit
        time_step = 2 * unit.femtoseconds
        temperature = 310 *unit.kelvin
        steps = int(time/time_step)

        print(f'Initialize DNA openMM simulation at {temperature._value} K for', time, 'time units')
        topology = pdb.topology.to_openmm()
        modeller = app.Modeller(topology, pdb.xyz[0])

        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        modeller.addHydrogens(forcefield)
        if solvated:
            print('Solvate DNA with padding of 1.0 nm and 0.1 M KCl')
            modeller.addSolvent(forcefield, padding=1.0*unit.nanometers, ionicStrength=0.1*unit.molar, positiveIon='K+')

        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.CutoffNonPeriodic)
        integrator = mm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, time_step)

        simulation = app.Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        simulation.reporters.append(HDF5Reporter(f'./{sequence}'+'.h5', 100))
        simulation.reporters.append(app.StateDataReporter(f'./output_{sequence}.csv', 100, step=True, potentialEnergy=True, temperature=True,speed=True))
        
        print('Minimize energy')
        simulation.minimizeEnergy()
        
        print('Run simulation for', steps, 'steps')
        simulation.step(steps)
        simulation.reporters[0].close()
        print('Simulation completed')
        print('Saved trajectory as:', f'./{sequence}'+'.h5')
        traj = md.load_hdf5(f'./{sequence}'+'.h5')
        return traj





# class Minimize:
#     def 
#NucleicFrames, Mutate, Hoogsteen, Methylate, Build, Minimize):


class Nucleic():
        
        """Contains mdna DNA structure with reference frames and trajectory"""

        def __init__(self, sequence=None, n_bp=None, traj=None, frames=None, chainids=None):
            """Initialize the DNA structure
            Args:
                sequence: DNA sequence
                n_bp: number of base pairs
                traj: trajectory
                frames: reference frames
                chainids: chain ids"""


            if traj is not None:
                if frames is not None:
                    raise ValueError('Provide either a trajectory or reference frames, not both')
                # Extract sequence from the trajectory
                sequence = get_sequence_letters(traj, leading_chain=chainids[0])
                n_bp = len(sequence)
                frames = None  # Nucleic class will handle extraction from traj

            elif frames is not None:
                if frames.ndim == 3:
                    # Case (n_bp, 4, 3)
                    frames = np.expand_dims(frames, axis=1)
                if frames.ndim != 4:
                    raise ValueError('Frames should be of shape (n_bp, n_timesteps, 4, 3) or (n_bp, 4, 3)')
                n_bp = frames.shape[0]
                if sequence is not None:
                    if len(sequence) != n_bp:
                        raise ValueError('Number of base pairs in the sequence and frames do not match')  
                    else:
                        sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)      
            else:
                raise ValueError('Provide either a trajectory or reference frames')


            self.sequence, self.n_bp = sequence, n_bp
            self.traj = traj
            self.frames = frames
            self.chainids = chainids
            self.circular = self._is_circular()
            self.rigid = None # Container for rigid base parameters class output

        def describe(self):
            """Print the DNA structure information"""
            print(f'{"Circular " if self.circular else ""}DNA structure with {self.n_bp} base pairs')
            print('Sequence:', ''.join(self.sequence))
            if self.traj is not None:
                print('Trajectory:',self.traj)
            else:
                print('Trajectory not loaded')
            if self.frames is not None:
                print('Frames: ', self.frames.shape)
            else:
                print('Frames not loaded')
                
        def frames_to_traj(self, frame=-1):
            """Convert reference frames to trajectory"""
            if self.frames is None:
                raise ValueError('Load reference frames first')
            self.traj = StructureGenerator(frames=self.frames[:,frame,:,:], sequence=self.sequence, circular=self.circular).get_traj()
        
        def traj_to_frames(self):
            """Convert trajectory to reference frames"""
            if self.traj is None:
                raise ValueError('Load trajectory first')
            self.rigid = NucleicFrames(self.traj, self.chainids)
            self.frames =self.rigid.frames
        
        def get_frames(self):
            """Get the reference frames of the DNA structure belonging to the base steps:
            Returns: array of reference frames of shape (n_frames, n_bp, 4, 3)
            where n_frames is the number of frames, n_bp is the number of base pairs, 
            and 4 corresponds to the origin and the 3 vectors of the reference frame"""
            if self.frames is None:
                self.traj_to_frames()
            return self.frames
        
        def get_traj(self):
            """Get the trajectory"""
            if self.traj is None:
                self.frames_to_traj()
            return self.traj
        
        def get_rigid_parameters(self):
            """Get the rigid base parameters class object of the DNA structure"""
            if self.rigid is None and self.traj is not None:
                self.rigid = NucleicFrames(self.traj, self.chainids)
                return self.rigid
            elif self.rigid is None and self.traj is None:
                self.frames_to_traj()
                self.rigid = NucleicFrames(self.traj, self.chainids)
                return self.rigid
            else:
                return self.rigid
        
        def _is_circular(self, frame=0):
            """Detect if the DNA structure is circular for a given chain and frame

            Parameters
            ----------
            chainid : int, optional
                ID of the chain to check, by default 0
            frame : int, optional
                Frame index to check, by default 0

            Returns
            -------
            bool
                True if the DNA is circular, False otherwise
            """
            if self.frames is None:
                self.traj_to_frames()
                
            start = self.frames[0,frame,0]
            end = self.frames[-1,frame,0]
            
            distance = np.linalg.norm(start - end)
            return distance < 0.4 # 0.34 nm is the distance between base pairs
        

        # REBUILDING DNA STRUCTURE
        
        # def equilibrate(self,frame=0):
        #     """Equilibrate the DNA structure"""
        #     pass


        # PLOTTING DNA STRUCTURE
    

        def _plot_chain(self, ax, traj, chainid, frame, lw=1, markersize=2, color='k'):
            """Plot the DNA structure of a chain"""
            phosphor = traj.top.select(f'name P and chainid {chainid}')
            x = traj.xyz[frame, phosphor, 0]
            y = traj.xyz[frame, phosphor, 1]
            z = traj.xyz[frame, phosphor, 2]
            
            ax.plot(x, y, z, '-o', c=color, markersize=markersize*1.2, lw=lw)
            
            if self.circular:
                # Connect the last point to the first point
                ax.plot([x[-1], x[0]], [y[-1], y[0]], [z[-1], z[0]], '-o', c=color, markersize=markersize*1.2, lw=lw)

        def _plot_helical_axis(self, ax, frame, lw=1):
            helical_axis = self.frames[:,frame,0]
            ax.plot(helical_axis[:,0],helical_axis[:,1],helical_axis[:,2],':',c='k',lw=lw*0.7)
            if self.circular:
                ax.plot([helical_axis[-1,0],helical_axis[0,0]],[helical_axis[-1,1],helical_axis[0,1]],[helical_axis[-1,2],helical_axis[0,2]],':',c='k',lw=lw*0.7)

        def draw(self, ax=None, fig=None, save=False, frame=-1, markersize=2, lw=1, helical_axis=True, backbone=True, lead=False, anti=False, triads=False, length=0.23,color_lead='k',color_anti='darkgrey'):
            """Draw 3D representation of the DNA structure with optial helical axis, backbone, lead, anti, and triads

            Parameters
            ----------
            ax : object, optional
                matplotlib axis, by default None
            fig : object, optional
                figure axis, by default None
            save : bool, optional
                save image with name png, by default False
            frame : int, optional
                index of trajectory to visualize, by default 0
            markersize : int, optional
                width of backbone plot, by default 2
            helical_axis : bool, optional
                central axis passing through frame orgins, by default True
            backbone : bool, optional
                'o-' line plot through phosphor atoms, by default True
            lead : bool, optional
                plot leading strand, by default False
            anti : bool, optional
                plot anti sense opposing leading strand, by default False
            triads : bool, optional
                plot triads in order of  b_L (blue), b_N (green), b_T (red), by default False
            length : float, optional
                length of triad vectors, by default 0.23
            """

            # TODO: handle circular DNA and when trajetory is not loaded make frames uniform 
            # in shape (time/n_frames, n_bp, 4, 3)

            if self.traj is None:
                self.frames_to_traj()
            elif self.frames is None:
                self.traj_to_frames()
                    
            if fig is None and ax is None:
                fig = plt.figure()#figsize=(4,4))
                ax = fig.add_subplot(111, projection='3d')

            if backbone:
                lead = True
                anti = True
            if lead:
                self._plot_chain(ax, self.traj, 0, frame=frame, markersize=markersize, lw=lw, color=color_lead)
            if anti:
                self._plot_chain(ax, self.traj, 1, frame=frame, markersize=markersize, lw=lw, color=color_anti)
            if helical_axis:
                self._plot_helical_axis(ax, frame=frame, lw=lw)
            if triads:
                for triad in self.frames:
                    triad = triad[frame]
                    ax.scatter(triad[0,0],triad[0,1],triad[0,2],c='k',s=markersize*1.2)
                    ax.quiver(triad[0,0],triad[0,1],triad[0,2],triad[1,0],triad[1,1],triad[1,2],color='b',length=length)
                    ax.quiver(triad[0,0],triad[0,1],triad[0,2],triad[2,0],triad[2,1],triad[2,2],color='g',length=length)
                    ax.quiver(triad[0,0],triad[0,1],triad[0,2],triad[3,0],triad[3,1],triad[3,2],color='r',length=length)
            
            ax.axis('equal')
            ax.axis('off')
            if save:
                fig.savefig('dna.png', dpi=300,bbox_inches='tight')

        def minimize(self, frame: int = -1, exvol_rad : float = 2.0, temperature : int = 300,  simple : bool = False, equilibrate_writhe : bool = False, endpoints_fixed : bool = True, fixed : List[int] = [], dump_every : int = 1):
            """
            Minimize the DNA structure.

            Args:
                frame (int): The trajectory frame to minimize. Defaults to -1.
                simple (bool): Whether to use simple equilibration. Defaults to False.
                equilibrate_writhe (bool): Whether to equilibrate writhe. Defaults to False. Only works for simple equilibration.
                endpoints_fixed (bool): Whether the endpoints are fixed. Defaults to True.
                fixed (list): List of fixed base pairs. Defaults to an empty list.
                exvol_rad (float): Excluded volume radius. Defaults to 2.0.
                temperature (int): Temperature for equilibration. Defaults to 300.
                dump_every (int): Frequency of dumping frames. Defaults to 1.

            Additional keyword arguments can be provided and will be passed to the minimizer.

            Notes:

                For the simple equilibation, we rely on checking whether the considered quantity starts to fluctuate around a fixed value. 
                This options is compatible with With the argument equilibrate_writhe, which you can specify that writhe should also be considered for equilibration. 
                
                The other option is to use the full equilibration, which is based on the actual energy of the system.
                We assume the energy to converge exponentially to the equilibrated value.
                This works fairly well for most examples I checked but is not entirely robust. 
                Considering autocorrelation has some issues when there are relaxations at different timescales.
                Also, I wasn't able to use something consistent to equilibrate writhe, since that involves a barrier crossing. 
                It is really non-trivial to set a criterion for whether or not a globally stable value is reached. 


            Example:
                nuc = load(traj)
                nuc.minimize(simple=True, temperature=310, exvol_rad=2.5)
            """
            minimizer = Minimizer(self)
            minimizer.minimize(frame=frame, exvol_rad=exvol_rad, temperature=temperature, simple=simple, equilibrate_writhe=equilibrate_writhe, endpoints_fixed=endpoints_fixed, fixed=fixed, dump_every=dump_every)    
            #self.frames = minimizer.frames
            self.frames_to_traj()



class Minimizer:
        
        def __init__(self, nucleic):
            # Dynamically set attributes from the nucleic instance
            self.__dict__.update(nucleic.__dict__)

            # Check if the required import is available
            if not self.check_import():
                raise ImportError("Run class from pmcpy.run.run is not available.")

        def check_import(self):
            try:
                from pmcpy.run.run import Run
                self.Run = Run  # Store the imported class in the instance
                return True
            except ImportError as e:
                print(f"ImportError: {e}")
                return False

        def _initialize_mc_engine(self):
            """Initialize the Monte Carlo engine"""
    
            
            pos = self.frames[:,self.frame,0,:]
            triads = self.frames[:,self.frame,1:,:].transpose(0,2,1) # flip row vectors to column vectors

            mc = self.Run(triads=triads,positions=pos,
                            sequence=self.sequence,
                            closed=self.circular,
                            endpoints_fixed=self.endpoints_fixed,
                            fixed=self.fixed,
                            temp=self.temperature,
                            exvol_rad=self.exvol_rad)
            return  mc

            
        def update_frames(self, out):
 
            # update the spline with new positions and triads
            self.frames[:,self.frame,0,:] = out['positions'] # set the origins of the frames
            self.frames[:,self.frame,1:,:] = out['triads'].transpose(0,2,1) # set the triads of the frames as row vectors
            

        def _get_positions_and_triads(self, out):
            """Get the positions and triads from the output"""

            # get the positions and triads of the simulation
            positions = out['confs'][:,:,:3,3] 
            triads = out['confs'][:,:,:3,:3]

            # get the last frames of the simulation
            out['triads'] = triads[-1]
            out['positions'] = positions[-1]

            return positions, triads.transpose(0,1,3,2) # flip column vectors to row vectors


        def minimize(self,  frame: int = -1, exvol_rad : float = 2.0, temperature : int = 300,  simple : bool = False, equilibrate_writhe : bool = False, endpoints_fixed : bool = True, fixed : List[int] = [], dump_every : int = 1):

            # Set the parameters
            self.endpoints_fixed = endpoints_fixed
            self.fixed = fixed
            self.exvol_rad = exvol_rad
            self.temperature = temperature
            self.frame = frame

            print('Minimize the DNA structure:\nsimple equilibration =', simple, 'equilibrate writhe =', equilibrate_writhe, 'excluded volume radius =', exvol_rad, 'temperature =', temperature)
            minimizer = self._initialize_mc_engine()    
            
            # Run the Monte Carlo simulation
            if simple:
                out = minimizer.equilibrate_simple(equilibrate_writhe=equilibrate_writhe,dump_every=dump_every)
            else:
                if equilibrate_writhe:
                    raise ValueError("Equilibration of writhe is only supported for simple equilibration.")
                out = minimizer.equilibrate(dump_every=dump_every,plot_equi=True)

            positions, triads = self._get_positions_and_triads(out)

            self.update_frames(out)
        

        # def run(self, cycles: int, dump_every: int = 0, start_id: int = 0) -> np.ndarray:
        #     """Run the Monte Carlo simulation."""

        #     mc = self._initialize_pmcpy()
        #     out = mc.run(cycles=cycles, dump_every=dump_every, start_id=start_id)

        #     positions, triads = self._get_positions_and_triads(out)



