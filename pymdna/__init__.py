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
import mdtraj as md
from scipy.spatial.transform import Rotation as R
import copy

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
    """
    return Nucleic(sequence=sequence, n_bp=None, traj=traj, frames=frames, chainids=chainids)

def make(sequence: str = None, control_points: np.ndarray = None, closed: bool = False, n_bp : int = None, dLk : int = None):
    """Generate DNA structure from sequence and control points
    Args:
        sequence: (optional) DNA sequence
        control_points: (optional) control points of shape (n,3) with n > 3, default is a straight line, see mdna.Shapes for more geometries
        closed: (default False) is the DNA structure circular
        n_bp: (optinal) number of base pairs to scale shape with
        dLk: (optinal) Change in twist in terms of Linking number of DNA structure to output
    Returns:
        Nucleic object"""

    # Check if control points are provided otherwise generate a straight line
    if control_points is not None:
        if  len(control_points) < 4:
            raise ValueError('Control points should contain at least 4 points [x,y,z]')
        elif len(control_points) > 4 and n_bp is None:
            n_bp = len(control_points) # Number of base pairs
    else:
        # Linear strand of control points
        control_points = Shapes.line(length=1)
    
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

    # TODO update with make function
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

    # TODO update with make function
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



class Nucleic:
        
        """Contains mdna DNA structure with reference frames and trajectory"""

        def __init__(self, sequence=None, n_bp=None, traj=None, frames=None, chainids=None):
            """Initialize the DNA structure
            Args:
                sequence: DNA sequence
                n_bp: number of base pairs
                traj: trajectory
                frames: reference frames
                chainids: chain ids"""

            # Check for trajectory
            if traj is not None:
                if frames is not None:
                    raise ValueError('Provide either a trajectory or reference frames, not both')
                # Extract sequence from the trajectory
                sequence = get_sequence_letters(traj, leading_chain=chainids[0])
                n_bp = len(sequence)
                frames = None  # Nucleic class will handle extraction from traj

            # Check for reference frames
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
            self.minimizer = None # Container for minimizer class output

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
                
        def _frames_to_traj(self, frame=-1):
            """Convert reference frames to trajectory"""
            if self.frames is None:
                raise ValueError('Load reference frames first')
            self.traj = StructureGenerator(frames=self.frames[:,frame,:,:], sequence=self.sequence, circular=self.circular).get_traj()
        
        def _traj_to_frames(self):
            """Convert trajectory to reference frames"""
            if self.traj is None:
                raise ValueError('Load trajectory first')
            self.rigid = NucleicFrames(self.traj, self.chainids)
            self.frames =self.rigid.frames
        
        def get_frames(self):
            """Get the reference frames of the DNA structure belonging to the base steps:
            Returns: array of reference frames of shape (n_frames, n_bp, 4, 3)
            where n_frames is the number of frames, n_bp is the number of base pairs, 
            and 4 corresponds to the origin and the 3 vectors of the reference frame
            
            Returns:
                np.ndarray: reference frames of the DNA structure"""

            if self.frames is None:
                self._traj_to_frames()
            return self.frames
        
        def get_traj(self):
            """Get the trajectory of the current state of the DNA structure
            Returns:
                MDtraj object"""
            if self.traj is None:
                self._frames_to_traj()
            return self.traj
        
        def get_rigid_parameters(self):
            """Get the rigid base parameters class object of the DNA structure
            Returns:
                NucleicFrames object"""
            if self.rigid is None and self.traj is not None:
                self.rigid = NucleicFrames(self.traj, self.chainids)
                return self.rigid
            elif self.rigid is None and self.traj is None:
                self._frames_to_traj()
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
                self._traj_to_frames()
                
            start = self.frames[0,frame,0]
            end = self.frames[-1,frame,0]
            distance = np.linalg.norm(start - end)

            # 0.34 nm is roughly the distance between base pairs and 20 is the minimum number of base pairs for circular DNA
            return distance < 0.5 and self.frames.shape[0] > 20 

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
            Returns
            -------
            object (optional)
                matplotlib figure

            """

            # TODO: handle circular DNA and when trajetory is not loaded make frames uniform 
            # in shape (time/n_frames, n_bp, 4, 3)

            if self.traj is None:
                self._frames_to_traj()
            elif self.frames is None:
                self._traj_to_frames()
                    
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
            self.minimizer = Minimizer(self)
            self.minimizer.minimize(frame=frame, exvol_rad=exvol_rad, temperature=temperature, simple=simple, equilibrate_writhe=equilibrate_writhe, endpoints_fixed=endpoints_fixed, fixed=fixed, dump_every=dump_every)    
            # Update the reference frames
            self._frames_to_traj()

        def get_MC_traj(self):
            """Get the MC sampling energy minimization trajectory of the new spline."""
            if self.minimizer is None:
                raise ValueError('Run minimization first')
            return self.minimizer.get_MC_traj()
 

        def mutate(self, mutations: dict = None, complementary : bool = True, frame : int = -1):
            """Mutate the DNA trajectory, updates the topology and coordinates of the DNA structure


            # Make this more comprehensive:
            Options for nucleobases are A, T, G, C
            Hachimoji: A, T, G, C, P, Z, S, B
            Fluorescent: A, T, G, C, F
            Hydrophobic pairs: A, T, G, C, I, O
            # Mutation (non-canonical//fluorescent base analogue)
            # P = 2AP (2-aminopurine)  https://doi.org/10.1002/anie.201001312 (2kv0)
            # D = tricyclic cytosin base analogue (1tuq)
            # Hydrophobic bases! Hachimoji DNA DOI: 10.1126/science.aat0971

            Parameters
            ----------
            mutation : dict
                Dictionary containing the mutation information. E.g. `mutations = {0: 'A', 1: 'T', 2: 'G'}` will mutate the first three base pairs to A, T, and G, respectively.
            complementary : bool, optional
                Whether to mutate the complementary strand, by default True
            frame : int, optional
                The frame to mutate, by default -1 (aka the last frame)
            """

            if self.traj is None:
                self._frames_to_traj()
            if mutations is None:
                raise ValueError('Provide a mutation dictionary')

            # TODO check if valid letters in muations dictionary

            mutant = Mutate(self.traj[frame], mutations,complementary=complementary)
            self.traj = mutant.get_traj()
            # update sequence
            self.sequence = ''.join(get_sequence_letters(self.traj, leading_chain=self.chainids[0]))

        def flip(self, fliplist: list = [], deg : int = 180, frame : int = -1):
            """Flip the nucleobases of the DNA structure"""
            
            if self.traj is None:
                self._frames_to_traj()
            if len(fliplist) == 0:
                raise ValueError('Provide a fliplist')

            flipper = Hoogsteen(self.traj, fliplist=fliplist,deg=deg,verbose=True)
            self.traj = flipper.get_traj()

        def methylate(self, methylations: list = [],  CpG : bool = False, leading_strand : int = 0, frame : int = -1):
            """Methylate the nucleobases of the DNA structure"""

            if self.traj is None:
                self._frames_to_traj()
            # if len(methylations) == 0:
            #     raise ValueError('Provide a methylation list')

            methylator = Methylate(self.traj, methylations=methylations,CpG=CpG,leading_strand=leading_strand)
            self.traj = methylator.get_traj()
        
        def extend(self, n_bp: int, sequence: str = None,fixed_endpoints: bool = False, forward: bool = True, frame: int = -1, shape: np.ndarray = None, margin : int = 1, minimize : bool = True):  
            """Extend the DNA sequence in the specified direction using the five_end or three_end as reference.

            Parameters
            ----------
            n_bp : int
                Number of base pairs to extend the DNA sequence
            sequence : str, optional
                DNA sequence to extend the DNA structure, by default None
            fixed_endpoints : bool, optional
                Fixed endpoint for extending the DNA sequence, by default False
            forward : bool, optional
                Extend the DNA sequence in the forward direction. If False, extend in the backward direction, by default True
            frame : int, optional
                The time frame to extend, by default -1
            shape : np.ndarray, optional
                control_points of shape (n,3) with n > 3, by default None
            margin : int, optional
                Number of base pairs to fix at the end, by default 1
            """

            if self.circular:
                raise ValueError('Cannot extend circular DNA structure')
            if not n_bp and not fixed_endpoint:
                raise ValueError("Either a fixed endpoint or a length must be specified for extension.")    
            if self.traj is None:
                self._frames_to_traj()
            if shape is None:
                shape = Shapes.line(length=1)
            if self.frames is None:
                self._traj_to_frames()

            # Check the input sequence and number of base pairs
            sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)

            extender = Extend(self, n_bp=n_bp, sequence=sequence, fixed_endpoints=fixed_endpoints, frame=frame, forward=forward, shape=shape, margin=margin)
            # Also update, n_bp, sequence, frames etc
            self.nuc = extender.nuc

            if minimize:
                self.nuc.minimize(fixed=extender.fixed, endpoints_fixed=fixed_endpoints)

            # Update attributes
            self.sequence = self.nuc.sequence
            self.traj = self.nuc.get_traj()
            self.frames = self.nuc.get_frames()
            self.n_bp = self.nuc.n_bp

        def invert(self):
            """Inverse the direction of the DNA structure so from 5' to 3' to 3' to 5'"""
            raise NotImplementedError('Not implemented yet')



def connect(Nucleic0, Nucleic1, sequence : str = None, n_bp : int =  None, leader: int = 0, frame : int = -1, margin : int = 1, minimize : bool = True, exvol_rad : float = 0.0, temperature : int = 300):
    """Connect two DNA structures by creating a new DNA structure with a connecting DNA strand. 
    The 3' end of the first DNA structure is connected to the 5' end of the second DNA structure.
    To connect the two strands we interpolate a straight line between the two ends,
    and distribute the optiminal number of base pairs that result in a neutral twist.

    Note:
    The minimization does not use excluded volume interactions by default. 
    This is because the excluded volume interactions require the EV beads to have no overlap. 
    However, how the initial configuration is generated, the EV beads are likely have overlap.

    If one desires the resulting Nucleic object can be minimized once more with the excluded volume interactions.

    Args:
        Nucleic0: Nucleic object
            First DNA structure to connect
        Nucleic1: Nucleic object
            Second DNA structure to connect
        sequence: str, optional
            DNA sequence of the connecting DNA strand, by default None
        n_bp: int, optional
            Number of base pairs of the connecting DNA strand, by default None
        leader: int, optional
            The leader of the DNA structure to connect, by default 0
        frame: int, optional
            The time frame to connect, by default -1
        margin: int, optional
            Number of base pairs to fix at the end, by default 1
        minimize : bool, optional
            Whether to minimize the new DNA structure, by default True

    Returns:
        Nucleic object: DNA structure with the two DNA structures connected
    """

    if Nucleic0.circular or Nucleic1.circular:
        raise ValueError('Cannot connect circular DNA structures')
    
    if sequence is not None and n_bp is None:
        n_bp = len(sequence)

    # Connect the two DNA structures
    connector = Connect(Nucleic0, Nucleic1, sequence=sequence, n_bp=n_bp, leader=leader, frame=frame, margin=margin)
    if minimize:
        connector.connected_nuc.minimize(exvol_rad=exvol_rad, temperature=temperature, fixed=connector.fixed)

    return connector.connected_nuc
    

class Connect:
    def __init__(self, Nucleic0, Nucleic1, sequence : str = None, n_bp : int =  None, leader: int = 0, frame : int = -1, margin : int = 1):
        
        # Store the two Nucleic objects
        self.Nucleic0 = Nucleic0
        self.Nucleic1 = Nucleic1
 
        # Might be possible sequence and n_bp are not falid
        self.sequence = sequence
        self.n_bp = n_bp
        self.frame = frame
        self.leader = leader
        self.margin = margin
        self.twist_tolerance = np.abs((360 / 10.4) - (360 / 10.6))

        # Get the frames of the two nucleic acids
        self.frames0 = Nucleic0.frames[:,self.frame,:,:]
        self.frames1 = Nucleic1.frames[:,self.frame,:,:]

        # Connect the two nucleic acids and store the new nucleic acid
        self.connected_nuc = self.connect()

    def connect(self, index=0):
        """Connect two nucleic acids by creating a new nucleic acid with a connecting DNA strand."""
        # Get the start and end points of the two nucleic acids (assuming the leader is 0 and we connect the end of A to start of B)
        self.start, self.end = self._get_start_and_end()
        rotation_difference = self._get_twist_difference()

        # Find optimal number of base pairs to match rotational difference between start and end
        if self.n_bp is None and self.sequence is None:
                        
            optimal_bps = self._find_optimal_bps(np.array([self.start[0], self.end[0]]), 
                                                bp_per_turn=10.5, 
                                                rise=0.34, 
                                                bp_range=1000, 
                                                rotation_difference=rotation_difference, 
                                                tolerance=self.twist_tolerance, 
                                                plot=False
                                                )
            
            # get the optimal number of base pairs (smallest amount of base pairs that satisfies the tolerance)
            self.n_bp = optimal_bps[index]['optimal_bp']
 
        # Guess the shape of the spline C by interpolating the start and end points
        # Note, we add to extra base pairs to account for the double count of the start and end points of the original strands
        control_points_C = self._interplotate_points(self.start[0], self.end[0], self.n_bp+2)
        distance = np.linalg.norm(self.start-self.end)

        # Create frames object with the sequence and shape of spline C while squishing the correct number of BPs in the spline
        spline_C = SplineFrames(control_points=control_points_C, frame_spacing=distance/len(control_points_C))

        # exclude first and last frame of C because they are already in spline A and B
        frames_C = np.concatenate([self.frames0,spline_C.frames[1:-1],self.frames1])

        # remember the fixed nodes/frames of A and B
        fixed_0 = list(range(self.frames0.shape[0]-self.margin))
        fixed_1 = list(range(frames_C.shape[0]-self.frames1.shape[0]+self.margin,frames_C.shape[0]))
        self.fixed = fixed_0 + fixed_1

        # Check if the sequence and number of base pairs are valid of the new connecting DNA
        self.sequence, self.n_bp = _check_input(sequence=self.sequence, n_bp=spline_C.frames.shape[0]-2)
        new_sequence = self.Nucleic0.sequence + self.sequence + self.Nucleic1.sequence
        self.n_bp = len(self.sequence)

        # Create a new Nucleic object with the new sequence and frames
        return Nucleic(sequence=new_sequence, frames=frames_C)

        
    def _get_start_and_end(self):
        """Get the start and end points of the two nucleic acids."""
        if self.leader == 0:
            start = self.Nucleic0.frames[-1, self.frame,:,:]
            end = self.Nucleic1.frames[0, self.frame,:,:]
        else:
            start = self.Nucleic1.frames[0, self.frame,:,:]
            end = self.Nucleic0.frames[-1, self.frame,:,:]

        return start, end

    def _compute_euler_angles(self, frame_A, frame_B):
        """Compute the Euler angles between two frames."""
        # Compute the rotation matrix R that transforms frame A to frame B
        rotation_matrix = np.dot(frame_B.T, frame_A)
        
        # Create a rotation object from the rotation matrix
        rotation = R.from_matrix(rotation_matrix)
        
        # Convert the rotation to Euler angles (ZYX convention)
        euler_angles = rotation.as_euler('zyx', degrees=True)
        
        # Return the Euler angles: yaw (Z), pitch (Y), and roll (X)
        return euler_angles

    def _get_twist_difference(self):
        """Calculates the twist difference between two frames."""
        b1 = self.start[1:]/np.linalg.norm(self.start[1:])
        b2 = self.end[1:]/np.linalg.norm(self.end[1:])

        euler_angles = self._compute_euler_angles(b1, b2)
        return euler_angles[-1]

    def _interplotate_points(self,start, end, n):
        """Interpolates n points between start and end."""
        return np.array([start + (end-start)*i/n for i in range(n+1)])

    def _find_minima(self, lst):
        """Finds the indices of local minima in a list."""
        return [i for i in range(1, len(lst) - 1) if lst[i - 1] > lst[i] and lst[i + 1] > lst[i]]

    def _compute_left_over(self, bp_range, min_bp, bp_per_turn, rotation_difference):
        """Computes the left-over rotational difference for a range of base pairs."""
        cumul_twist = np.arange(min_bp, min_bp + bp_range) * 360 / bp_per_turn
        return cumul_twist % 360 - rotation_difference

    def _compute_twist_diff_per_bp(self, optimal_bp, left_over, min_bp):
        """Calculates the twist difference per base pair for an optimal base pair number."""
        total_twist_diff = left_over[optimal_bp - min_bp]
        return total_twist_diff / optimal_bp

    def _check_within_tolerance(self, twist_diff_per_bp, tolerance):
        """Checks if the twist difference per base pair is within the specified tolerance."""
        return np.abs(twist_diff_per_bp) < tolerance

    def _plot_leftover(self, min_bp,left_over):
        """Plotting the left-over rotational differences"""
        plt.plot(np.arange(min_bp, min_bp + len(left_over)), np.abs(left_over))
        plt.xlabel('Number of Base Pairs')
        plt.ylabel('Absolute Left Over')
        plt.show()

    def _find_optimal_bps(self, positions, bp_per_turn, rise, bp_range, rotation_difference, tolerance, plot=False):
        """Finds optimal base pairs that satisfy the given tolerance.

        Args:
        positions: The positions of base pairs.
        bp_per_turn: Base pairs per turn.
        rise: Component of arc length.
        bp_range: Range of base pairs to consider.
        rotation_difference: The target rotation difference.
        tolerance: The tolerance for accepting an optimal base pair number.
        plot: If True, plots the left-over rotational differences.

        Returns:
        A list of dictionaries containing optimal base pair numbers and their twist differences per base pair.
        """
        min_arc = np.linalg.norm(positions[0] - positions[-1])
        min_bp = int(np.ceil(min_arc / rise))
        left_over = self._compute_left_over(bp_range, min_bp, bp_per_turn, rotation_difference)
        
        if plot:
            self._plot_leftover(min_bp,left_over)

        minima = self._find_minima(np.abs(left_over))
        results = []

        for min_val in minima:
            optimal_bp = min_bp + min_val
            twist_diff_per_bp = self._compute_twist_diff_per_bp(optimal_bp, left_over, min_bp)
            if self._check_within_tolerance(twist_diff_per_bp, tolerance):
                results.append({
                    'optimal_bp': optimal_bp ,
                    'twist_diff_per_bp': np.round(twist_diff_per_bp, 3)
                })
        if len(results) > 0:
            for result in results[:1]:
                print(f'Optimal BP: {result["optimal_bp"]}, Twist Difference per BP: {result["twist_diff_per_bp"]} degrees')
        else:
            print("No optimal number of base pairs found within the specified tolerance.")
        return results


            
class Extend:
    """Extend the DNA sequence in the specified direction using the five_end or three_end as reference."""

    def __init__(self, nucleic, n_bp: int, sequence: str = None, fixed_endpoints: bool = False, frame : int = -1, forward: bool = True, shape: np.ndarray = None, margin : int = 1):
        """Initialize the DNA sequence extender"""
        self.__dict__.update(nucleic.__dict__)

        # Check the input sequence and number of base pairs
        self._n_bp = n_bp # Number of base pairs to extend the DNA sequence
        self._sequence = sequence # DNA sequence to extend the DNA structure
        self._frames = self.frames[:,frame,:,:] # Reference frames of the DNA structure

        # Add other parameters to the instance
        self.fixed_endpoints = fixed_endpoints
        self.forward = forward
        self.shape = shape
        self.frame = frame
        self.margin = margin

        # Get direction of extension
        self.start, self.direction = self.get_start()

        # Compute lengtht of the extension and target position
        length = self._n_bp * 0.34 # Length of a base pair in nm
        target_position = self.start + length * self.direction

         # Interpolate control points for the new spline
        if self.forward:
            control_points = self.interplotate_points(self.start, target_position, self._n_bp)
        else:
            control_points = self.interplotate_points(target_position, self.start, self._n_bp)

        # Create a new spline with the interpolated control points
        spline = SplineFrames(control_points, frame_spacing=0.34)

        if self.forward:
            # fix the strand A except the margin at the end
            fixed = list(range(self._frames.shape[0]-self.margin))
        else:
            # fix the strand A but shift the fixed indices to the end
            fixed = list(range(self.frames.shape[0]))
            fixed = [i + self._n_bp for i in fixed][self.margin:]
        self.fixed = fixed

        # Update the sequence
        if self.forward:
            new_sequence = self.sequence + self._sequence
        else:
            new_sequence = self._sequence + self.sequence
        
        # Combine splines A and the new extension spline 
        if forward:
            spline.frames = np.concatenate([self._frames[:-1],spline.frames])
        else:
            spline.frames = np.concatenate([spline.frames,self._frames[1:]])

        self.nuc = Nucleic(sequence=new_sequence, frames=spline.frames)
     

    def get_start(self):           
        if self.forward:
            return self._frames[-1][0], self._frames[-1][-1]
        else:
            # Reverse the direction of the normal base plane vector of the start last frame
            return self._frames[0][0], -self._frames[0][-1]

    def interplotate_points(self,start, end, n):
        return np.array([start + (end-start)*i/n for i in range(n+1)])

class Minimizer:
    """Minimize the DNA structure using Monte Carlo simulations with pmcpy"""

    def __init__(self, nucleic):
        # Dynamically set attributes from the nucleic instance
        self.__dict__.update(nucleic.__dict__)

        # Check if the required import is available
        if not self._check_import():
            raise ImportError("Run class from pmcpy.run.run is not available.")

    def _check_import(self):
        """Check if the required import is available"""
        try:
            from pmcpy.run.run import Run
            self.Run = Run  # Store the imported class in the instance
            return True
        except ImportError as e:
            print(f"ImportError: {e}")
            return False

    def _initialize_mc_engine(self):
        """Initialize the Monte Carlo engine"""
        # Get the positions and triads of the current frame
        pos = self.frames[:,self.frame,0,:]
        triads = self.frames[:,self.frame,1:,:].transpose(0,2,1) # flip row vectors to column vectors

        # Initialize the Monte Carlo engine
        mc = self.Run(triads=triads,positions=pos,
                        sequence=self.sequence,
                        closed=self.circular,
                        endpoints_fixed=self.endpoints_fixed,
                        fixed=self.fixed,
                        temp=self.temperature,
                        exvol_rad=self.exvol_rad)
        return  mc

    def _update_frames(self):
        """Update the reference frames with the new positions and triads"""
        # update the spline with new positions and triads
        self.frames[:,self.frame,0,:] = self.out['positions'] # set the origins of the frames
        self.frames[:,self.frame,1:,:] = self.out['triads'].transpose(0,2,1) # set the triads of the frames as row vectors
        
    def _get_positions_and_triads(self):
        """Get the positions and triads from the output"""
        # get the positions and triads of the simulation
        positions = self.out['confs'][:,:,:3,3] 
        triads = self.out['confs'][:,:,:3,:3]

        # get the last frames of the simulation
        self.out['triads'] = triads[-1]
        self.out['positions'] = positions[-1]
        return positions, triads.transpose(0,1,3,2) # flip column vectors to row vectors

    def minimize(self,  frame: int = -1, exvol_rad : float = 2.0, temperature : int = 300,  simple : bool = False, equilibrate_writhe : bool = False, endpoints_fixed : bool = True, fixed : List[int] = [], dump_every : int = 1):
        """Minimize the DNA structure."""
        # Set the parameters
        self.endpoints_fixed = endpoints_fixed
        self.fixed = fixed
        self.exvol_rad = exvol_rad
        self.temperature = temperature
        self.frame = frame
        print('Minimize the DNA structure:\nsimple equilibration =', simple, '\nequilibrate writhe =', equilibrate_writhe, '\nexcluded volume radius =', exvol_rad, '\ntemperature =', temperature)
        minimizer = self._initialize_mc_engine()    

        # Run the Monte Carlo simulation
        if equilibrate_writhe:
            self.out = minimizer.equilibrate_simple(equilibrate_writhe=equilibrate_writhe,dump_every=dump_every)
        elif not simple and equilibrate_writhe:
            raise ValueError("Minimization of writhe is only supported for simple equilibration.")
        else:
            self.out = minimizer.equilibrate(dump_every=dump_every,plot_equi=True)

        # Update the reference frames
        positions, triads = self._get_positions_and_triads()
        self._update_frames()

    def get_MC_traj(self):
        """Get the MC sampling energy minimization trajectory of the new spline."""
        # Get the xyz coordinates of the new spline
        xyz = self.out['confs'][:, :, :3, 3]
        
        # Get the triads and calculate the major vector positions
        triads = self.out['confs'][:, :, :3, :3].transpose(0, 1, 3, 2)
        major_vector = triads[:, :, 0, :]
        major_positions = xyz + major_vector*1.1  # Scale the major vector by 1.1
        
        # Concatenate the original xyz and the major_positions
        all_positions = np.concatenate((xyz, major_positions), axis=1)

        # Create a topology for the new spline
        topology = md.Topology()
        # Add a chain to the topology
        chain = topology.add_chain()
        # Add argon atoms to the topology
        num_atoms = xyz.shape[1]
        for i in range(num_atoms):
            residue = topology.add_residue(name='Ar', chain=chain)
            topology.add_atom('Ar', element=md.element.argon, residue=residue)

        # Add dummy atoms to the topology
        for i in range(num_atoms):
            residue = topology.add_residue(name='DUM', chain=chain)
            topology.add_atom('DUM', element=md.element.helium, residue=residue)  # Using helium as a placeholder element

        # Add bonds to the topology
        for i in range(num_atoms - 1):
            topology.add_bond(topology.atom(i), topology.atom(i + 1))

        # Add bonds between each atom and its corresponding dummy atom
        for i in range(num_atoms):
            topology.add_bond(topology.atom(i), topology.atom(num_atoms + i))

        if self.circular:
            # Add a bond between the first and last atom
            topology.add_bond(topology.atom(0), topology.atom(num_atoms - 1))

        # Create a trajectory from the all_positions coordinates and the topology
        traj = md.Trajectory(all_positions, topology=topology)
        
        return traj

    def run(self, cycles: int, dump_every: int = 0, start_id: int = 0) -> np.ndarray:
        """Run the Monte Carlo simulation"""
        raise NotImplementedError("This method is not implemented yet.")