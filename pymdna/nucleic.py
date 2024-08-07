import numpy as np
import matplotlib.pyplot as plt
from typing import List
import mdtraj as md

from .utils import Shapes, get_sequence_letters, _check_input
from .spline import SplineFrames, Twister
from .geometry import ReferenceBase, NucleicFrames, NucleicFrames_quaternion
from .generators import SequenceGenerator, StructureGenerator
from .modifications import Mutate, Hoogsteen, Methylate
from .analysis import GrooveAnalysis, TorsionAnalysis, ContactCount
from .build import Minimizer, Extender, Connector



def load(traj=None, frames=None, sequence=None, chainids=[0,1], circular=None):
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
        circular: bool
            is the DNA structure circular, optional
    Returns:
        Nucleic object
    """
    return Nucleic(sequence=sequence, n_bp=None, traj=traj, frames=frames, chainids=chainids, circular=None)

def make(sequence: str = None, control_points: np.ndarray = None, circular : bool = False, closed: bool = False, n_bp : int = None, dLk : int = None):
    """Generate DNA structure from sequence and control points
    Args:
        sequence: (optional) DNA sequence
        control_points: (optional) control points of shape (n,3) with n > 3, default is a straight line, see mdna.Shapes for more geometries
        circular: (default False) is the DNA structure circular, optinional
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
    elif control_points is None and circular:
        control_points = Shapes.circle(radius=1)
        closed = True
    else:
        # Linear strand of control points
        control_points = Shapes.line(length=1)
    
    sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)
    spline = SplineFrames(control_points=control_points, n_bp=n_bp, closed=closed, dLk=dLk)
    #generator = StructureGenerator(sequence=sequence, spline=spline, circular=closed)
    #frames = generator.frames

    return Nucleic(sequence=sequence, n_bp=n_bp, frames=spline.frames, chainids=[0,1],circular=circular)

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
    connector = Connector(Nucleic0, Nucleic1, sequence=sequence, n_bp=n_bp, leader=leader, frame=frame, margin=margin)
    if minimize:
        connector.connected_nuc.minimize(exvol_rad=exvol_rad, temperature=temperature, fixed=connector.fixed)

    return connector.connected_nuc

def compute_rigid_parameters(traj, chainids=[0,1]):
    """Compute the rigid base parameters of the DNA structure
    Args:
        traj: trajectory
        chainids: chain ids
    Returns:
        rigid base object"""

    return NucleicFrames(traj, chainids)

def compute_curvature(traj, chainids=[0,1]):
    """Compute the curvature of the DNA structure"""
    raise NotImplementedError

def compute_linking_number(traj, chainids=[0,1]):
    """Compute the linking number of the DNA structure"""
    raise NotImplementedError

def compute_groove_width(traj, chainids=[0,1]):
    """Compute the groove width of the DNA structure"""
    raise NotImplementedError

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

    def __init__(self, sequence=None, n_bp=None, traj=None, frames=None, chainids=None, circular=None):
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
        self.circular = self._is_circular() if circular is None else circular 
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
        self.traj = None
        generator = StructureGenerator(frames=self.frames[:,frame,:,:], sequence=self.sequence, circular=self.circular)
        self.traj = generator.get_traj()
    
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

        if self.traj.n_atoms > 99999:
            print('Warning: Trajectory contains more than 99999 atoms, consider saving as .h5')
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
        return distance < 1 and self.frames.shape[0] > 20 

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

    def minimize(self, frame: int = -1, exvol_rad : float = 2.0, temperature : int = 300,  simple : bool = False, equilibrate_writhe : bool = False, endpoints_fixed : bool = False, fixed : List[int] = [], dump_every : int = 20):
        """
        Minimize the DNA structure.

        Args:
            frame (int): The trajectory frame to minimize. Defaults to -1.
            simple (bool): Whether to use simple equilibration. Defaults to False.
            equilibrate_writhe (bool): Whether to equilibrate writhe. Defaults to False. Only works for simple equilibration.
            endpoints_fixed (bool): Whether the endpoints are fixed. Defaults to False.
            fixed (list): List of fixed base pairs. Defaults to an empty list.
            exvol_rad (float): Excluded volume radius. Defaults to 2.0.
            temperature (int): Temperature for equilibration. Defaults to 300.
            dump_every (int): Frequency of dumping frames. Defaults to 20.

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

        Options for nucleobases are:
            Canoncial bases: A, T, G, C, U
            Hachimoji: B [A_ana], S [T_ana], P [C_ana], Z [G_ana],  DOI: 10.1126/science.aat0971
            Fluorescent: 2-aminopurine 2AP (E), triC (D), https://doi.org/10.1002/anie.201001312 (2kv0) tricyclic cytosin base analogue (1tuq)
            Hydrophobic pairs: d5SICS (L), dNaM (M)

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
        if len(methylations) == 0:
            raise ValueError('Provide a methylation list')

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
            Number of base pairs to fix at the end/start, by default 1
        """

        # TODO: add target frame to extend to

        if self.circular:
            raise ValueError('Cannot extend circular DNA structure')
        if not n_bp and not fixed_endpoints:
            raise ValueError("Either a fixed endpoint or a length must be specified for extension.")    
        if self.traj is None:
            self._frames_to_traj()
        if shape is None:
            shape = Shapes.line(length=1)
        if self.frames is None:
            self._traj_to_frames()

        # Check the input sequence and number of base pairs
        sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)

        extender = Extender(self, n_bp=n_bp, sequence=sequence, fixed_endpoints=fixed_endpoints, frame=frame, forward=forward, shape=shape, margin=margin)
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

    def get_linking_number(self, frame : int = -1):
        """Get the linking number of the DNA structure based on Gauss's linking number theorem.

        Parameters
        ----------
        frame : int, optional
            Time frame of trajectory, by default -1

        Returns
        -------
        np.ndarray
            Numpy array containing the linking number, writhe, and twist corresponding to the time frame
        """

        from pmcpy import pylk

        if self.frames is None:
                self._traj_to_frames()
        frames = self.frames[:,frame,:,:]
        positions = frames[:,0]
        triads = frames[:,1:].transpose(0,2,1) # Flip row vectors to columns

        writhe = pylk.writhe(positions)
        lk = pylk.triads2link(positions, triads)
        return np.array([lk, writhe, lk - writhe])


            