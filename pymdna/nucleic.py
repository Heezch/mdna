import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
import mdtraj as md
from scipy.spatial.transform import Rotation as R

from .utils import Shapes, get_sequence_letters, _check_input
from .spline import SplineFrames, Twister
from .geometry import ReferenceBase, NucleicFrames
from .generators import SequenceGenerator, StructureGenerator
from .modify import Mutate, Hoogsteen, Methylate
# from .analysis import GrooveAnalysis, TorsionAnalysis, ContactCount
from .minimizer import Minimizer
# from .build import Extender, Connector



def load(traj=None, frames=None, sequence=None, chainids=[0,1], circular=None, filename=None, top=None, stride=None):
    """Load DNA representation from either base step mean reference frames/spline frames or an MDtraj trajectory.

    Args:
        traj (object, optional): MDtraj trajectory containing the DNA structure. If provided, the frames and sequence arguments are ignored. (default: None)
        frames (np.array, optional): Base step mean reference frames of shape (n_bp, n_timesteps, 4, 3) or (n_bp, 4, 3). If provided, the traj and sequence arguments are ignored. (default: None)
        sequence (str, optional): DNA sequence. If provided, the traj and frames arguments are ignored. (default: None)
        chainids (list, optional): Chain IDs of the DNA structure. (default: [0,1])
        circular (bool, optional): Flag indicating if the DNA structure is circular/closed. If not provided, it will be determined based on the input data. (default: None)
        filename (str, optional): The filename or filenames of the trajectory. If provided, the traj and frames arguments are ignored. (default: None)
        top (str, optional): The topology file of the trajectory. (default: None)
        stride (int, optional): The stride of the trajectory. (default: None)

    Returns:
        Nucleic (object): DNA structure object.

    Notes:
        - The `traj` argument is prioritized over frames and sequence.
        - If the `filename_or_filenames` argument is provided, the other arguments are ignored, except for the `top` and `stride` arguments and `chainids`.

    Example:
        Load a DNA structure from a trajectory
        ```python
        traj = md.load('dna.pdb')
        dna = mdna.load(traj=traj, chainids=[0, 1])
        ```
    """
    # Load the trajectory directly using MDtraj from a file
    if filename is not None and top is None:
        traj = md.load(filename_or_filenames=filename, stride=stride)
    elif filename is not None and top is not None:
        traj = md.load(filename_or_filenames=filename, top=top, stride=stride)

    return Nucleic(sequence=sequence, n_bp=None, traj=traj, frames=frames, chainids=chainids, circular=None)

def make(sequence: str = None, control_points: np.ndarray = None, circular : bool = False, closed: bool = False, n_bp : int = None, dLk : int = None):
    """Generate a DNA structure from a given DNA sequence and control points.

    Args:
        sequence (str, optional): DNA sequence code. If not provided, the default sequence 'CGCGAATTCGCG' will be used. (default: None)
        control_points (ndarray, optional): Control points of the DNA structure. Should be a numpy array of shape (n, 3) where n is the number of control points. If not provided, a straight line will be used as the default control points. (default: None)
        circular (bool, optional): Flag indicating if the DNA structure is circular/closed. If True, the DNA structure will be closed. If False, the DNA structure will be open. (default: False)
        closed (bool, optional): Flag indicating if the DNA structure is closed. If True, the DNA structure will be closed. If False, the DNA structure will be open. This argument is deprecated and will be removed in a future version. Please use the 'circular' argument instead. (default: False)
        n_bp (int, optional): Number of base pairs to scale the shape with. If not provided, the number of base pairs will be determined based on the length of the control points or the sequence. (default: None)
        dLk (int, optional): Change in twist in terms of Linking number of the DNA structure. If not provided, a neutral twist based on bp_per_turn = 10.5 will be used. (default: None)

    Returns:
        Nucleic (object): DNA structure object.

    Example:
        Generate a DNA structure from a sequence
        ```python
        dna = make(sequence='CGCGAATTCGCG', control_points=None, circular=False, closed=False, n_bp=None, dLk=None)
        ```
    """

    # Check if control points are provided, otherwise generate a straight line
    if control_points is not None:
        if len(control_points) < 4:
            raise ValueError('Control points should contain at least 4 points [x, y, z]')
        elif len(control_points) > 4 and n_bp is None and sequence is None:
            n_bp = len(control_points)  # Number of base pairs
    elif control_points is None and circular:
        control_points = Shapes.circle(radius=1)
        closed = True
    else:
        # Linear strand of control points
        control_points = Shapes.line(length=1)
    
    sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)
    spline = SplineFrames(control_points=control_points, n_bp=n_bp, closed=circular, dLk=dLk)

    return Nucleic(sequence=sequence, n_bp=n_bp, frames=spline.frames, chainids=[0, 1], circular=circular)

def connect(Nucleic0, Nucleic1, sequence: Union[str|List] = None, n_bp: int = None, leader: int = 0, frame: int = -1, margin: int = 1, minimize: bool = True, exvol_rad: float = 0.0, temperature: int = 300):
    """Connect two DNA structures by creating a new DNA structure with a connecting DNA strand.

    The 3' end of the first DNA structure is connected to the 5' end of the second DNA structure.
    To connect the two strands, a straight line is interpolated between the two ends,
    and the optimal number of base pairs is distributed to achieve a neutral twist.

    Args:
        Nucleic0 (Nucleic): First DNA structure to connect.
        Nucleic1 (Nucleic): Second DNA structure to connect.
        sequence (str or List, optional): DNA sequence of the connecting DNA strand. Default is None.
        n_bp (int, optional): Number of base pairs of the connecting DNA strand. Default is None.
        leader (int, optional): The leader of the DNA structure to connect. Default is 0.
        frame (int, optional): The time frame to connect. Default is -1.
        margin (int, optional): Number of base pairs to fix at the end. Default is 1.
        minimize (bool, optional): Whether to minimize the new DNA structure. Default is True.
        exvol_rad (float, optional): Radius for excluded volume interactions during minimization. Default is 0.0.
        temperature (int, optional): Temperature for minimization. Default is 300.

    Returns:
        Nucleic (object): DNA structure with the two DNA structures connected.

    Raises:
        ValueError: If either of the DNA structures is circular.

    Notes:
        - The minimization does not use excluded volume interactions by default.This is because the excluded volume interactions require the EV beads to have no overlap. However, in the initial configuration, the EV beads are likely to have overlap. If desired, the resulting Nucleic object can be further minimized with the excluded volume interactions.

    Example:
        Connect two DNA structures
        ```python
        dna = connect(Nucleic0, Nucleic1, margin=5)
        ```
    """
    if Nucleic0.circular or Nucleic1.circular:
        raise ValueError('Cannot connect circular DNA structures')

    if (sequence is not None and n_bp is None) or (sequence is None and n_bp is not None) or (sequence is not None and n_bp is not None):
        sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)
   
    # Connect the two DNA structures
    connector = Connector(Nucleic0, Nucleic1, sequence=sequence, n_bp=n_bp, leader=leader, frame=frame, margin=margin)
    if minimize:
        connector.connected_nuc.minimize(exvol_rad=exvol_rad, temperature=temperature, fixed=connector.fixed)
    return connector.connected_nuc

def compute_rigid_parameters(traj, chainids=[0,1]):
    """Compute the rigid base parameters of the DNA structure.

    Args:
        traj (object): MDtraj trajectory containing the DNA structure.
        chainids (list, optional): List of chain IDs of the DNA structure. Default is [0, 1].

    Returns:
        NucleicFrames (object): Object representing the rigid base parameters of the DNA structure.

    Raises:
        ValueError: If the traj argument is not provided.

    Notes:
        - The returned NucleicFrames object contains information about the rigid base parameters of the DNA structure, such as the positions and orientations of the base steps.

    Example:
        Compute the rigid base parameters of a DNA structure
        ```python
        traj = md.load('dna.pdb')
        rigid_params = mdna.compute_rigid_parameters(traj, chainids=[0, 1])
        ````
    """
    if traj is None:
        raise ValueError("The traj argument must be provided.")
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

def sequence_to_pdb(sequence: str = 'CGCGAATTCGCG', filename: str = 'my_dna', save: bool = True, output: str = 'GROMACS', shape: np.ndarray = None, n_bp: int = None, circular: bool = False, dLk: int = None) -> md.Trajectory:
    """Generate a DNA structure from a DNA sequence code.

    Args:
        sequence (str, optional): The DNA sequence code. Default is 'CGCGAATTCGCG'.
        filename (str, optional): The filename for the pdb output. Default is 'my_dna'.
        save (bool, optional): Whether to save the pdb file. Default is True.
        output (str, optional): The type of pdb DNA format. Default is 'GROMACS'.
        shape (np.ndarray, optional): Control points of shape (n,3) with n > 3 that is used for spline interpolation to determine DNA shape. Default is None, which is a straight line.
        n_bp (int, optional): Number of base pairs to scale shape with. Default is None, then the sequence is used to determine n_bp.
        circular (bool, optional): Indicates if the structure is circular/closed. Default is False.
        dLk (int, optional): Change in twist in terms of Linking number of DNA structure to output. Default is None, which corresponds to a neutral twist based on bp_per_turn = 10.5.

    Returns:
        md.Trajectory: An MDtraj trajectory object of the DNA structure (containing only a single frame).

    Raises:
        ValueError: If the sequence is not provided.

    Notes:
        - The pdb file is saved in the current directory with the specified filename.

    Example:
        Generate a DNA structure from a sequence
        ```python
        traj = mdna.sequence_to_pdb(sequence='CGCGAATTCGCG', filename='my_dna', save=True, output='GROMACS', shape=None, n_bp=None, circular=False, dLk=None)
        ```
    """

    # Check if the sequence is provided
    if sequence is None:
        raise ValueError("The sequence argument must be provided.")

    # TODO: Update with make function
    sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)

    # Linear strand of control points
    if shape is None:
        shape = Shapes.line(length=1)

    # Convert the control points to a spline
    spline = SplineFrames(control_points=shape, n_bp=n_bp, closed=circular, dLk=dLk)

    # Generate the DNA structure
    generator = StructureGenerator(sequence=sequence, spline=spline, circular=circular)

    # Edit the DNA structure to make it compatible with the AMBER force field
    traj = generator.traj
    if output == 'GROMACS':
        phosphor_termini = traj.top.select(f'name P OP1 OP2 and resid 0 {traj.top.chain(0).n_residues}')
        all_atoms = traj.top.select('all')
        traj = traj.atom_slice([at for at in all_atoms if at not in phosphor_termini])

    # Save the DNA structure as a pdb file
    if save:
        traj.save(f'./{filename}.pdb')

    return traj

def sequence_to_md(sequence=None, time=10, time_unit='picoseconds',temperature=310, solvated=False,  filename='my_dna', save=True, output='GROMACS',shape=None,n_bp=None,circular=False,dLk=None):
    """Simulate DNA sequence using OpenMM.

        Args:
            sequence (str): DNA sequence code.
            time (int): Simulation time.
            time_unit (str): Time unit (picoseconds or nanoseconds).
            temperature (int): Temperature in Kelvin.
            solvated (bool): Solvate DNA with water and ions.
            filename (str): Filename for pdb output.
            save (bool): Save the trajectory.
            output (str): Output format for the trajectory (GROMACS or HDF5).
            shape (str): Shape of the DNA structure (linear or circular).
            n_bp (int): Number of base pairs in the DNA structure.
            circular (bool): Flag indicating if the DNA structure is circular.
            dLk (int): Change in linking number of the DNA structure.

        Returns:
            MDTraj (object): MDtraj trajectory object of DNA structure.

        Notes:
            - This function uses the OpenMM library to simulate the behavior of a DNA sequence.
            - The simulation can be performed for a specified time period at a given temperature.
            - The DNA structure can be solvated with water and ions.
            - The trajectory of the simulation can be saved in either GROMACS or HDF5 format.

        Example:
            Simulate a linear DNA structure for 100 picoseconds at 300 K
            ```python
            trajectory = mdna.sequence_to_md(sequence='ATCGATA', time=100, time_unit='picoseconds', temperature=300, shape='linear')
            ```
        """

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
            """Initializes the DNA structure.

            Args:
                sequence (str): The DNA sequence, e.g. 'CGCGAATTCGCG'.
                n_bp (int): The number of base pairs. Default is None.
                traj (object): The MDTraj trajectory. Default is None.
                frames (np.ndarray): The reference frames of the DNA structure. Default is None.
                chainids (list): The chain IDs. Default is None.
                circular (bool): A flag that indicates if the structure is circular/closed. Default is None.

            Raises:
                ValueError: If both traj and frames are provided.
                ValueError: If frames have an invalid shape.
                ValueError: If the number of base pairs in the sequence and frames do not match.
                ValueError: If neither traj nor frames are provided.

            Notes:
                - If traj is provided, sequence and n_bp will be extracted from the trajectory.
                - If frames is provided, n_bp will be determined from the shape of frames.
                - If sequence is provided, it will be checked against the number of base pairs.

            Attributes:
                sequence (str): The DNA sequence.
                n_bp (int): The number of base pairs.
                traj (object): The MDTraj trajectory.
                frames (np.ndarray): The reference frames of the DNA structure.
                chainids (list): The chain IDs.
                circular (bool): A flag that indicates if the structure is circular/closed.
                rigid (None): A container for rigid base parameters class output.
                minimizer (None): A container for minimizer class output.
            """
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
            self.base_pair_map = {'A':'T','T':'A','G':'C','C':'G','U':'A','D':'G','E':'T','L':'M','M':'L','B':'S','S':'B','Z':'P','P':'Z'}

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
            frames (np.ndarray): reference frames of the DNA structure"""
        
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
    
    def get_rigid_object(self):
        """Get the rigid base class object of the DNA structure

        Returns:
            NucleicFrames (object): Object representing the rigid base parameters of the DNA structure."""
        if self.rigid is None and self.traj is not None:
            self.rigid = NucleicFrames(self.traj, self.chainids)
            return self.rigid
        elif self.rigid is None and self.traj is None:
            self._frames_to_traj()
            self.rigid = NucleicFrames(self.traj, self.chainids)
            return self.rigid
        else:
            return self.rigid
        
    def get_parameters(self, step : bool = False, base : bool = False):
        """By default retuns all the parameters of the DNA structure.
        Use arguments to get a specific parameter group of the DNA structure.

        Args:
            step (bool, optional): Returns only the step parameters of consequative bases. Defaults to False.
            base (bool, optional): Returns onlt the base pair parameters of opposing bases. Defaults to False.

        Returns:
            (parameters, names) (tuple) : Returns the names of the computed parameters of shape (n_frames, n_base_pairs, n_parameters)"""

        if self.rigid is None:
            self.get_rigid_object()
        return self.rigid.get_parameters(step=step, base=base)
    
    def get_parameter(self, parameter_name : str):
        """Get a specific parameter from the rigid base parameters class object of the DNA structure
            
        Args:
            parameter_name (str): The name of the parameter to retrieve.

        Notes:
            The following parameters can be retrieved:
            - shift, slide, rise, tilt, roll, twist, shear, stretch, stagger, buckle, propeller, opening

        Returns:
            np.ndarray: The parameter values of the DNA structure."""
        if self.rigid is None:
            self.get_rigid_object()
        return self.rigid.get_parameter(parameter_name)
    
    def get_base_frames(self):
        """Get the base reference frames of the DNA structure
        
        Returns:
            dict: A dictionary containing the base reference frames of the DNA structure. 
              The keys are residue topologies of the MDTraj object (traj.top.residues) and the values are the reference frames in shape (n_frames, 4, 3), 
              where the rows represent the origin, b_D, b_L, and b_N vectors."""

        if self.rigid is None:
            self.get_rigid_object()
        return self.rigid.get_base_reference_frames()

    
    def _is_circular(self, frame=0):
        """Detects if the DNA structure is circular for a given chain and frame.

        Args:
            frame (int, optional): Frame index to check. Default is 0.

        Returns:
            bool: True if the DNA is circular, False otherwise.
        """
        if self.frames is None:
            self._traj_to_frames()
            
        start = self.frames[0, frame, 0]
        end = self.frames[-1, frame, 0]
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
        """Draws a 3D representation of the DNA structure with optional helical axis, backbone, lead, anti, and triads.

        Args:
            ax (object, optional): Matplotlib axis. Default is None.
            fig (object, optional): Figure axis. Default is None.
            save (bool, optional): Save image as png. Default is False.
            frame (int, optional): Index of trajectory to visualize. Default is -1.
            markersize (int, optional): Width of backbone plot. Default is 2.
            lw (int, optional): Line width of plots. Default is 1.
            helical_axis (bool, optional): Plot central axis passing through frame origins. Default is True.
            backbone (bool, optional): Plot backbone as 'o-' line plot through phosphor atoms. Default is True.
            lead (bool, optional): Plot leading strand. Default is False.
            anti (bool, optional): Plot anti-sense opposing leading strand. Default is False.
            triads (bool, optional): Plot triads in order of b_L (blue), b_N (green), b_T (red). Default is False.
            length (float, optional): Length of triad vectors. Default is 0.23.
            color_lead (str, optional): Color of the leading strand. Default is 'k'.
            color_anti (str, optional): Color of the anti strand. Default is 'darkgrey'.

        Notes:
            - The function draws a 3D representation of the DNA structure using matplotlib.
            - The function requires either the trajectory or reference frames to be loaded before calling.

        Example:
            Make a DNA structure and draw the 3D representation
            ```python
            dna = nuc.make(sequence='CGCGAATTCGCG')
            dna.draw()
            ```
        """

        # TODO: handle circular DNA and when trajectory is not loaded make frames uniform 
        # in shape (time/n_frames, n_bp, 4, 3)

        if self.traj is None:
            self._frames_to_traj()
        elif self.frames is None:
            self._traj_to_frames()
                
        if fig is None and ax is None:
            fig = plt.figure()
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

    def minimize(self, frame: int = -1, exvol_rad : float = 2.0, temperature : int = 300,  simple : bool = False, equilibrate_writhe : bool = False, endpoints_fixed : bool = False, fixed : List[int] = [], dump_every : int = 5, plot : bool = False):
        """
        Minimize the DNA structure. This method updates the  of the DNA structure.

        Args:
            frame (int): The trajectory frame to minimize. Defaults to -1.
            simple (bool): Whether to use simple equilibration. Defaults to False.
            equilibrate_writhe (bool): Whether to equilibrate writhe. Defaults to False. Only works for simple equilibration.
            endpoints_fixed (bool): Whether the endpoints are fixed. Defaults to False.
            fixed (list): List of fixed base pairs. Defaults to an empty list.
            exvol_rad (float): Excluded volume radius. Defaults to 2.0.
            temperature (int): Temperature for equilibration. Defaults to 300.
            dump_every (int): Frequency of dumping frames. Defaults to 5.
            plot (bool): Whether to plot the energy. Defaults to False.

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
            Load a DNA structure and minimize it
            ```python
            nuc = mdna.load(traj)
            nuc.minimize(temperature=310, exvol_rad=2.0)
            ```
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

    def mutate(self, mutations: dict = None, complementary: bool = True, frame: int = -1):
        """Mutate the DNA trajectory, updating the topology and coordinates of the DNA structure.
        The method updates the `traj` attribute and the `sequence` attribute of the DNA object.


        Args:
            mutations (dict, optional): A dictionary containing the mutation information. The keys represent the indices of the base pairs to be mutated, and the values represent the new nucleobases. For example, `mutations = {0: 'A', 1: 'T', 2: 'G'}` will mutate the first three base pairs to A, T, and G, respectively. Defaults to None.
            complementary (bool, optional): Whether to mutate the complementary strand. Defaults to True.
            frame (int, optional): The frame to mutate. Defaults to -1.

        Raises:
            ValueError: If no mutation dictionary is provided.

        Notes:
            - Valid nucleobases for mutations include:
                - Canonical bases: A, T, G, C, U
                - Hachimoji: B [A_ana], S [T_ana], P [C_ana], Z [G_ana] (DOI: 10.1126/science.aat0971)
                - Fluorescent: 2-aminopurine 2AP (E), triC (D) (DOI: 10.1002/anie.201001312), tricyclic cytosine base analogue (1tuq)
                - Hydrophobic pairs: d5SICS (L), dNaM (M)
             
        Example:
            Create a DNA object 
            ```python
            dna = DNA()
            mutations = {0: 'A', 1: 'T', 2: 'G'}
            dna.mutate(mutations=mutations, complementary=True, frame=-1)
            ```
        """
        if self.traj is None:
            self._frames_to_traj()
        if mutations is None:
            raise ValueError('Provide a mutation dictionary')

        # TODO: Check if valid letters in mutations dictionary

        mutant = Mutate(self.traj[frame], mutations, complementary=complementary)
        self.traj = mutant.get_traj()
        # Update sequence
        self.sequence = ''.join(get_sequence_letters(self.traj, leading_chain=self.chainids[0]))


    def flip(self, fliplist: list = [], deg: int = 180, frame: int = -1):
            """Flips the nucleobases of the DNA structure.
            The method updates the `traj` attribute of the DNA object.


            Args:
                fliplist (list): A list of base pairs to flip. Defaults to an empty list.
                deg (int): The degrees to flip. Defaults to 180.
                frame (int): The frame to flip. Defaults to -1.

            Raises:
                ValueError: If no fliplist is provided.

            Notes:
                - Rotating the nucleobase by 180 degrees corresponds to the Hoogsteen base pair configuration.

            Example:
                Flip DNA
                ```python
                dna = mdna.make('GCAAAGC)
                dna.flip(fliplist=[3,4], deg=180)
                ```

            """
            
            if self.traj is None:
                self._frames_to_traj()
            if len(fliplist) == 0:
                raise ValueError('Provide a fliplist')

            flipper = Hoogsteen(self.traj, fliplist=fliplist, deg=deg, verbose=True)
            self.traj = flipper.get_traj()

    def methylate(self, methylations: list = [], CpG: bool = False, leading_strand: int = 0, frame: int = -1):
            """Methylate the nucleobases of the DNA structure.
            The method updates the `traj` attribute of the DNA object.


            Args:
                methylations (list): List of base pairs to methylate. Defaults to [].
                CpG (bool): Whether to methylate CpG sites. Defaults to False.
                leading_strand (int): The leading strand to methylate. Defaults to 0.
                frame (int): The frame to methylate. Defaults to -1.

            Raises:
                ValueError: If the DNA structure is not loaded.
                ValueError: If the methylations list is empty.

            Notes:
                Using the `CpG` flag will methylate the CpG sites in the DNA structure. This flag supercedes the methylations list.

            Example:
                Methylate DNA
                ```python
                dna = mdna.make('GCGCGCGAGCGA)
                dna.metyhlate(fliplist=[3,4])
                ```
            """
            if self.traj is None:
                self._frames_to_traj()
            if len(methylations) == 0 and not CpG:
                raise ValueError('Provide a non-empty methylations list')

            methylator = Methylate(self.traj, methylations=methylations, CpG=CpG, leading_strand=leading_strand)
            self.traj = methylator.get_traj()
    
    def extend(self, n_bp: int = None, sequence: Union[str|List] = None, fixed_endpoints: bool = False, forward: bool = True, frame: int = -1, shape: np.ndarray = None, margin: int = 1, minimize: bool = True, plot : bool = False):  
        """Extend the DNA structure in the specified direction.
            The method updates the attributes of the DNA object.


        Args:
            n_bp (int): Number of base pairs to extend the DNA structure. Defaults to None.
            sequence (str or List, optional): DNA sequence to extend the DNA structure. If not provided, the sequence will be generated randomly. Defaults to None.
            fixed_endpoints (bool, optional): Whether to fix the endpoints of the DNA structure during extension. Defaults to False.
            forward (bool, optional): Whether to extend the DNA structure in the forward direction. If False, the DNA structure will be extended in the backward direction. Defaults to True.
            frame (int, optional): The time frame to extend. Defaults to -1.
            shape (np.ndarray, optional): Control points of the shape to be used for extension. The shape should be a numpy array of shape (n, 3), where n is greater than 3. Defaults to None.
            margin (int, optional): Number of base pairs to fix at the end/start of the DNA structure during extension. Defaults to 1.
            minimize (bool, optional): Whether to minimize the new DNA structure after extension. Defaults to True.
            plot (bool, optional): Whether to plot the Energy during minmization. Defaults to False.

        Raises:
            ValueError: If the DNA structure is circular and cannot be extended.
            ValueError: If neither a fixed endpoint nor a length is specified for extension.
            ValueError: If the input sequence is invalid or the number of base pairs is invalid.

        Notes:
            - If the DNA structure is circular, it cannot be extended.

        Example:
            Extend DNA structure
            ```python
            nuc = mdna.make(n_bp=100)
            nuc.extend(n_bp=10, forward=True, margin=2, minimize=True)
            ```
        """
        if self.circular:
            raise ValueError('Cannot extend circular DNA structure')  
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
            self.nuc.minimize(fixed=extender.fixed, endpoints_fixed=fixed_endpoints, plot=plot)

        # Update attributes
        self.sequence = self.nuc.sequence
        self.traj = self.nuc.get_traj()
        self.frames = self.nuc.get_frames()
        self.n_bp = self.nuc.n_bp

    def invert(self):
        """Inverse the direction of the DNA structure so from 5' to 3' to 3' to 5
         The method updates attributes of the DNA object.
         
         Raises:
            NotImplementedError."""
        raise NotImplementedError('Not implemented yet')

    def get_linking_number(self, frame : int = -1):
        """Get the linking number of the DNA structure based on Gauss's linking number theorem.

        Args:
            frame (int, optional): Time frame of trajectory, by default -1

        Returns:
            linking_number (np.ndarray): Numpy array containing the linking number, writhe, and twist corresponding to the time frame
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
    
    def save_pdb(self, filename : str = None, frame : int = -1):
        """Save the DNA structure as a pdb file.

        Args:
            filename (str, optional): Filename to save the pdb file. Defaults to None.
            frame (int, optional): If the trajectory has multiple frames, specify the frame to save. Defaults to -1.
        """

        # check if traj
        if self.traj is None:
            self._frames_to_traj()
        if filename is None:
            filename = 'my_mdna'
        self.traj[frame].save(f'{filename}.pdb')




class Extender:
    """Extend the DNA sequence in the specified direction using the five_end or three_end as reference."""

    def __init__(self, nucleic, n_bp: int, sequence: Union[str | List] = None, fixed_endpoints: bool = False, frame : int = -1, forward: bool = True, shape: np.ndarray = None, margin : int = 1):
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



class Connector:
    def __init__(self, Nucleic0, Nucleic1, sequence : Union[str | List] = None, n_bp : int =  None, leader: int = 0, frame : int = -1, margin : int = 1):
        
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
            print(f'Optimal number of base pairs: {self.n_bp}')
 
        # Guess the shape of the spline C by interpolating the start and end points
        # Note, we add to extra base pairs to account for the double count of the start and end points of the original strands
        control_points_C = self._interplotate_points(self.start[0], self.end[0], self.n_bp+2)# if opti else self.n_bp)
        distance = np.linalg.norm(self.start-self.end)

        # Create frames object with the sequence and shape of spline C while squishing the correct number of BPs in the spline
        spline_C = SplineFrames(control_points=control_points_C, frame_spacing=distance/len(control_points_C),n_bp=self.n_bp+2)
     
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
           



