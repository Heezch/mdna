from .nucleic import (
        load,
        make,
        connect,
        compute_rigid_parameters,
        compute_curvature,
        compute_linking_number,
        compute_groove_width,
        sequence_to_pdb,
        sequence_to_md
    )
from .utils import Shapes, get_mutations
from .geometry import ReferenceBase
from .analysis import TorsionAnalysis, GrooveAnalysis

__all__ = ["load", "make", "connect", "compute_rigid_parameters", "compute_curvature", "compute_linking_number", "compute_groove_width", "sequence_to_pdb", "sequence_to_md", "Shapes", "ReferenceBase", "get_mutations", "TorsionAnalysis", "GrooveAnalysis"]

__version__ = "0.00"


# import numpy as np
# import mdtraj as md
# from .nucleic import Nucleic
# from .utils import _check_input, Shapes
# from .spline import SplineFrames
# from .geometry import NucleicFrames
# from .build import Connector

# def load(traj=None, frames=None, sequence=None, chainids=[0,1], circular=None):
#     """Load DNA representation from:
#         - base step mean reference frames/spline frames
#         - or MDtraj trajectory
#     Args:
#         frames: np.array 
#             base step mean reference frames of shape (n_bp, n_timesteps, 4, 3)
#             or (n_bp, 4, 3)
#         traj: object
#             mdtraj trajectory
#         sequence: str
#             DNA sequence corresponding to the frames
#         chainids: list
#             chain ids of the DNA structure
#         circular: bool
#             is the DNA structure circular, optional
#     Returns:
#         Nucleic object
#     """
#     return Nucleic(sequence=sequence, n_bp=None, traj=traj, frames=frames, chainids=chainids, circular=None)

# def make(sequence: str = None, control_points: np.ndarray = None, circular : bool = False, closed: bool = False, n_bp : int = None, dLk : int = None):
#     """Generate DNA structure from sequence and control points
#     Args:
#         sequence: (optional) DNA sequence
#         control_points: (optional) control points of shape (n,3) with n > 3, default is a straight line, see mdna.Shapes for more geometries
#         circular: (default False) is the DNA structure circular, optinional
#         closed: (default False) is the DNA structure circular
#         n_bp: (optinal) number of base pairs to scale shape with
#         dLk: (optinal) Change in twist in terms of Linking number of DNA structure to output
#     Returns:
#         Nucleic object"""

#     # Check if control points are provided otherwise generate a straight line
#     if control_points is not None:
#         if  len(control_points) < 4:
#             raise ValueError('Control points should contain at least 4 points [x,y,z]')
#         elif len(control_points) > 4 and n_bp is None:
#             n_bp = len(control_points) # Number of base pairs
#     elif control_points is None and circular:
#         control_points = Shapes.circle(radius=1)
#         closed = True
#     else:
#         # Linear strand of control points
#         control_points = Shapes.line(length=1)
    
#     sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)
#     spline = SplineFrames(control_points=control_points, n_bp=n_bp, closed=closed, dLk=dLk)
#     #generator = StructureGenerator(sequence=sequence, spline=spline, circular=closed)
#     #frames = generator.frames

#     return Nucleic(sequence=sequence, n_bp=n_bp, frames=spline.frames, chainids=[0,1],circular=circular)

# def connect(Nucleic0, Nucleic1, sequence : str = None, n_bp : int =  None, leader: int = 0, frame : int = -1, margin : int = 1, minimize : bool = True, exvol_rad : float = 0.0, temperature : int = 300):

#     """Connect two DNA structures by creating a new DNA structure with a connecting DNA strand. 
#     The 3' end of the first DNA structure is connected to the 5' end of the second DNA structure.
#     To connect the two strands we interpolate a straight line between the two ends,
#     and distribute the optiminal number of base pairs that result in a neutral twist.

#     Note:
#     The minimization does not use excluded volume interactions by default. 
#     This is because the excluded volume interactions require the EV beads to have no overlap. 
#     However, how the initial configuration is generated, the EV beads are likely have overlap.

#     If one desires the resulting Nucleic object can be minimized once more with the excluded volume interactions.

#     Args:
#         Nucleic0: Nucleic object
#             First DNA structure to connect
#         Nucleic1: Nucleic object
#             Second DNA structure to connect
#         sequence: str, optional
#             DNA sequence of the connecting DNA strand, by default None
#         n_bp: int, optional
#             Number of base pairs of the connecting DNA strand, by default None
#         leader: int, optional
#             The leader of the DNA structure to connect, by default 0
#         frame: int, optional
#             The time frame to connect, by default -1
#         margin: int, optional
#             Number of base pairs to fix at the end, by default 1
#         minimize : bool, optional
#             Whether to minimize the new DNA structure, by default True

#     Returns:
#         Nucleic object: DNA structure with the two DNA structures connected
#     """

#     if Nucleic0.circular or Nucleic1.circular:
#         raise ValueError('Cannot connect circular DNA structures')
    
#     if sequence is not None and n_bp is None:
#         n_bp = len(sequence)

#     # Connect the two DNA structures
#     connector = Connector(Nucleic0, Nucleic1, sequence=sequence, n_bp=n_bp, leader=leader, frame=frame, margin=margin)
#     if minimize:
#         connector.connected_nuc.minimize(exvol_rad=exvol_rad, temperature=temperature, fixed=connector.fixed)

#     return connector.connected_nuc
    

# def compute_rigid_parameters(traj, chainids=[0,1]):
#     """Compute the rigid base parameters of the DNA structure
#     Args:
#         traj: trajectory
#         chainids: chain ids
#     Returns:
#         rigid base object"""

#     return NucleicFrames(traj, chainids)

# def compute_curvature(traj, chainids=[0,1]):
#     """Compute the curvature of the DNA structure"""
#     raise NotImplementedError

# def compute_linking_number(traj, chainids=[0,1]):
#     """Compute the linking number of the DNA structure"""
#     raise NotImplementedError

# def compute_groove_width(traj, chainids=[0,1]):
#     """Compute the groove width of the DNA structure"""
#     raise NotImplementedError


# def sequence_to_pdb(sequence='CGCGAATTCGCG', filename='my_dna', save=True, output='GROMACS',shape=None,n_bp=None,circular=False,dLk=None):
#     """_summary_

#     Parameters
#     ----------
#     sequence : str, optional
#         DNA sequence code, by default 'CGCGAATTCGCG'
#     filename : str, optional
#         filename for pdb output, by default 'my_dna'
#     save : bool, optional
#         boolean to save pdb or not, by default True
#     output : str, optional
#         Type of pdb DNA format, by default 'GROMACS'
#     shape : ndarray, optional
#         control_points of shape (n,3) with n > 3 that is used for spline interpolation to determine DNA shape, by default None which is a straight line
#     n_bp : _type_, optional
#         Number of base pairs to scale shape with, by default None then sequence is used to determine n_bp
#     circular : bool, optional
#         Key that tells if structure is circular/closed, by default False
#     dLk : int, optional
#         Change in twist in terms of Linking number of DNA structure to output, by default None (neutral twist base on bp_per_turn = 10.5)

#     Returns
#     -------
#     MDtraj object
#         returns MDtraj trajectory object of DNA structure (containing only a single frame)
#     """

#     # TODO update with make function
#     sequence, n_bp = _check_input(sequence=sequence, n_bp=n_bp)

#     # Linear strand of control points 
#     if shape is None:
#         shape = Shapes.line(length=1)
#         # Convert the control points to a spline
#     spline = SplineFrames(control_points=shape, n_bp=n_bp,closed=circular,dLk=dLk)
#     # Generate the DNA structure
#     generator = StructureGenerator(sequence=sequence,spline=spline, circular=circular)

#     # Edit the DNA structure to make it compatible with the AMBER force field
#     traj = generator.traj
#     if output == 'GROMACS':
#         phosphor_termini = traj.top.select(f'name P OP1 OP2 and resid 0 {traj.top.chain(0).n_residues}')
#         all_atoms = traj.top.select('all')
#         traj = traj.atom_slice([at for at in all_atoms if at not in phosphor_termini])

#     # Save the DNA structure as pdb file
#     if save:
#         traj.save(f'./{filename}.pdb')

#     return traj

# def sequence_to_md(sequence=None, time=10, time_unit='picoseconds',temperature=310, solvated=False,  filename='my_dna', save=True, output='GROMACS',shape=None,n_bp=None,circular=False,dLk=None):
#     """Simulate DNA sequence using OpenMM
#         Args:
#             sequence: DNA sequence
#             time: simulation time
#             time_unit: time unit
#             temperature: temperature
#             solvated: solvate DNA
#         Returns:
#             mdtraj trajectory"""

#     # TODO update with make function
#     try:
#         import openmm as mm
#         import openmm.app as app
#         import openmm.unit as unit
#         from mdtraj.reporters import HDF5Reporter
#         import mdtraj as md
#         openmm_available = True
#     except ImportError:
#         openmm_available = False
#         print("Openmm is not installed. You shall not pass.")

#     pdb = sequence_to_pdb(sequence=sequence, filename='my_dna', save=True, output='GROMACS',shape=None,n_bp=None,circular=False,dLk=None)
    
#     if not openmm_available:
#         print('But here is your DNA structure')
#         return pdb
#     else:
#         if time_unit == 'picoseconds':
#             time_unit = time * unit.picoseconds
#         elif time_unit == 'nanoseconds':
#             time_unit = time * unit.nanoseconds

#         time = time * time_unit
#         time_step = 2 * unit.femtoseconds
#         temperature = 310 *unit.kelvin
#         steps = int(time/time_step)

#         print(f'Initialize DNA openMM simulation at {temperature._value} K for', time, 'time units')
#         topology = pdb.topology.to_openmm()
#         modeller = app.Modeller(topology, pdb.xyz[0])

#         forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
#         modeller.addHydrogens(forcefield)
#         if solvated:
#             print('Solvate DNA with padding of 1.0 nm and 0.1 M KCl')
#             modeller.addSolvent(forcefield, padding=1.0*unit.nanometers, ionicStrength=0.1*unit.molar, positiveIon='K+')

#         system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.CutoffNonPeriodic)
#         integrator = mm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, time_step)

#         simulation = app.Simulation(modeller.topology, system, integrator)
#         simulation.context.setPositions(modeller.positions)
#         simulation.reporters.append(HDF5Reporter(f'./{sequence}'+'.h5', 100))
#         simulation.reporters.append(app.StateDataReporter(f'./output_{sequence}.csv', 100, step=True, potentialEnergy=True, temperature=True,speed=True))
        
#         print('Minimize energy')
#         simulation.minimizeEnergy()
        
#         print('Run simulation for', steps, 'steps')
#         simulation.step(steps)
#         simulation.reporters[0].close()
#         print('Simulation completed')
#         print('Saved trajectory as:', f'./{sequence}'+'.h5')
#         traj = md.load_hdf5(f'./{sequence}'+'.h5')
#         return traj

