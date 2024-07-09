from .utils import  *
from .spline import SplineFrames, Twister
from .geometry import ReferenceBase, NucleicFrames, NucleicFrames_quaternion
from .generators import SequenceGenerator, StructureGenerator
from .modifications import Mutate, Hoogsteen, Methylate
from .analysis import GrooveAnalysis, TorsionAnalysis, ContactCount
from .build import Build

import numpy as np

def load(traj=None, frames=None, sequence=None, chainids=[0,1]):
    """Load DNA representation from:
        - mean reference frames/spline frames
        - or trajectory
    Args:
        frames: reference frames
        traj: trajectory
        chainids: chain ids"""

    if traj is not None and frames is None:
        # check if chaind ids correspond to nucleic acids
        sequence = get_sequence_letters(traj,leading_chain=chainids[0])
        n_bp = len(sequence)

    elif frames is not None and sequence is not None and traj is None:
        # check shape of frames
        if frames.shape[1] != len(sequence):
            raise ValueError('Number of base pairs in the sequence and frames do not match')
        n_bp = frames.shape[1]

    elif frames is not None and sequence is None:
        raise ValueError('Provide a sequence along with reference frames.')
    elif traj is not None and frames is not None:
        raise ValueError('Provide either a trajectory or reference frames, not both')
    elif traj is None and frames is None:
        raise ValueError('Provide either a trajectory or reference frames')

    return Nucleic(sequence=sequence, n_bp=n_bp, traj=traj, frames=frames, chainids=chainids)



class Nucleic(NucleicFrames, Mutate, Hoogsteen, Methylate, Build):

        def __init__(self, sequence=None, n_bp=None, traj=None, frames=None, chainids=None):
            """Initialize the DNA structure
            Args:
                sequence: DNA sequence
                n_bp: number of base pairs
                traj: trajectory
                frames: reference frames
                chainids: chain ids"""

            self.sequence, self.n_bp = _check_input(sequence=sequence, n_bp=n_bp)
            self.traj = traj
            self.frames = frames
            self.chainids = chainids


        def frames_to_traj(self):
            """Convert reference frames to trajectory"""
            if not hasattr(self, 'frames'):
                raise ValueError('Load reference frames first')
            self.traj = StructureGenerator(frames=self.frames, sequence=self.sequence).get_traj()

        def get_traj(self):
            """Get the trajectory"""
            if not hasattr(self, 'traj'):
                self.frames_to_traj()
            return self.traj
        
        # def compute_rigid_parameters(self):
        #     if not hasattr(self, 'traj'):
        #         raise ValueError('Load or generate a trajectory first')
        #     self.rigid = NucleicFrames(self.traj, self.chainids)
        
        def get_frames(self):
            """Get the reference frames of the DNA structure belonging to the base steps:
            Returns: array of reference frames of shape (n_frames, n_bp, 4, 3)
            where n_frames is the number of frames, n_bp is the number of base pairs, 
            and 4 corresponds to the origin and the 3 vectors of the reference frame"""

            if not hasattr(self, 'rigid'):
                self.get_rigid_parameters()
            return self.rigid.mean_reference_frames

        def compute_torsions(self):
            pass
        def compute_grooves(self):
            pass



def _check_input(sequence=None, n_bp=None):

    if sequence is None and n_bp is not None:
        sequence = ''.join(np.random.choice(list('ACGT'), n_bp))
        print('Random sequence:', sequence)

    elif sequence is not None and n_bp is None:
        n_bp = len(sequence)
        print('Sequence:', sequence)
        print('Number of base pairs:', n_bp)

    elif sequence is None and n_bp is None:
        sequence = 'CGCGAATTCGCG'
        n_bp = len(sequence)
        print('Default sequence:', sequence)
        print('Number of base pairs:', n_bp)

    elif sequence is not None and n_bp is not None:
        if n_bp != len(sequence):
            raise ValueError('Sequence length and n_bp do not match')
        print('Sequence:', sequence)
        print('Number of base pairs:', n_bp)
        
    return sequence, n_bp


def sequence_to_pdb(sequence='CGCGAATTCGCG', filename='my_dna', save=True, output='GROMACS'):
    """Sequence to MDtraj object with option to save as pdb file 
        adhering to the AMBER force field format
        Args:
            sequence: DNA sequence
            filename: name of the pdb file
            save: save the pdb file
            output: GROMACS or AMBER
        Returns:
            mdtraj trajectory"""

    sequence, _ = _check_input(sequence=sequence)

    # Linear strand of control points 
    point = Shapes.line((len(sequence)-1)*0.34)
    # Convert the control points to a spline
    spline = SplineFrames(point)
    # Generate the DNA structure
    generator = StructureGenerator(sequence=sequence,spline=spline)

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

def sequence_to_md(sequence=None, time=10, time_unit='picoseconds',temperature=310, solvated=False):
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

    pdb = sequence_to_pdb(sequence=sequence)
    
    if not openmm_available:
        print('But here is your DNA structure')
        pdb = sequence_to_pdb(sequence=sequence)
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
    
def plot_parameters(parameters, names, fig=None, ax=None, mean=True, std=True,figsize=[10,3.5], save=False):
    """Plot the rigid base parameters of the DNA structure
    Args:
        parameters: rigid base parameters
        names: parameter names
        fig: figure
        ax: axis
        mean: plot mean
        std: plot standard deviation
        figsize: figure size
        save: save figure
    Returns:
        figure, axis"""

    import matplotlib.pyplot as plt
    if fig is None and ax is None:
        fig,ax = plt.subplots(2,6, figsize=figsize)
    
    ax = ax.flatten()
    for _,name in enumerate(names):
        if _ > 5:
            color = 'coral'
        else:
            color = 'cornflowerblue'

        para = parameters[:,:,names.index(name)]
        mean = np.mean(para, axis=0)
        std = np.std(para, axis=0)
        x = range(len(mean))
        #ax[_].errorbar(x,mean, yerr=std, fmt='-', color=color)
        ax[_].fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
        ax[_].plot(mean, color=color,lw=1)    
        ax[_].scatter(x=x,y=mean,color=color,s=10)
        ax[_].set_title(name)

    fig.tight_layout()
    if save:
        fig.savefig('parameters.png')
    return fig, ax 
