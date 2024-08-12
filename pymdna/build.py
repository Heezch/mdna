import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mdtraj as md
from .spline import SplineFrames
from .nucleic import Nucleic
from .utils import _check_input
from pmcpy.run.run import Run
import copy
from typing import List


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
        print('Circular:',self.circular)
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

    def minimize(self,  frame: int = -1, exvol_rad : float = 2.0, temperature : int = 300,  simple : bool = False, equilibrate_writhe : bool = False, endpoints_fixed : bool = True, fixed : List[int] = [], dump_every : int = 20):
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

    def run(self, cycles: int, dump_every: int = 20, start_id: int = 0) -> np.ndarray:
        """Run the Monte Carlo simulation"""
        raise NotImplementedError("This method is not implemented yet.")

