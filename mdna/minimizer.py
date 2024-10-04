import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mdtraj as md
from .nucleic import Nucleic
from .spline import SplineFrames
from .utils import _check_input
from .PMCpy.pmcpy.run.run import Run
import copy
from typing import List

# Helper functions
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

    def minimize(self,  frame: int = -1, exvol_rad : float = 2.0, temperature : int = 300,  simple : bool = False, equilibrate_writhe : bool = False, endpoints_fixed : bool = True, fixed : List[int] = [], dump_every : int = 20, plot : bool = False):
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
            self.out = minimizer.equilibrate(dump_every=dump_every,plot_equi=plot)

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



class Extender:
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



class Connector:
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
           



