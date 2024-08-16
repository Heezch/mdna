import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import mdtraj as md
import utils 
from scipy.interpolate import CubicSpline


class GrooveAnalysis:
    
    """
    DNA groove analysis

    Example usage:

    traj = md.load('./data/md/0_highaff/FI/drytrajs/dry_10.xtc',top='./data/md/0_highaff/FI/drytrajs/dry_10.pdb')
    grooves = GrooveAnalysis(traj) 
    fig,ax=plt.subplots(figsize=(4,2))
    grooves.plot_groove_widths(ax=ax)

    """
    
    def __init__(self, raw_traj, points=None):
        
        self.points = points
        self.sequence = utils._sequence_letters(raw_traj)
        self.base_pairs = utils._base_pair_letters(raw_traj)
        self.get_phosphors_only_traj(raw_traj) 
        
        self.nbp = len(self.DNA_traj.top._residues)//2
        
        # Fit cubic spline curves based on phosphor atoms in strands
        if self.points:
            pass
        else:
            self.points = (self.nbp-1)*4
        
        self.fit_cubic_spline()
        self.compute_groove_widths()
    
    def describe(self):
        print(f'Your DNA has {self.nbp} base pairs and contains {self.DNA_traj.n_frames} frames.')
     
    def get_phosphors_only_traj(self, raw_traj):

        # Assuming first two chains are the DNA strands
        DNA_indices = [i.index for c in raw_traj.top._chains[:2] for i in c.atoms]
        self.DNA_traj = raw_traj.atom_slice(DNA_indices)

         # Select only phosphor atoms
        phos_indices = self.DNA_traj.top.select('name P')
        self.phos_traj = self.DNA_traj.atom_slice(phos_indices).center_coordinates()
        
        # Make new topology of chains
        phos_chain_A = self.phos_traj.topology._chains[0]
        phos_chain_B = self.phos_traj.topology._chains[1]
        
        # Split phos traj in two seperate traj for each strand
        self.strand_A = self.phos_traj.atom_slice([i.index for i in phos_chain_A.atoms])
        self.strand_B = self.phos_traj.atom_slice([i.index for i in phos_chain_B.atoms])
        
    def fit_cubic_spline(self):
    
        # fit cubic spline curve
        curves_A = self.fit_curves(self.strand_A, self.points)
        curves_B = self.fit_curves(self.strand_B, self.points)

        # reshape back to trajectory xyz format
        a_curve_xyz = curves_A.T.swapaxes(1,2)
        b_curve_xyz = curves_B.T.swapaxes(1,2)

        # convert curves to xyz format
        curves = np.hstack((a_curve_xyz,b_curve_xyz))

        # Retrieve "predicted points/particles"
        n_points = curves.shape[1]
        points_strand_A = range(0,(int((n_points/2))))
        points_strand_B = range((int((n_points/2))),(int((n_points))))

        # Generate pairs between the two strands
        #self.pairs = list(itertools.product(points_strand_A, points_strand_B))
        self.pairs = [(point_A, point_B) for point_A in points_strand_A for point_B in points_strand_B]

        # Calculate pair distances
        self.distances = utils._compute_distance(curves, self.pairs)

        # Reshape pair distances to n x n matrix and account for vdWaals radii of P atoms
        s = self.distances.shape
        self.distance_matrices = self.distances.reshape(s[0],len(points_strand_A),len(points_strand_B))[::-1,:,::-1] - 0.58
    

    def fit_curves(self, traj, points):
        
        # make curve with shape[n_particles, xyz, time]
        xyz = traj.xyz.T.swapaxes(0,1)
        n_particles = len(xyz)
        particle_list = list(range(0,n_particles))

        # initialize predictor based on # of actual particles
        predictor = CubicSpline(particle_list,xyz)

        # points of curve to interpolate the particles 
        return predictor(np.linspace(0,len(xyz),points))
    
    @staticmethod
    def find_first_local_minimum(arr):
        for i in range(1, len(arr) - 1):
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                return arr[i]  # Return the first local minimum found
        return np.NaN  # Return None if no local minimum is found

    @staticmethod
    def split_array(array):
        # Compute the midpoint of the array
        midpoint = len(array) // 2
        
        # If the length of the array is odd, exclude the middle element. 
        # If it's even, split the array into two equal halves.
        # Return the two halves that correspond to lower and upper triangles of the distance matrix
        # Reverse the first array because the order of the anti diagonal's in the opposite direction of backone/trailing diagonal
        return array[:midpoint][::-1], array[midpoint + len(array) % 2:]

    @staticmethod
    def get_anti_diagonal_slices(matrix):
        n = matrix.shape[0] # Get the size of the matrix and skip the first and last anti diagonal
        return [np.diagonal(np.flipud(matrix), offset) for offset in range(-(n - 2), n - 2)]


    def get_minor_major_widths(self,distance_matrix):

        # Split the distance matrix into anti diagonal slices
        diagonal_slices = self.get_anti_diagonal_slices(distance_matrix)
        
        minor_widths, major_widths = [],[]
        for slice_ in diagonal_slices:
            minor, major = self.split_array(slice_)
            major_widths.append(self.find_first_local_minimum(major))  # Find local minimum in the major slice and add it to the list
            minor_widths.append(self.find_first_local_minimum(minor))  # Find local minimum in the minor slice and add it to the list 
        
        return minor_widths, major_widths
    
    def compute_groove_widths(self):

        # Parallelize the computation and subsract 0.58 to account for the vdw radius of the phosphorus atoms
        results = Parallel(n_jobs=-1)(delayed(self.get_minor_major_widths)(distance_matrix) for distance_matrix in self.distance_matrices)
        minor_widths_batch, major_widths_batch = zip(*results)
        self.minor_widths = minor_widths_batch
        self.major_widths = major_widths_batch


    def plot_width(self,ax,groove,color=None,std=True,ls='-',lw=0.5):
            # Calculate the mean and standard deviation of major widths
            mean = np.nanmean(groove, axis=0)
            stds = np.nanstd(groove, axis=0)
            ax.plot(mean,color=color,ls=ls,lw=lw)
            if std:
                # Fill the area around the mean for widths
                ax.fill_between(range(len(mean)), mean - stds, mean + stds, alpha=0.1,color=color,ls='-',lw=lw)

    def plot_groove_widths(self, minor=True, major=True, std=True, color='k', c_minor=None, lw=0.5,c_major=None, ax=None, base_labels=True,ls='-'):

        # Create a figure and axes for plotting
        if ax is None:
            _, ax = plt.subplots()

        # If c_minor or c_major are not provided, use the general color
        if c_minor is None:
            c_minor = color
        if c_major is None:
            c_major = color
        if (minor and major) and (c_minor == c_major):
            c_minor = 'cornflowerblue'
            c_major = 'coral'

        if minor:
            self.plot_width(ax, self.minor_widths, std=std, color=c_minor,ls=ls,lw=lw)
        if major:
            self.plot_width(ax, self.major_widths, std=std, color=c_major,ls=ls,lw=lw)

        if base_labels:
            ax.set_xticks(np.linspace(0,len(self.major_widths[0]),len(self.base_pairs)).astype(float))
            ax.set_xticklabels(self.base_pairs,rotation=90)

    def calculate_groove_depths(self):
        # Having defined a groove width by a minimal distance at some point along the nucleic acid fragment,
        # we have to calculate the corresponding groove depth. 

        # At a base pair level, this is defined as the distance from the centre of 
        # the backbone-to-backbone width vector to the mid-point of a vector defining the corresponding base pair. 
        # This vector is constructed using the C8 atom of purines and the C6 atom of pyrimidines 

        # For groove depths half-way between base pair levels,
        # we use the average of the corresponding base pair vector mid-points. 
        pass

    def calculate_groove_depths(self):
        # Calculate the C8 and C6 atom positions of purines and pyrimidines
        c8_indices = self.DNA_traj.top.select('name C8')
        c6_indices = self.DNA_traj.top.select('name C6')
        c8_positions = self.DNA_traj.xyz[:, c8_indices]
        c6_positions = self.DNA_traj.xyz[:, c6_indices]

        # Calculate mid-points of vectors defining base pairs
        base_pair_mid_points = 0.5 * (c8_positions + c6_positions)

        # Calculate backbone-to-backbone width vector centres
        backbone_width_vector_centres = 0.5 * (self.strand_A.xyz + self.strand_B.xyz)

        # Calculate groove depths at base pair level
        groove_depths_base_pair_level = np.linalg.norm(base_pair_mid_points - backbone_width_vector_centres, axis=-1)

        # For groove depths half-way between base pair levels, we use the average of the corresponding base pair vector mid-points
        groove_depths_half_way = 0.5 * (groove_depths_base_pair_level[:-1] + groove_depths_base_pair_level[1:])

        return groove_depths_base_pair_level, groove_depths_half_way
