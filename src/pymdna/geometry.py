import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import quaternionic as qt
from .utils import RigidBody, get_data_file_path, get_sequence_letters
from numba import jit 

NUCLEOBASE_DICT =  {'A': ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
                    'T': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6'],
                    'G': ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
                    'C': ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
                    'U': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'],
                    'D': ['N1','C2','O2','N3','C4','C6','C14','C13','N5','C11','S12','C7','C8','C9','C10'],
                    'E': ['N9', 'C8', 'N7', 'C5', 'C6', 'N1', 'C2', 'N2', 'N3', 'C4'],
                    'L': ['C1','N1','S1','C2','C3','C4','C5','C6','C7', 'C8','C9','C10'],
                    'M': ['C1','C2','C3','C4','C5','C6','C20','C21','C22','C23','O37','C38'],
                    'B': ['N1', 'C2', 'N2', 'N3', 'C4', 'N5', 'C6', 'O6', 'C7', 'C8', 'N9'],
                    'S': ['N', 'C1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', 'ON1', 'ON2'],
                    'Z': ['C1', 'C2', 'C4', 'C6', 'C7', 'N2', 'N3', 'N5', 'O4'],
                    'P': ['N9', 'C8', 'N7', 'C6', 'N6', 'C5', 'N1', 'C2', 'O2', 'N3', 'C4']}

class ReferenceBase:
    """_summary_
    """
    def __init__(self, traj):
        """_summary_

        Args:
            traj (_type_): _description_
        """
        self.traj = traj
        # Determine base type (purine/pyrimidine/other)
        self.base_type = self.get_base_type()
        # Define the Tsukuba convention parameters
        self.tau_1, self.tau_2, self.d = np.radians(141.478), -np.radians(54.418), 0.4702     
        # Get coordinates of key atoms based on base type
        self.C1_coords, self.N_coords, self.C_coords = self.get_coordinates()
        # Calculate base reference point and base vectors
        self.b_R, self.b_L, self.b_D, self.b_N = self.calculate_base_frame()
        # self.basis = np.array([self.b_D.T, self.b_L.T, self.b_N])
    
    def _select_atom_by_name(self, name):
        """_summary_

        Args:
            name (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Select an atom by name returns shape (n_frames, 1, [x,y,z])
        return np.squeeze(self.traj.xyz[:,[self.traj.topology.select(f'name {name}')[0]],:],axis=1)
        
    def get_base_type(self):
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # Extracts all non-hydrogen atoms from the trajectory topology
        atoms = {atom.name for atom in self.traj.topology.atoms if atom.element.symbol != 'H'}
    
        # Check each base in the dictionary to see if all its atoms are present in the extracted atoms set
        for base, base_atoms in NUCLEOBASE_DICT.items():
            if all(atom in atoms for atom in base_atoms):
                return base
        # If no base matches, raise an error
        raise ValueError("Cannot determine the base type from the PDB file.")
    
    def get_coordinates(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # Get the coordinates of key atoms based on the base type
        C1_coords = self._select_atom_by_name('"C1\'"')
        if self.base_type in ['C','T','U','D']:# "pyrimidine"
            N_coords = self._select_atom_by_name("N1")
            C_coords = self._select_atom_by_name("C2")
        elif self.base_type in ['A','G','E','B','P']:# "purine":
            N_coords = self._select_atom_by_name("N9")
            C_coords = self._select_atom_by_name("C4") 
        elif self.base_type in ['S','Z']: # Hachi pyrimidine analogues
            N_coords = self._select_atom_by_name("C1")
            C_coords = self._select_atom_by_name("C2")
        elif self.base_type in ['L']: # UBPs hydrophobic
            N_coords = self._select_atom_by_name("N1")
            C_coords = self._select_atom_by_name("C5")
        elif self.base_type in ['M']: # UBPs hydrophilic
            N_coords = self._select_atom_by_name("C1")
            C_coords = self._select_atom_by_name("C6")
        return C1_coords, N_coords, C_coords
    
    
    def calculate_base_frame(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        # Calculate normal vector using cross product of vectors formed by key atoms
        #  The coords have the shape (n,1,3)
        b_N = np.cross((self.N_coords - self.C1_coords), (self.N_coords-self.C_coords), axis=1)
        b_N /= np.linalg.norm(b_N, axis=1, keepdims=True)  # Normalize b_N to have unit length

        # Compute displacement vector N-C1' 
        N_C1_vector = self.C1_coords - self.N_coords  # Pointing from N to C1'
        N_C1_vector /= np.linalg.norm(N_C1_vector, axis=1, keepdims=True)

        # Rotate N-C1' vector by angle tau_1 around b_N to get the direction for displacement
        R_b_R = RigidBody.get_rotation_matrix(self.tau_1 * b_N)
       
        # Displace N along this direction by a distance d to get b_R
        b_R = self.N_coords + np.einsum('ijk,ik->ij', R_b_R, N_C1_vector * self.d)
     
        # Take a unit vector in the N-C1' direction, rotate it around b_N by angle tau_2 to get b_L
        R_b_L = RigidBody.get_rotation_matrix(self.tau_2 * b_N)
        b_L = np.einsum('ijk,ik->ij', R_b_L, N_C1_vector) 

        # Calculate b_D using cross product of b_L and b_N
        b_D = np.cross(b_L, b_N, axis=1)
        
        return np.array([b_R, b_D, b_L, b_N])
        #return np.array([b_R, -b_D, -b_L, -b_N])

    def plot_baseframe(self,atoms=True, frame=True, ax=None,length=1):
        """_summary_

        Args:
            atoms (bool, optional): _description_. Defaults to True.
            frame (bool, optional): _description_. Defaults to True.
            ax (_type_, optional): _description_. Defaults to None.
            length (int, optional): _description_. Defaults to 1.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = False

        # Plot the DNA atoms
        if atoms:
            atoms_coords = self.traj.xyz[0]
            ax.scatter(atoms_coords[:,0], atoms_coords[:,1], atoms_coords[:,2], alpha=0.6)

        # Plot the reference frame vectors
        if frame:
            origin = self.b_R[0]
            ax.quiver(origin[0], origin[1], origin[2], 
                    self.b_L[0][0], self.b_L[0][1], self.b_L[0][2], 
                    color='r', length=length, normalize=True)
            ax.quiver(origin[0], origin[1], origin[2], 
                    self.b_D[0][0], self.b_D[0][1], self.b_D[0][2], 
                    color='g', length=length, normalize=True)
            ax.quiver(origin[0], origin[1], origin[2], 
                    self.b_N[0][0], self.b_N[0][1], self.b_N[0][2], 
                    color='b', length=length, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if fig: 
            # Make axes of equal length
            max_range = np.array([
                atoms_coords[:,0].max()-atoms_coords[:,0].min(), 
                atoms_coords[:,1].max()-atoms_coords[:,1].min(), 
                atoms_coords[:,2].max()-atoms_coords[:,2].min()
            ]).max() / 2.0

            mid_x = (atoms_coords[:,0].max()+atoms_coords[:,0].min()) * 0.5
            mid_y = (atoms_coords[:,1].max()+atoms_coords[:,1].min()) * 0.5
            mid_z = (atoms_coords[:,2].max()+atoms_coords[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.axis('equal')

class NucleicFrames:
    """Class to compute the rigid base parameters of a DNA structure.
    
    loc = '/Users/thor/surfdrive/Scripts/notebooks/HNS-sequence/WorkingDir/nolinker/data/md/0_highaff/FI/drytrajs/'
    traj = md.load(loc+'dry_10.xtc',top=loc+'dry_10.pdb')

    dna = NucleicFrames(traj)
    params, names = dna.get_paramters()
    params.shape, names

    # Confidence intervals 
    from scipy.stats import t

    fig, ax = plt.subplots(2,6,figsize=(12,4))
    fig.tight_layout()
    ax = ax.flatten()
    M = np.mean(params, axis=0)
    S = np.std(params, axis=0)
    n = params.shape[0]
    ci = t.ppf(0.975, df=n-1) * S / np.sqrt(n)
    x = np.arange(0, params.shape[1])
    for _, i in enumerate(M.T):
        if _ >= 6:
            c1, c2 = 'red','coral'
        else:
            c1, c2 = 'blue','cornflowerblue'
        ax[_].plot(i[::-1], '-o',color=c1)
        ax[_].fill_between(x, (i-ci[_])[::-1], (i+ci[_])[::-1], color=c2, alpha=0.2)
        ax[_].set_title(names[_])
    """

    def __init__(self, traj, chainids=[0,1]):
        """Initialize the NucleicFrames object.

        Args:
            traj (object): MDtraj trajectory object.
            chainids (list, optional): Chainids of sense- and anti-sense strands. Defaults to [0,1].
        """
        self.traj = traj
        self.top = traj.topology
        self.res_A = self.get_residues(chain_index=chainids[0], reverse=False)
        self.res_B = self.get_residues(chain_index=chainids[1], reverse=True)
        self.mean_reference_frames = np.empty((len(self.res_A), 1, 4, 3))
        self.base_frames = self.get_base_reference_frames()
        self.analyse_frames()

    def get_residues(self, chain_index, reverse=False):
        """Get residues from specified chain."""
        if chain_index >= len(self.top._chains):
            raise IndexError("Chain index out of range.")
        chain = self.top._chains[chain_index]
        residues = chain._residues
        return list(reversed(residues)) if reverse else residues

    def load_reference_bases(self):
        """Load reference bases from local files."""
        # Not used at the moment??
        bases = ['C', 'G', 'T', 'A']
        #return {f'D{base}': md.load_pdb(get_data_file_path(f'./atomic/NDB96_{base}.pdb')) for base in bases}
        return {f'D{base}': md.load_hdf5(get_data_file_path(f'./atomic/bases/BDNA_{base}.h5')) for base in bases}

    def get_base_vectors(self, res):
        """Compute base vectors from reference base."""
        ref_base = ReferenceBase(res)
        return np.array([ref_base.b_R, ref_base.b_L, ref_base.b_D, ref_base.b_N]).swapaxes(0,1)
    
    def get_base_reference_frames(self):
        """Get reference frames for each residue."""
        reference_frames = {} # Dictionary to store the base vectors for each residue
        for res in self.res_A + self.res_B:
            res_traj = self.traj.atom_slice([at.index for at in res.atoms])
            base_vectors = self.get_base_vectors(res_traj)
            reference_frames[res] = base_vectors # Store the base vectors for the residue index (with shape (4, n_frames, 3))
        return reference_frames

    def reshape_input(self,input_A,input_B,is_step=False):
        
        """Reshape the input to the correct format for the calculations.
        
        Args:
        input_A (ndarray): Input array for the first triad.
        input_B (ndarray): Input array for the second triad.
        is_step (bool, optional): Flag indicating if the input is a single step or a trajectory. Defaults to False.
        
        Returns:
        rotation_A (ndarray): Rotation matrices of shape (n, 3, 3) for the first triad.
        rotation_B (ndarray): Rotation matrices of shape (n, 3, 3) for the second triad.
        origin_A (ndarray): Origins of shape (n, 3) for the first triad.
        origin_B (ndarray): Origins of shape (n, 3) for the second triad.
        original_shape (tuple): The original shape of the input.
        """

        # Store original shape
        original_shape = input_A.shape

        # Flatten frames to compute rotation matrices for each time step simultaneously
        input_A_ = input_A.reshape(-1,original_shape[-2],original_shape[-1])  # shape (n, 4, 3)
        input_B_ = input_B.reshape(-1,original_shape[-2],original_shape[-1])  # shape (n, 4, 3)

        # Extract the triads without origin (rotation matrices)
        rotation_A = input_A_[:,1:]  # shape (n, 3, 3)
        rotation_B = input_B_[:,1:]  # shape (n, 3, 3)

        if not is_step:
            # flip (connecting the backbones) and the (baseplane normals).
            # so the second and third vector b_L, b_N
            rotation_B[:,[1,2]] *= -1
     
        # Extract origins of triads
        origin_A = input_A_[:,0]  # shape (n, 3)
        origin_B = input_B_[:,0]  # shape (n, 3)

        return rotation_A, rotation_B, origin_A, origin_B, original_shape


    def compute_parameters(self, rotation_A, rotation_B, origin_A, origin_B):
        """Calculate the parameters between each base pair and mean reference frames.

        Args:
            rotation_A (ndarray): Rotation matrices of shape (n, 3, 3) for the first triad.
            rotation_B (ndarray): Rotation matrices of shape (n, 3, 3) for the second triad.
            origin_A (ndarray): Origins of shape (n, 3) for the first triad.
            origin_B (ndarray): Origins of shape (n, 3) for the second triad.

        Returns:
            rigid_parameters (ndarray): The parameters of shape (n, 12) representing the relative translation and rotation between each base pair.
            trans_mid (ndarray): The mean translational vector of shape (n, 3) between the triads.
            rotation_mid (ndarray): The mean rotation matrix of shape (n, 3, 3) between the triads.
        """
        
        # Linear interpolation of translations
        trans_mid = 0.5 * (origin_A + origin_B)
    
        # Relative translation
        trans_AB = origin_A - origin_B

        # Get relative rotation matrix of base pair
        rotation_BA = rotation_B.transpose(0,2,1) @ rotation_A  # returns shape (n, 3, 3)

        # Get rotation angles based on  rotation matrices
        rotation_angle_BA = RigidBody.extract_omega_values(rotation_BA)

        # Compute halfway rotation matrix and triad (mid frame)
        rotation_halfway = RigidBody.get_rotation_matrix(rotation_angle_BA * 0.5)

        # Get rotation matrix of base pair (aka mean rotation frame)
        rotation_mid = rotation_B @ rotation_halfway 
        
        # Get transaltional coordinate vector and convert to angstroms
        translational_parameters = np.einsum('ijk,ik->ij',rotation_mid.transpose(0,2,1), trans_AB) * 10

        # Get rotational parameters and convert to degrees
        rotational_parameters = np.rad2deg(np.einsum('ijk,ik->ij', rotation_BA.transpose(0,2,1), rotation_angle_BA))
                
        # Merge translational and rotational parameters
        rigid_parameters = np.hstack((translational_parameters, rotational_parameters))

        # Return the parameters and the mean reference frame
        return rigid_parameters, trans_mid, rotation_mid


    def calculate_parameters(self,frames_A, frames_B, is_step=False):
        """Calculate the parameters between each base pair and mean reference frames.

        Assumes frames are of shape (n_frames, n_residues, 4, 3) where the last two dimensions are the base triads.
        The base triads consist of an origin (first index) and three vectors (latter 3 indices) representing the base frame.
        With the order of the vectors being: b_R, b_L, b_D, b_N.

        Args:
            frames_A (ndarray): Frames of shape (n_frames, n_residues, 4, 3) representing the base triads for chain A.
            frames_B (ndarray): Frames of shape (n_frames, n_residues, 4, 3) representing the base triads for chain B.
            is_step (bool, optional): Flag indicating if the input is a single step or a trajectory. Defaults to False.

        Notes:
            Note the vectors are stored rowwise in the base triads, and not the usual column representation of the rotation matrices.

        Returns:
            params (ndarray): The parameters of shape (n_frames, n_residues, 6) representing the relative translation and rotation between each base pair.
            mean_reference_frames (ndarray): The mean reference frames of shape (n_bp, n_frames, 4, 3) representing the mean reference frame of each base pair.
        """
                
        # Reshape frames
        rotation_A, rotation_B, origin_A, origin_B, original_shape = self.reshape_input(frames_A,frames_B, is_step=is_step)

        # Compute parameters
        if not is_step:
            # Flip from row to column representation of the rotation matrices
            rotation_A = rotation_A.transpose(0,2,1)
            rotation_B = rotation_B.transpose(0,2,1)
            params, mean_origin, mean_rotation = self.compute_parameters(rotation_A, rotation_B, origin_A, origin_B)
        else:
            # Switch the input of the B and A triads to get the correct parameters
            params, mean_origin, mean_rotation = self.compute_parameters(rotation_B, rotation_A, origin_B, origin_A)

        # Reshape the parameters to the original shape
        params = params.reshape(original_shape[0], original_shape[1], 6).swapaxes(0, 1)

        # Collect mean reference frames from mid frames of each base pair
        mean_reference_frames = np.hstack((mean_origin[:, np.newaxis, :],mean_rotation)).reshape(original_shape)

        if is_step:
            # Creating an array of zeros with shape (10000, 1, 6)
            extra_column = np.zeros((params.shape[0], 1, 6))

            # Concatenating the existing array and the extra column along the second axis
            params = np.concatenate((extra_column,params), axis=1)

        # Return the parameters and the mean reference frames
        return  params, mean_reference_frames if not is_step else params


    def analyse_frames(self):
        """Analyze the trajectory and compute parameters."""

        # Get base reference frames for each residue
        frames_A = np.array([self.base_frames[res] for res in self.res_A])
        frames_B = np.array([self.base_frames[res] for res in self.res_B])

        # Compute parameters between each base pair and mean reference frames
        self.bp_params, self.mean_reference_frames = self.calculate_parameters(frames_A, frames_B)
        
        # Extract mean reference frames for each neighboring base pair
        B1_triads = self.mean_reference_frames[:-1] # select all but the last frame
        B2_triads = self.mean_reference_frames[1:] # select all but the first frame

        # Compute parameters between each base pair and mean reference frames
        self.step_params = self.calculate_parameters(B1_triads, B2_triads, is_step=True)[0]

        # Store mean reference frame / aka base pair triads as frames and transpose rotation matrices back to row wise
        self.frames = self.mean_reference_frames
        self.frames[:, :, 1:, :] = np.transpose(self.frames[:, :, 1:, :], axes=(0, 1, 3, 2))
        self._clean_parameters()

    def _clean_parameters(self):
        """Clean the parameters by removing the first and last frame."""
        self.step_parameter_names = ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']
        self.base_parameter_names = ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']
        self.names = self.base_parameter_names + self.step_parameter_names
        self.parameters = np.dstack((self.bp_params, self.step_params))

    def get_parameters(self,step=False,base=False):
        """Return the computed parameters of shape (n_frames, n_base_pairs, n_parameters)"""
        if step and not base:
            return self.step_params, self.step_parameter_names
        elif base and not step:
            return self.bp_params, self.base_parameter_names
        elif not step and not base:
            return self.parameters, self.names
        
    def get_parameter(self,name='twist'):
        """Get the parameter of the DNA structure, choose frome the following:
        - shift, slide, rise, tilt, roll, twist, shear, stretch, stagger, buckle, propeller, opening

        Args:
            name (str): parameter name

        Returns:
            parameter(ndarray) : parameter in shape (n_frames, n_base_pairs)"""

        if name not in self.names:
            raise ValueError(f"Parameter {name} not found.")
        return self.parameters[:,:,self.names.index(name)]
    

    def plot_parameters(self, fig=None, ax=None, mean=True, std=True,figsize=[10,3.5], save=False,step=True,base=True,base_color='cornflowerblue',step_color='coral'):
        """Plot the rigid base parameters of the DNA structure
        Args:
            fig: figure
            ax: axis
            mean: plot mean
            std: plot standard deviation
            figsize: figure size
            save: save figure
        Returns:
            figure, axis"""

        import matplotlib.pyplot as plt

        cols = step + base

        if fig is None and ax is None:
            fig,ax = plt.subplots(cols,6, figsize=[12,2*cols])
            ax = ax.flatten()
        if step and not base:
            names = self.step_parameter_names
        elif base and not step:
            names = self.base_parameter_names
        elif base and step:
            names = self.names

        for _,name in enumerate(names):
            if name in self.step_parameter_names:
                color = step_color
            else:
                color = base_color
            para = self.get_parameter(name)
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

        
        

class NucleicFrames_quaternion:

    def __init__(self, traj,chainids=[0,1], euler=False,cayley=False,angle=False):

        self.traj = traj
        self.top = traj.topology
        self.euler = euler
        self.cayley = cayley
        self.angle = angle
        self.angles = []
        if not self.cayley and not self.euler and not self.angle:
            self.euler = True
            
        self.sequence_list = get_sequence_letters(traj,leading_chain=chainids[0])
        self.n_bp = len(self.sequence_list)
        self.res_A = self._get_residues(chain_index=chainids[0], reverse=False)
        self.res_B = self._get_residues(chain_index=chainids[1], reverse=True)
        self.base_frames = self.get_base_reference_frames()
        self.base_quats = {}
        self.analyze()
        

    def _get_residues(self, chain_index, reverse=False):
        """Get residues from specified chain."""
        if chain_index >= len(self.top._chains):
            raise IndexError("Chain index out of range.")
        chain = self.top._chains[chain_index]
        residues = chain._residues
        return list(reversed(residues)) if reverse else residues

    def _get_base_vectors(self, res):
        """Compute base vectors from reference base."""
        ref_base = ReferenceBase(res)
        return np.array([ref_base.b_R, ref_base.b_L, ref_base.b_D, ref_base.b_N]).swapaxes(0,1)
        
    def get_base_reference_frames(self):
        """Get reference frames for each residue.

            The MDtraj residue instance is the key to the  value which contains 
            the reference point and base vectors for each frame (n_frames, 4, 3).
            The base vectors are ordered as follows: b_R, b_L, b_D, b_N.    

        Returns:
            reference_frames: Dictionary to store the base vectors for each residue.                   
        """
        reference_frames = {} # Dictionary to store the base vectors for each residue
        for res in self.res_A + self.res_B:
            res_traj = self.traj.atom_slice([at.index for at in res.atoms])
            base_vectors = self._get_base_vectors(res_traj)
            reference_frames[res] = base_vectors # Store the base vectors for the residue index (with shape (4, n_frames, 3))
        return reference_frames
    
    
    def convert_base_frame(self, frame, anti=False): 
        """Convert a base pair frame to a translation and quaternion representation.
        Args:
            frame: a 4x4 matrix representing the base pair frame where the rotations are stored row wise
        
            Unflips the quaternion if necessary when dealing with time series data. 
        
        returns: a tuple of (translations, quaternion)"""

        # flip (connecting the backbones) and the (baseplane normals) (2,3)  vector b_L, b_N
        if anti:
            frame[:,[2,3]] *= -1

        # Get the translation and rotation from the base pair frame
        translations = frame[:,0]  # extract translations
        rotation = frame[:,1:].transpose(0, 2, 1) # extract rotation matrices as column vectors

        # Convert the rotation matrices to quaternions
        quaternion = qt.array.from_rotation_matrix(rotation)
        #quaternion = quaternion.to_minimal_rotation(range(quaternion.shape[0]))
        # quaternion = qt.array([[np.abs(q.w),q.x, q.y, q.z] for q in quaternion])
        # print(quaternion.shape,quaternion[0])
        # # Check if the quaternion needs to be unflipped when dealing with time series
        # if quaternion.shape[0] > 1:
        #     #quaternion = qt.unflip_rotors(quaternion)
        #     quaternion = quaternion.to_minimal_rotation(range(quaternion.shape[0]),iterations=100)
            
      # Return the translation and quaternion
        return (translations, quaternion)
    
    def process_chain(self, residues, anti=False):
        """Process the chain and convert the base frames to translation and quaternion representations."""

        # Preallocate numpy arrays for efficiency
        translations = np.zeros((len(residues), self.traj.n_frames,3))
        quaternions = qt.array(np.zeros((len(residues), self.traj.n_frames, 4)))

        # Loop through the residues and convert the base frames to translation and quaternion representations
        for i, res in enumerate(residues):
            translations[i], quaternions[i] = self.convert_base_frame(np.copy(self.base_frames[res]),anti=anti)

        # Flatten the arrays to stack all residues after one another
        return translations, quaternions




    def compute_parameters_(self,translations_A, quaternions_A, translations_B, quaternions_B, t=0.5):
        """Compute the rigid body parameters between two frames."""
        
        # Linear interpolation of translations
        trans_mid = (1 - t) * translations_A + t * translations_B

        # Slerp (spherical linear interpolation) for quaternion
        # Note that output slerp(q1, q2, 1) may be different from q2. (slerp(q1, q2, 0) is always equal to q1.)
        quat_mid = qt.slerp(quaternions_A, quaternions_B, tau=t)

        # Convert quaternion to rotation matrix
        rotation_mid = quat_mid.to_rotation_matrix  

        # Compute the relative translation
        translation = translations_A - translations_B

        # Get translational coordinate vector and convert to angstroms
        translational_parameters = np.einsum('ijk,ik->ij', rotation_mid.transpose(0,2,1), translation)*10

        # Get the elative rotation matrix
        A = quaternions_B.inverse * quaternions_A
        self.angles.append(self.compute_angle(A))

        if self.euler:
            # https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
            # We have to be careful because the gimbal lock problem can occur when the pitch angle is close to +/-90 degrees,
            # or the yaw and roll axes of rotation are aligned with each other in the world coordinate system, and therefore produce the same effect.
            # See link above for more details on how to mitigate this problem.
            # Maybe also have another look at: https://amu.hal.science/hal-03848730/document
            # Title: Quaternion to Euler angles conversion: a direct, general and computationally efficient method from 2022
            pitch = np.arcsin(2.0*(A.w*A.y - A.x*A.z))
            yaw = np.arctan2(2.0*(A.y*A.z + A.w*A.x), A.w*A.w - A.x*A.x - A.y*A.y + A.z*A.z)
            roll = np.arctan2(2.0*(A.x*A.y + A.w*A.z), A.w*A.w + A.x*A.x - A.y*A.y - A.z*A.z)
            # # 
            # # # Initialize yaw and roll with default values that will apply when no specific condition is met
            # yaw = np.zeros_like(pitch)
            # roll = np.zeros_like(pitch)

            # # Condition for pitch = pi/2
            # mask_pi_2 = pitch == np.pi/2
            # yaw[mask_pi_2] = -np.arctan2(A.x[mask_pi_2],A.w[mask_pi_2])
            # roll[mask_pi_2] = 0.0

            # # Condition for pitch = -pi/2
            # mask_neg_pi_2 = pitch == -np.pi/2
            # yaw[mask_neg_pi_2] = np.arctan2(A.x[mask_neg_pi_2],A.w[mask_neg_pi_2])
            # roll[mask_neg_pi_2] = 0.0

            # # Default condition (where neither pi/2 nor -pi/2 conditions are met)
            # mask_else = ~(mask_pi_2 | mask_neg_pi_2)
            # yaw[mask_else] = np.arctan2(2.0*(A.y[mask_else]*A.z[mask_else] + A.w[mask_else]*A.x[mask_else]),
            #                             A.w[mask_else]**2 - A.x[mask_else]**2 - A.y[mask_else]**2 + A.z[mask_else]**2)
            # roll[mask_else] = np.arctan2(2.0*(A.x[mask_else]*A.y[mask_else] + A.w[mask_else]*A.z[mask_else]),
            #                             A.w[mask_else]**2 + A.x[mask_else]**2 - A.y[mask_else]**2 - A.z[mask_else]**2)
        
                
            rotational_parameters = np.vstack((yaw, pitch, roll)).swapaxes(0,1)
        else:
            # Get axis angle representation of the relative rotation matrix
            # Each vector represents the axis of the rotation, with norm equal to the angle of the rotation in radians.
            rotational_parameters = A.to_axis_angle # unfortunately this results in angle wraps.

        # Stack the translational and rotational parameters and convert the latter to degrees
        rigid_parameters = np.hstack((translational_parameters, np.rad2deg(rotational_parameters)))

        # Return the rigid body parameters and the mid/halfway transformation
        return rigid_parameters, trans_mid, quat_mid # qt.unflip_rotors(quat_mid)#, quat_mid
    
    def compute_angle(self,quaternion):
        w =  np.arccos(np.clip(quaternion.w,-1,1))*2    
        return w
    
    def compute_angle_(self, q):
        w, x, y, z = q.w, q.x, q.y, q.z

        # Initialize angle and axis arrays
        angle = np.zeros(w.shape)
        axis = np.zeros((w.shape[0], 3))

        # Handle cases where w is close to 1 or -1
        close_to_1 = np.isclose(w, 1)
        close_to_neg1 = np.isclose(w, -1)

        # Handle small angles
        small_angle = 2 * np.sqrt(2 * (1 - w[close_to_1]))
        angle[close_to_1] = small_angle
        axis[close_to_1] = q[close_to_1].vector

        # Handle angles close to 360 degrees
        large_angle = 2 * np.pi - 2 * np.sqrt(2 * (1 + w[close_to_neg1]))
        angle[close_to_neg1] = large_angle
        axis[close_to_neg1] = q[close_to_neg1].vector

        # Regular cases
        normal_case = ~(close_to_1 | close_to_neg1)
        angle[normal_case] = 2 * np.arccos(w[normal_case])
        s = np.sqrt(1 - w[normal_case]**2)
        small_s = s < 1e-8
        normal_indices = np.arange(len(q))[normal_case]

        # Safe division
        axis[normal_case] = q[normal_case].vector / s[:, np.newaxis]
        axis[normal_indices[small_s], :] = np.array([1, 0, 0])  # Default axis for very small s

        # Normalize all axes
        axis_norms = np.linalg.norm(axis, axis=1, keepdims=True)
        valid_norms = axis_norms[:, 0] > 0  # Avoid division by zero
        axis[valid_norms] = axis[valid_norms] / axis_norms[valid_norms]

        return angle
    
    def compute_relative_rotation(self,quaternions_A, quaternions_B, mask):
        q0 = qt.array(np.copy(quaternions_A))
        q1 = qt.array(np.copy(quaternions_B))
        q1[mask] = -q1[mask]
        return q1.inverse * q0

    def compute_midframe(self, quaterions_A, quaterions_B, mask):
        q0 = qt.array(np.copy(quaterions_A))
        q1 = qt.array(np.copy(quaterions_B))
        q0[mask] = -q0[mask]
        return qt.slerp(q0, q1, tau=0.5)
    
        # public void set(Quat4d q1) {
    #     double test = q1.x*q1.y + q1.z*q1.w;
    #     if (test > 0.499) { // singularity at north pole
    #         heading = 2 * atan2(q1.x,q1.w);
    #         attitude = Math.PI/2;
    #         bank = 0;
    #         return;
    #     }
    #     if (test < -0.499) { // singularity at south pole
    #         heading = -2 * atan2(q1.x,q1.w);
    #         attitude = - Math.PI/2;
    #         bank = 0;
    #         return;
    #     }
    #     double sqx = q1.x*q1.x;
    #     double sqy = q1.y*q1.y;
    #     double sqz = q1.z*q1.z;
    #     heading = atan2(2*q1.y*q1.w-2*q1.x*q1.z , 1 - 2*sqy - 2*sqz);
    #     attitude = asin(2*test);
    #     bank = atan2(2*q1.x*q1.w-2*q1.y*q1.z , 1 - 2*sqx - 2*sqz)

    def compute_euler(self,q):
        test = q.x*q.y + q.z*q.w
        north_mask = test > 0.499 # singularity at north pole
        south_mask = test < -0.499 # singularity at south pole
        mask_else = ~(north_mask | south_mask)

        heading = np.zeros_like(test)
        attitude = np.zeros_like(test)
        bank = np.zeros_like(test)

        heading[mask_else] = np.arctan2(2*q.y[mask_else]*q.w[mask_else]-2*q.x[mask_else]*q.z[mask_else] , 1 - 2*q.y[mask_else]**2 - 2*q.z[mask_else]**2)
        attitude[mask_else] = np.arcsin(2*test[mask_else])
        bank[mask_else] = np.arctan2(2*q.x[mask_else]*q.w[mask_else]-2*q.y[mask_else]*q.z[mask_else] , 1 - 2*q.x[mask_else]**2 - 2*q.z[mask_else]**2)

        heading[north_mask] = 2 * np.arctan2(q.x[north_mask],q.w[north_mask])
        attitude[north_mask] = np.pi/2
        bank[north_mask] = 0

        heading[south_mask] = -2 * np.arctan2(q.x[south_mask],q.w[south_mask])
        attitude[south_mask] = -np.pi/2
        bank[south_mask] = 0

        return np.vstack((bank,heading,attitude)).swapaxes(0,1)


            #       # Condition for pitch = pi/2
            # mask_pi_2 = pitch == np.pi/2
            # yaw[mask_pi_2] = -np.arctan2(A.x[mask_pi_2],A.w[mask_pi_2])
            # roll[mask_pi_2] = 0.0

            # # Condition for pitch = -pi/2
            # mask_neg_pi_2 = pitch == -np.pi/2
            # yaw[mask_neg_pi_2] = np.arctan2(A.x[mask_neg_pi_2],A.w[mask_neg_pi_2])
            # roll[mask_neg_pi_2] = 0.0

            # # Default condition (where neither pi/2 nor -pi/2 conditions are met)
            # mask_else = ~(mask_pi_2 | mask_neg_pi_2)
            # yaw[mask_else] = np.arctan2(2.0*(A.y[mask_else]*A.z[mask_else] + A.w[mask_else]*A.x[mask_else]),
            #                             A.w[mask_else]**2 - A.x[mask_else]**2 - A.y[mask_else]**2 + A.z[mask_else]**2)
            # roll[mask_else] = np.arctan2(2.0*(A.x[mask_else]*A.y[mask_else] + A.w[mask_else]*A.z[mask_else]),
            #                             A.w[mask_else]**2 + A.x[mask_else]**2 - A.y[mask_else]**2 - A.z[mask_else]**2)

            # rotational_parameters = np.vstack((yaw, pitch, roll)).swapaxes(0,1)

    def compute_axis_angle(self,quaternion):
        # Get the axis of rotation and the angle of rotation
        angle = 2 * np.arccos(quaternion.w)
        norms = np.linalg.norm(quaternion.vector, axis=-1, keepdims=True) #+ epsilon
        axis = quaternion.vector / norms


        # Compute dot products between consecutive frames, skipping the first frame.
        # axis[:-1] is from 0 to n-2, axis[1:] is from 1 to n-1, hence pairs consecutive frames.
        dot_products = np.einsum('ij,ij->i', axis[:-1], axis[1:])

        # Evaluate the conditions.
        # Check where the dot product is less than zero and angle of current frame is greater than pi/2.
        condition = (dot_products < 0.0) & (angle[1:] > np.pi / 2)

        # Update axis where condition is True.
        axis[1:][condition] = -axis[1:][condition]

        # Update angles where condition is True.
        angle[1:][condition] = 2 * np.pi - angle[1:][condition]
        return axis * angle[..., np.newaxis]


    def compute_axis_angle(self,quaternion):
        # Get the axis of rotation and the angle of rotation
        angle = 2 * np.arccos(quaternion.w)
        norms = np.linalg.norm(quaternion.vector, axis=-1, keepdims=True) #+ epsilon
        axis = np.copy(quaternion.vector) / norms

        for _,v in enumerate(axis):
            if _ == 0:
                continue
            if (np.dot(axis[_-1],v) < 0.0) and (angle[_] > np.pi/2):
                axis[_] = -v
                angle[_] = 2*np.pi - angle[_]
        
        return axis * angle[..., np.newaxis]

    def compute_parameters(self,translations_A, quaternions_A, translations_B, quaternions_B, t=0.5):
        """Compute the rigid body parameters between two frames."""
        #dot = np.dot(quaternions_A, quaternions_B)
        
        dot_products = (quaternions_A.w*quaternions_B.w) + (quaternions_A.x*quaternions_B.x) + (quaternions_A.y*quaternions_B.y) + (quaternions_A.z*quaternions_B.z)
        mask = dot_products < 0
        quaternions_A[mask] = -quaternions_A[mask]

        # Get the relative rotation matrix
        A = quaternions_B.inverse * quaternions_A
       
        #A = quaternions_B.inverse*quaternions_A
        #A = self.compute_relative_rotation(quaternions_A, quaternions_B, mask)

        # Slerp (spherical linear interpolation) for quaternion
        # Note that output slerp(q1, q2, 1) may be different from q2. (slerp(q1, q2, 0) is always equal to q1.)
        quat_mid = qt.slerp(quaternions_A, quaternions_B, tau=t)
    
        # Linear interpolation of translations
        trans_mid = (1 - t) * translations_A + t * translations_B
    
        # Convert quaternion to rotation matrix
        rotation_mid = quat_mid.to_rotation_matrix  
  
        # Compute the relative translation
        translation = translations_A - translations_B

        # Get translational coordinate vector and convert to angstroms
        translational_parameters = np.einsum('ijk,ik->ij', rotation_mid.transpose(0,2,1), translation)*10

        # translational_parameters  = quat_mid.rotate(translation)
        # translational_parameters = translation * quat_mid.inverse
        #print(translational_parameters.shape,translation.shape, quat_mid.shape)   
        # # Get the elative rotation matrix
        # A = quaternions_B.inverse * quaternions_A
        # self.angles.append(self.compute_angle(A))

        if self.euler:
            # https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
            # We have to be careful because the gimbal lock problem can occur when the pitch angle is close to +/-90 degrees,
            # or the yaw and roll axes of rotation are aligned with each other in the world coordinate system, and therefore produce the same effect.
            # See link above for more details on how to mitigate this problem.
            # Maybe also have another look at: https://amu.hal.science/hal-03848730/document
            # Title: Quaternion to Euler angles conversion: a direct, general and computationally efficient method from 2022
            # pitch = np.arcsin(2.0*(A.w*A.y - A.x*A.z))
 
            # # yaw = np.arctan2(2.0*(A.y*A.z + A.w*A.x), A.w*A.w - A.x*A.x - A.y*A.y + A.z*A.z)
            # # roll = np.arctan2(2.0*(A.x*A.y + A.w*A.z), A.w*A.w + A.x*A.x - A.y*A.y - A.z*A.z)
            # # 
            # # # Initialize yaw and roll with default values that will apply when no specific condition is met
            # yaw = np.zeros_like(pitch)
            # roll = np.zeros_like(pitch)

            # # Condition for pitch = pi/2
            # mask_pi_2 = pitch == np.pi/2
            # yaw[mask_pi_2] = -np.arctan2(A.x[mask_pi_2],A.w[mask_pi_2])
            # roll[mask_pi_2] = 0.0

            # # Condition for pitch = -pi/2
            # mask_neg_pi_2 = pitch == -np.pi/2
            # yaw[mask_neg_pi_2] = np.arctan2(A.x[mask_neg_pi_2],A.w[mask_neg_pi_2])
            # roll[mask_neg_pi_2] = 0.0

            # # Default condition (where neither pi/2 nor -pi/2 conditions are met)
            # mask_else = ~(mask_pi_2 | mask_neg_pi_2)
            # yaw[mask_else] = np.arctan2(2.0*(A.y[mask_else]*A.z[mask_else] + A.w[mask_else]*A.x[mask_else]),
            #                             A.w[mask_else]**2 - A.x[mask_else]**2 - A.y[mask_else]**2 + A.z[mask_else]**2)
            # roll[mask_else] = np.arctan2(2.0*(A.x[mask_else]*A.y[mask_else] + A.w[mask_else]*A.z[mask_else]),
            #                             A.w[mask_else]**2 + A.x[mask_else]**2 - A.y[mask_else]**2 - A.z[mask_else]**2)

            # rotational_parameters = np.vstack((yaw, pitch, roll)).swapaxes(0,1)
            rotational_parameters = self.compute_euler(A)   
            # heading = np.arctan2(2*A.y*A.w-2*A.x*A.z , 1 - 2*A.y**2 - 2*A.z**2)
            # attitude = np.arcsin(2*A.x*A.y + 2*A.z*A.w)
            # bank = np.arctan2(2*A.x*A.w-2*A.y*A.z , 1 - 2*A.x**2 - 2*A.z**2)
            # rotational_parameters = np.vstack((bank,heading,attitude )).swapaxes(0,1)
            
        elif self.cayley:
            # rot_A = quaternions_A.to_rotation_matrix # shape n_frames, 3, 3
            # rot_B = quaternions_B.to_rotation_matrix # shape n_frames, 3, 3
            
            # # Compute the relative rotation matrix for each frame
            # # Note: .T must be applied to each matrix in rot_B; assuming it's handled within the loop or vectorization
            # A = np.einsum('nij,njk->nik', rot_B.transpose((0, 2, 1)), rot_A)  # shape n_frames, 3, 3
            # Calculate the trace of each matrix A across the frames
            trace_A = np.einsum('nii->n', A.to_rotation_matrix)  # Sum over diagonal elements for each frame

            # Compute the Cayley-Klein parameters for all frames at once
            rotational_parameters = 2 * np.array([
                A[:, 2, 1] - A[:, 1, 2],
                A[:, 0, 2] - A[:, 2, 0],
                A[:, 1, 0] - A[:, 0, 1]
            ]).T / (trace_A[:, np.newaxis] + 1)  # Adding 1 to each element of trace_A and reshaping for broadcasting


        elif self.angle:
            # Get axis angle representation of the relative rotation matrix
            # Each vector represents the axis of the rotation, with norm equal to the angle of the rotation in radians.
            #rotational_parameters = A.to_axis_angle # unfortunately this results in angle wraps.
            rotational_parameters = self.compute_axis_angle(A)

        # Stack the translational and rotational parameters and convert the latter to degrees
        rigid_parameters = np.hstack((translational_parameters, np.rad2deg(rotational_parameters)))

        # Return the rigid body parameters and the mid/halfway transformation
        return rigid_parameters, trans_mid, quat_mid#, qt.unflip_rotors(quat_mid)#, quat_mid

    def analyze(self):
        """Analyze the trajectory and compute the pair and step parameters."""

        # Initialize arrays for the pair and step parameters
        self.pair_parameters = np.empty((self.n_bp, self.traj.n_frames, 6))
        self.step_parameters = np.zeros((self.n_bp, self.traj.n_frames, 6))

        # Initialize arrays for the translation and quaternion representations of the step frames
        self.trans  = np.empty((self.n_bp, self.traj.n_frames, 3))
        self.quat = qt.array(np.empty((self.n_bp, self.traj.n_frames, 4)))
    
        # Process the leading chain
        translations_A, quaternions_A = self.process_chain(self.res_A)
        # Process the lagging chain
        translations_B, quaternions_B = self.process_chain(self.res_B, anti=True)
        self.quats_A = quaternions_A
        self.quats_B = quaternions_B
        # Compute the pair parameters and get the halfway transformation
        for idx,t in enumerate(zip(translations_A, quaternions_A, translations_B, quaternions_B)):
            self.pair_parameters[idx], self.trans[idx], self.quat[idx] = self.compute_parameters(t[0],t[1],t[2],t[3])

        # compute step paramters
        for idx in range(1,self.n_bp):
            self.step_parameters[idx] , _, _ = self.compute_parameters(self.trans[idx], self.quat[idx], self.trans[idx-1], self.quat[idx-1])


    def get_parameters(self,step=False,pair=False):
        """Return the computed parameters of shape (n_base_pairs, n_frames, n_parameters)"""
        step_parameter_names = ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']
        base_parameter_names = ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']

        if step and not pair:
            return self.step_parameters, step_parameter_names
        elif pair and not step:
            return self.pair_parameters, base_parameter_names
        elif not step and not pair:
            return np.concatenate((self.pair_parameters, self.step_parameters),axis=2).swapaxes(0,1), base_parameter_names + step_parameter_names
        








    def analyze_vec(self):

        """Process the base frames and convert them to translation and quaternion representations."""

        # Process the leading chain
        translations_A, quaternions_A = self.process_chain(self.res_A)
        translations_A = translations_A.reshape(-1,3)
        quaternions_A = quaternions_A.reshape(-1,4)

        # Process the lagging chain
        translations_B, quaternions_B = self.process_chain(self.res_B, anti=True)
        translations_B = translations_B.reshape(-1,3)
        quaternions_B = quaternions_B.reshape(-1,4)

        # Flip the B chain quaternion (connecting the backbones) and the (baseplane normals) (2,3)  vector b_L, b_N
        #quaternions_B = quaternions_B * qt.array([0, 1, 0, 0])
        
        # Compute the rigid body parameters and the mid/halfway transformation
        pair_parameters, trans_step, quat_step = self.compute_parameters(translations_A, quaternions_A, translations_B, quaternions_B) 
        self.pair_parameters = pair_parameters.reshape(self.traj.n_frames, self.n_bp, 6).swapaxes(0,1)

        # reshape output and input for the step parameters computation
        trans_step = trans_step.reshape(self.traj.n_frames,self.n_bp,3)
        quat_step = quat_step.reshape(self.traj.n_frames,self.n_bp,4)#.swapaxes(0,1)
        #quat_step = qt.unflip_rotors(quat_step,axis=1)
        
        # if quat_step.shape[1] > 1:
        #     for q in quat_step:
        #         qt.unflip_rotors(q, inplace=True)
        #     # quat_step = qt.unflip_rotors(quat_step)
        
        t_step_a = trans_step[:,:-1,].reshape(-1,3)
        t_step_b = trans_step[:,1:,].reshape(-1,3)
        q_step_a = quat_step[:,:-1,].reshape(-1,4)
        q_step_b = quat_step[:,1:,].reshape(-1,4)

        # Compute the step parameters and store them in the step_parameters attribute
        step_parameters,  _, _ = self.compute_parameters(t_step_b, q_step_b, t_step_a, q_step_a)
        extra_column = np.zeros((self.traj.n_frames, 1, 6))
        step_parameters = step_parameters.reshape(self.traj.n_frames, self.n_bp-1, 6)
        self.step_parameters = np.concatenate((extra_column,step_parameters), axis=1).swapaxes(0,1)


    # def compute_step_parameters(self):
    #     """Process the step transformations and convert them to translation and quaternion representations."""

    #     # trans_step_0 = self.trans_step[:-1]
    #     # quat_step_0 = self.quat_step[:-1]

    #     # trans_step_1 = self.trans_step[1:]
    #     # quat_step_1 = self.quat_step[1:]

    #     step_parameters,  _, _ = self.compute_parameters(self.trans_step[1:], self.quat_step[1:], self.trans_step[:-1], self.quat_step[:-1])
    #     step_parameters = step_parameters.reshape(self.traj.n_frames, self.n_bp-1, 6)
    #     # add empty column of zeros to the step parameters
    #     extra_column = np.zeros((step_parameters.shape[0], 1, 6))
    #     self.step_parameters = np.concatenate((extra_column,step_parameters), axis=1)

