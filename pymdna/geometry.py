import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import quaternionic as qt
from .utils import RigidBody, get_data_file_path, get_sequence_letters

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

    def __init__(self, traj):
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
        # Select an atom by name returns shape (n_frames, 1, [x,y,z])
        return np.squeeze(self.traj.xyz[:,[self.traj.topology.select(f'name {name}')[0]],:],axis=1)
        
    def get_base_type(self):
        # Extracts all non-hydrogen atoms from the trajectory topology
        atoms = {atom.name for atom in self.traj.topology.atoms if atom.element.symbol != 'H'}
    
        # Check each base in the dictionary to see if all its atoms are present in the extracted atoms set
        for base, base_atoms in NUCLEOBASE_DICT.items():
            if all(atom in atoms for atom in base_atoms):
                return base
        # If no base matches, raise an error
        raise ValueError("Cannot determine the base type from the PDB file.")
    
    def get_coordinates(self):

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

class NucleicFrames_row:

    """
    Example Usage:
    
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

    def __init__(self, traj, chainids=[0,1],frames_only=False):
        self.traj = traj
        self.top = traj.topology
        self.res_A = self.get_residues(chain_index=chainids[0], reverse=False)
        self.res_B = self.get_residues(chain_index=chainids[1], reverse=True)
        self.mean_reference_frames = np.empty((len(self.res_A), 1, 4, 3))
        self.frames = self.get_reference_frames()
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
    
    def get_reference_frames(self):
        """Get reference frames for each residue."""
        reference_frames = {} # Dictionary to store the base vectors for each residue
        for res in self.res_A + self.res_B:
            res_traj = self.traj.atom_slice([at.index for at in res.atoms])
            base_vectors = self.get_base_vectors(res_traj)
            reference_frames[res] = base_vectors # Store the base vectors for the residue index (with shape (4, n_frames, 3))
        return reference_frames

    def reshape_input(self,input_A,input_B,is_step=False):
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

    def compute_rotation_matrices(self,rotation_A, rotation_B):
        # Compute rotation matrices
        rotation_matrix_AB = rotation_B.transpose(0,2,1) @ rotation_A  # returns shape (n, 3, 3)
        rotation_matrix_BA = rotation_B @ rotation_A.transpose(0,2,1)  # returns shape (n, 3, 3)
        return rotation_matrix_AB, rotation_matrix_BA
    
    def get_rotation_angles(self,rotation_matrix_AB, rotation_matrix_BA):
        # Get rotation angles based on rotation matrices
        return RigidBody.extract_omega_values(rotation_matrix_AB), RigidBody.extract_omega_values(rotation_matrix_BA)

    def get_mid_frame(self,rotation_angle_BA, rotation_A, origin_A, origin_B):
        # Compute halfway rotation matrix and triad (mid frame)
        halfway_rotation_matrix = RigidBody.get_rotation_matrix(rotation_angle_BA * 0.5)
        mid_frame_rotation = halfway_rotation_matrix @ rotation_A
        mid_frame_origin = 0.5 * (origin_A + origin_B)

        return mid_frame_rotation, mid_frame_origin

    def compute_params(self,mid_frame_rotation, origin_A, origin_B, rotation_angle_AB, original_shape, is_step=False):

        # Compute rotation matrix based on mid frame and translational vector based off mid frame
        rotation_mid_frame = np.sum(rotation_angle_AB[:, None] * mid_frame_rotation, axis=2)
        translational_vector_mid_frame = np.einsum('ijk,ik->ij', (origin_B - origin_A)[:, None] * mid_frame_rotation, np.ones((mid_frame_rotation.shape[0], 3)))
        
        # Fix direction of magnitudes and convert to degrees (if taking the negative of halfway_rotation_matrix, then it doesn't need to be multiplied by -1, but that screws with the orientation of the mean reference mid frame)
        #sign = 1 #if is_step else -1  # for the base pair paramters 
        sign = 1
        # concatenate the translational and rotational vectors and multiply by 10 to convert to angstroms and degrees
        params = sign*np.concatenate((translational_vector_mid_frame*10, np.rad2deg(rotation_mid_frame)), axis=1)
        return params.reshape(original_shape[0], original_shape[1], 6).swapaxes(0, 1)

    def calculate_parameters(self,frames_A, frames_B, is_step=False):

        # Reshape frames
        rotation_A, rotation_B, origin_A, origin_B, original_shape = self.reshape_input(frames_A,frames_B, is_step=is_step)

        # Compute rotation matrices
        rotation_matrix_AB, rotation_matrix_BA = self.compute_rotation_matrices(rotation_A, rotation_B)

        # Get rotation angles based on rotation matrices
        rotation_angle_AB, rotation_angle_BA = self.get_rotation_angles(rotation_matrix_AB, rotation_matrix_BA)

        # Compute halfway rotation matrix and triad (mid frame)
        mid_frame_rotation, mid_frame_origin = self.get_mid_frame(rotation_angle_BA, rotation_A, origin_A, origin_B)

        # Collect mean reference frames from mid frames of each base pair
        mean_reference_frames = np.hstack((mid_frame_origin[:, np.newaxis, :],mid_frame_rotation)).reshape(original_shape)

        # Compute parameters
        params = self.compute_params(mid_frame_rotation, origin_A, origin_B, rotation_angle_AB, original_shape, is_step=is_step)

        if is_step:
            # Creating an array of zeros with shape (10000, 1, 6)
            extra_column = np.zeros((params.shape[0], 1, 6))

            # Concatenating the existing array and the extra column along the second axis
            params = np.concatenate((extra_column,params), axis=1)

        return  params, mean_reference_frames if not is_step else params

    def analyse_frames(self):
        """Analyze the trajectory and compute parameters."""

        # Get base reference frames for each residue
        frames_A = np.array([self.frames[res] for res in self.res_A])
        frames_B = np.array([self.frames[res] for res in self.res_B])

        # Compute parameters between each base pair and mean reference frames
        self.bp_params, self.mean_reference_frames = self.calculate_parameters(frames_A, frames_B)
        
        # Extract mean reference frames for each neighboring base pair
        B1_triads = self.mean_reference_frames[:-1] # select all but the last frame
        B2_triads = self.mean_reference_frames[1:] # select all but the first frame

        # Compute parameters between each base pair and mean reference frames
        self.step_params = self.calculate_parameters(B1_triads, B2_triads, is_step=True)[0]

    def get_parameters(self,step=False,base=False):
        """Return the computed parameters of shape (n_frames, n_base_pairs, n_parameters)"""
        step_parameter_names = ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']
        base_parameter_names = ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']

        if step and not base:
            return self.step_params, step_parameter_names
        elif base and not step:
            return self.bp_params, base_parameter_names
        elif not step and not base:
            return np.dstack((self.bp_params, self.step_params)), base_parameter_names + step_parameter_names
        


class NucleicFrames:

    """
    Example Usage:
    
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

    def __init__(self, traj, chainids=[0,1],frames_only=False):
        self.traj = traj
        self.top = traj.topology
        self.res_A = self.get_residues(chain_index=chainids[0], reverse=False)
        self.res_B = self.get_residues(chain_index=chainids[1], reverse=True)
        self.mean_reference_frames = np.empty((len(self.res_A), 1, 4, 3))
        self.frames = self.get_reference_frames()
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
    
    def get_reference_frames(self):
        """Get reference frames for each residue."""
        reference_frames = {} # Dictionary to store the base vectors for each residue
        for res in self.res_A + self.res_B:
            res_traj = self.traj.atom_slice([at.index for at in res.atoms])
            base_vectors = self.get_base_vectors(res_traj)
            reference_frames[res] = base_vectors # Store the base vectors for the residue index (with shape (4, n_frames, 3))
        return reference_frames

    def reshape_input(self,input_A,input_B,is_step=False):
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

    def compute_rotation_matrices(self,rotation_A, rotation_B):
        # Compute rotation matrices
        rotation_matrix_AB = rotation_B.transpose(0,2,1) @ rotation_A  # returns shape (n, 3, 3)
        rotation_matrix_BA = rotation_B @ rotation_A.transpose(0,2,1)  # returns shape (n, 3, 3)
        return rotation_matrix_AB, rotation_matrix_BA
    
    def get_rotation_angles(self,rotation_matrix_AB, rotation_matrix_BA):
        # Get rotation angles based on rotation matrices
        return RigidBody.extract_omega_values(rotation_matrix_AB), RigidBody.extract_omega_values(rotation_matrix_BA)

    def get_mid_frame(self,rotation_angle_BA, rotation_A, origin_A, origin_B):
        # Compute halfway rotation matrix and triad (mid frame)
        halfway_rotation_matrix = RigidBody.get_rotation_matrix(rotation_angle_BA * 0.5)
        mid_frame_rotation = halfway_rotation_matrix @ rotation_A
        mid_frame_origin = 0.5 * (origin_A + origin_B)

        return mid_frame_rotation, mid_frame_origin

    def compute_params(self,mid_frame_rotation, origin_A, origin_B, rotation_angle_AB, original_shape, is_step=False):

        # Compute rotation matrix based on mid frame and translational vector based off mid frame
        if not is_step:
            rotation_mid_frame = np.sum(-rotation_angle_AB[:, None] * mid_frame_rotation, axis=2)
        else:
            rotation_mid_frame = np.sum(rotation_angle_AB[:, None] * mid_frame_rotation, axis=2)

        if not is_step:
            translational_vector_mid_frame = np.einsum('ijk,ik->ij', (origin_A - origin_B)[:, None] * mid_frame_rotation, np.ones((mid_frame_rotation.shape[0], 3)))
        else:
            translational_vector_mid_frame = np.einsum('ijk,ik->ij', (origin_B - origin_A)[:, None] * mid_frame_rotation, np.ones((mid_frame_rotation.shape[0], 3)))
        
        # Fix direction of magnitudes and convert to degrees (if taking the negative of halfway_rotation_matrix, then it doesn't need to be multiplied by -1, but that screws with the orientation of the mean reference mid frame)
        #sign = 1 #if is_step else -1  # for the base pair paramters 
        sign = 1
        # concatenate the translational and rotational vectors and multiply by 10 to convert to angstroms and degrees
        params = sign*np.concatenate((translational_vector_mid_frame*10, np.rad2deg(rotation_mid_frame)), axis=1)
        return params.reshape(original_shape[0], original_shape[1], 6).swapaxes(0, 1)

    def calculate_parameters(self,frames_A, frames_B, is_step=False):

        # Reshape frames
        rotation_A, rotation_B, origin_A, origin_B, original_shape = self.reshape_input(frames_A,frames_B, is_step=is_step)

        # Compute rotation matrices
        rotation_matrix_AB, rotation_matrix_BA = self.compute_rotation_matrices(rotation_A, rotation_B)

        # Get rotation angles based on rotation matrices
        rotation_angle_AB, rotation_angle_BA = self.get_rotation_angles(rotation_matrix_AB, rotation_matrix_BA)

        # Compute halfway rotation matrix and triad (mid frame)
        mid_frame_rotation, mid_frame_origin = self.get_mid_frame(rotation_angle_BA, rotation_A, origin_A, origin_B)

        # Collect mean reference frames from mid frames of each base pair
        mean_reference_frames = np.hstack((mid_frame_origin[:, np.newaxis, :],mid_frame_rotation)).reshape(original_shape)

        # Compute parameters
        params = self.compute_params(mid_frame_rotation, origin_A, origin_B, rotation_angle_AB, original_shape, is_step=is_step)

        if is_step:
            # Creating an array of zeros with shape (10000, 1, 6)
            extra_column = np.zeros((params.shape[0], 1, 6))

            # Concatenating the existing array and the extra column along the second axis
            params = np.concatenate((extra_column,params), axis=1)

        return  params, mean_reference_frames if not is_step else params

    def analyse_frames(self):
        """Analyze the trajectory and compute parameters."""

        # Get base reference frames for each residue
        frames_A = np.array([self.frames[res] for res in self.res_A])
        frames_B = np.array([self.frames[res] for res in self.res_B])

        # Compute parameters between each base pair and mean reference frames
        self.bp_params, self.mean_reference_frames = self.calculate_parameters(frames_A, frames_B)
        
        # Extract mean reference frames for each neighboring base pair
        B1_triads = self.mean_reference_frames[:-1] # select all but the last frame
        B2_triads = self.mean_reference_frames[1:] # select all but the first frame

        # Compute parameters between each base pair and mean reference frames
        self.step_params = self.calculate_parameters(B1_triads, B2_triads, is_step=True)[0]

    def get_parameters(self,step=False,base=False):
        """Return the computed parameters of shape (n_frames, n_base_pairs, n_parameters)"""
        step_parameter_names = ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']
        base_parameter_names = ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']

        if step and not base:
            return self.step_params, step_parameter_names
        elif base and not step:
            return self.bp_params, base_parameter_names
        elif not step and not base:
            return np.dstack((self.bp_params, self.step_params)), base_parameter_names + step_parameter_names
        

import copy

class NucleicFrames_column:

    """
    Example Usage:
    
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

    def __init__(self, traj, chainids=[0,1],frames_only=False):
        self.traj = traj
        self.top = traj.topology
        self.res_A = self.get_residues(chain_index=chainids[0], reverse=False)
        self.res_B = self.get_residues(chain_index=chainids[1], reverse=True)
        self.mean_reference_frames = np.empty((len(self.res_A), 1, 4, 3))
        self.frames = self.get_reference_frames()
        self.analyse_frames()
        self.angles = []

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
    
    def get_reference_frames(self):
        """Get reference frames for each residue."""
        reference_frames = {} # Dictionary to store the base vectors for each residue
        for res in self.res_A + self.res_B:
            res_traj = self.traj.atom_slice([at.index for at in res.atoms])
            base_vectors = self.get_base_vectors(res_traj)
            reference_frames[res] = base_vectors # Store the base vectors for the residue index (with shape (4, n_frames, 3))
        return reference_frames

    def reshape_input(self,input_A,input_B,is_step=False):
        # Store original shape
        original_shape = input_A.shape

        # Flatten frames to compute rotation matrices for each time step simultaneously
        input_A_ = input_A.reshape(-1,original_shape[-2],original_shape[-1])  # shape (n, 4, 3)
        input_B_ = input_B.reshape(-1,original_shape[-2],original_shape[-1])  # shape (n, 4, 3)

        # Extract origins of triads
        origin_A = input_A_[:,0]  # shape (n, 3)
        origin_B = input_B_[:,0]  # shape (n, 3)

        # Extract the triads without origin (rotation matrices) and switch rows with columns
        rotation_A = input_A_[:,1:].transpose(0, 2, 1)  # shape (n, 3, 3) 
        rotation_B = input_B_[:,1:].transpose(0, 2, 1)  # shape (n, 3, 3) 

        if not is_step:
            # flip (connecting the backbones) and the (baseplane normals).
            # so the second and third column vector b_L, b_N
            rotation_B[:, :, [1,2]] *= -1
            #rotation_B[:, [1, 2]] *= -1

        return rotation_A, rotation_B, origin_A, origin_B, original_shape

    def compute_rotation_matrices(self,rotation_A, rotation_B):
        # Compute rotation matrices
        rotation_matrix_BA = rotation_B.transpose(0,2,1) @ rotation_A  # returns shape (n, 3, 3) # 2.15
        rotation_matrix_AB = rotation_A.transpose(0,2,1) @ rotation_B # 2.21 
        return rotation_matrix_AB, rotation_matrix_BA # 2.15 corresponds to Da_bar.T Da = La # 2.15 average orientation of the two base frames
    
    def get_rotation_angles(self,rotation_matrix_AB, rotation_matrix_BA):
        # Get rotation angles based on rotation matrices
        return RigidBody.extract_omega_values(rotation_matrix_AB), RigidBody.extract_omega_values(rotation_matrix_BA)

    def get_mid_frame(self,rotation_angle, rotation_A, rotation_B, origin_A, origin_B, is_step=False):
        # Compute halfway rotation matrix and triad (mid frame)
        halfway_rotation_matrix = RigidBody.get_rotation_matrix(rotation_angle * 0.5)
        #mid_frame_rotation = halfway_rotation_matrix @ rotation_A
        if not is_step:
            mid_frame_rotation = rotation_B @ halfway_rotation_matrix  # 2.14  
        else:
            mid_frame_rotation = rotation_A @ halfway_rotation_matrix #  2.24 
     
        mid_frame_origin = 0.5 * (origin_A + origin_B) # 2.19 

        return mid_frame_rotation, mid_frame_origin

    def compute_params(self,mid_frame_rotation, origin_A, origin_B, rotation_angle_AB, rotation_angle_BA, original_shape, is_step=False):

        print(rotation_angle_BA.shape, mid_frame_rotation.shape)
        # Compute rotation matrix based on mid frame and translational vector based off mid frame
        #rotation_mid_frame = np.sum(rotation_angle_BA[:, None] * mid_frame_rotation, axis=2)
        if not is_step:
            rotation_mid_frame = np.einsum('ijk,ik->ij', mid_frame_rotation, rotation_angle_BA)
        else:
            rotation_mid_frame = np.einsum('ijk,ik->ij', mid_frame_rotation, rotation_angle_AB)

        # # Calculate the difference vector r^a - r'^a
        # vector_difference = origin_A - origin_B  # Shape: (n_samples, 3)
        print(origin_A.shape, origin_B.shape, mid_frame_rotation.shape)
        # Apply the transpose of the rotation matrix to the difference vector
        # translational_vector_mid_frame = np.einsum('ijk,ik->ij', (origin_B - origin_A)[:, None] * mid_frame_rotation, np.ones((mid_frame_rotation.shape[0], 3)))
        diff = origin_A - origin_B 
        translational_vector_mid_frame = np.einsum('ijk,ik->ij', mid_frame_rotation.transpose(0, 2, 1), diff) # 2.18
        # Fix direction of magnitudes and convert to degrees (if taking the negative of halfway_rotation_matrix, then it doesn't need to be multiplied by -1, but that screws with the orientation of the mean reference mid frame)
        #sign = 1 #if is_step else -1  # for the base pair paramters 
        sign = 1
        # concatenate the translational and rotational vectors and multiply by 10 to convert to angstroms and degrees
        params = sign*np.concatenate((translational_vector_mid_frame*10, np.rad2deg(rotation_mid_frame)), axis=1)
        return params.reshape(original_shape[0], original_shape[1], 6).swapaxes(0, 1)

    def calculate_parameters(self,frames_A, frames_B, is_step=False):

        # Reshape frames
        rotation_A, rotation_B, origin_A, origin_B, original_shape = self.reshape_input(frames_A,frames_B, is_step=is_step)

        # Compute rotation matrices
        rotation_matrix_AB, rotation_matrix_BA = self.compute_rotation_matrices(rotation_A, rotation_B)

        # Get rotation angles based on rotation matrices
        rotation_angle_AB, rotation_angle_BA = self.get_rotation_angles(rotation_matrix_AB, rotation_matrix_BA)

        # Compute halfway rotation matrix and triad (mid frame)
        if not is_step:
            mid_frame_rotation, mid_frame_origin = self.get_mid_frame(rotation_angle_BA, rotation_A, rotation_B, origin_A, origin_B, is_step=is_step)   
        else:
            mid_frame_rotation, mid_frame_origin = self.get_mid_frame(rotation_angle_AB, rotation_A, rotation_B, origin_A, origin_B, is_step=is_step)
     
        # Collect mean reference frames from mid frames of each base pair
        mean_reference_frames = np.hstack((mid_frame_origin[:, np.newaxis, :],mid_frame_rotation)).reshape(original_shape)

        # Compute parameters
        params = self.compute_params(mid_frame_rotation, origin_A, origin_B, rotation_angle_AB, rotation_angle_BA, original_shape, is_step=is_step)

        if is_step:
            # Creating an array of zeros with shape (10000, 1, 6)
            extra_column = np.zeros((params.shape[0], 1, 6))

            # Concatenating the existing array and the extra column along the second axis
            params = np.concatenate((extra_column,params), axis=1)

        return  params, mean_reference_frames if not is_step else params

    def analyse_frames(self):
        """Analyze the trajectory and compute parameters."""
        import copy
        # Get base reference frames for each residue
        frames_A = copy.deepcopy(np.array([self.frames[res] for res in self.res_A]))
        frames_B = copy.deepcopy(np.array([self.frames[res] for res in self.res_B]))

        # Compute parameters between each base pair and mean reference frames
        self.bp_params, self.mean_reference_frames = self.calculate_parameters(frames_A, frames_B)
        
        # Extract mean reference frames for each neighboring base pair
        B1_triads = self.mean_reference_frames[:-1] # select all but the last frame
        B2_triads = self.mean_reference_frames[1:] # select all but the first frame

        # Compute parameters between each base pair and mean reference frames
        self.step_params = self.calculate_parameters(B1_triads, B2_triads, is_step=True)[0]

    def get_parameters(self,step=False,base=False):
        """Return the computed parameters of shape (n_frames, n_base_pairs, n_parameters)"""
        step_parameter_names = ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']
        base_parameter_names = ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']

        if step and not base:
            return self.step_params, step_parameter_names
        elif base and not step:
            return self.bp_params, base_parameter_names
        elif not step and not base:
            return np.dstack((self.bp_params, self.step_params)), base_parameter_names + step_parameter_names
        


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
    

    def compute_parameters(self,translations_A, quaternions_A, translations_B, quaternions_B, t=0.5):
        """Compute the rigid body parameters between two frames."""
        #dot = np.dot(quaternions_A, quaternions_B)
        
        dot_products = (quaternions_A.w*quaternions_B.w) + (quaternions_A.x*quaternions_B.x) + (quaternions_A.y*quaternions_B.y) + (quaternions_A.z*quaternions_B.z)
        mask = dot_products < 0
        # quaternions_A[mask] = -quaternions_A[mask]

        # Get the relative rotation matrix
        A = quaternions_B.inverse * quaternions_A
        
        #A = quaternions_B.inverse*quaternions_A
        #A = self.compute_relative_rotation(quaternions_A, quaternions_B, mask)

        # Slerp (spherical linear interpolation) for quaternion
        # Note that output slerp(q1, q2, 1) may be different from q2. (slerp(q1, q2, 0) is always equal to q1.)
        quat_mid = qt.slerp(quaternions_A, quaternions_B, tau=t)
        #quat_mid = self.compute_midframe(quaternions_A, quaternions_B, mask)  

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
            pitch = np.arcsin(2.0*(A.w*A.y - A.x*A.z))
 
            # yaw = np.arctan2(2.0*(A.y*A.z + A.w*A.x), A.w*A.w - A.x*A.x - A.y*A.y + A.z*A.z)
            # roll = np.arctan2(2.0*(A.x*A.y + A.w*A.z), A.w*A.w + A.x*A.x - A.y*A.y - A.z*A.z)
            # 
            # # Initialize yaw and roll with default values that will apply when no specific condition is met
            yaw = np.zeros_like(pitch)
            roll = np.zeros_like(pitch)

            # Condition for pitch = pi/2
            mask_pi_2 = pitch == np.pi/2
            yaw[mask_pi_2] = -np.arctan2(A.x[mask_pi_2],A.w[mask_pi_2])
            roll[mask_pi_2] = 0.0

            # Condition for pitch = -pi/2
            mask_neg_pi_2 = pitch == -np.pi/2
            yaw[mask_neg_pi_2] = np.arctan2(A.x[mask_neg_pi_2],A.w[mask_neg_pi_2])
            roll[mask_neg_pi_2] = 0.0

            # Default condition (where neither pi/2 nor -pi/2 conditions are met)
            mask_else = ~(mask_pi_2 | mask_neg_pi_2)
            yaw[mask_else] = np.arctan2(2.0*(A.y[mask_else]*A.z[mask_else] + A.w[mask_else]*A.x[mask_else]),
                                        A.w[mask_else]**2 - A.x[mask_else]**2 - A.y[mask_else]**2 + A.z[mask_else]**2)
            roll[mask_else] = np.arctan2(2.0*(A.x[mask_else]*A.y[mask_else] + A.w[mask_else]*A.z[mask_else]),
                                        A.w[mask_else]**2 + A.x[mask_else]**2 - A.y[mask_else]**2 - A.z[mask_else]**2)

            rotational_parameters = np.vstack((yaw, pitch, roll)).swapaxes(0,1)

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
            rotational_parameters = A.to_axis_angle # unfortunately this results in angle wraps.

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

