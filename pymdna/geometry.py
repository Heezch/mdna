import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .utils import RigidBody, get_data_file_path

class ReferenceBase:

    def __init__(self, traj):
        self.traj = traj
        # Determine base type (purine/pyrimidine)
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
        # Identify whether base is a purine or pyrimidine based on presence of N1/N9
        for atom in self.traj.topology.atoms:
            if atom.name == "N1":
                return "pyrimidine"
            elif atom.name == "N9":
                return "purine"
        raise ValueError("Cannot determine the base type from the PDB file.")

    def get_coordinates(self):
        # Get the coordinates of key atoms based on the base type
        C1_coords = self._select_atom_by_name('"C1\'"')
        if self.base_type == "pyrimidine":
            N_coords = self._select_atom_by_name("N1")
            C_coords = self._select_atom_by_name("C2")
        elif self.base_type == "purine":
            N_coords = self._select_atom_by_name("N9")
            C_coords = self._select_atom_by_name("C4") # changed this for HS testing from C4 to C5
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
        bases = ['C', 'G', 'T', 'A']
        return {f'D{base}': md.load_pdb(get_data_file_path(f'./atomic/NDB96_{base}.pdb')) for base in bases}

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