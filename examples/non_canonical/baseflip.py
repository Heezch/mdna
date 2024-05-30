import numpy as np
from utils import RigidBody
import mdtraj as md

def select_atom_by_name(traj, name):
    # Select an atom by name returns shape (n_frames, 1, [x,y,z])
    return np.squeeze(traj.xyz[:,[traj.topology.select(f'name {name}')[0]],:],axis=1)

# Load the base A from the atomic library
loc = '../../pymdna/atomic/'
base_A = md.load(loc+'BDNA_G.pdb')

top = base_A.topology

# Get the coordinates of the atoms involved in the rotation
c1_prime_coords = select_atom_by_name(base_A, '"C1\'"')
n9_coords = select_atom_by_name(base_A, "N9")

# Calculate the Euler vector for the 180-degree rotation around the specified axis and normalize the axis vector
rotation_axis = c1_prime_coords - n9_coords
rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

# Set the rotation angle to 180 degrees
theta = np.deg2rad(180)  # Convert angle to radians

# Update the xyz of the nucleobase in base_A.xyz using the rotation
nucleobase_selection = top.select("(name =~ 'N[1-9]') or (name C1 C2 C3 C4 C5 C6 C7 C8 C9) or name O6")
relative_positions = base_A.xyz[:, nucleobase_selection, :] - n9_coords[:, None, :]

# Apply the rotation to each atom's relative position
rotated_positions = np.array([RigidBody.rotate_vector(v, rotation_axis[0], theta) for v in relative_positions[0]])

# Translate the rotated positions back to the original coordinate system
new_xyz = rotated_positions + n9_coords[:, None, :]

# Update the coordinates in the trajectory
base_A.xyz[:, nucleobase_selection, :] = new_xyz

# Proceed with visualization and saving...
base_A.save('BDNA_A_rotated.pdb')
