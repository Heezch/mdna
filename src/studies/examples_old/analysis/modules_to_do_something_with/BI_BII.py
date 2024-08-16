import mdtraj as md
import numpy as np

def load_trajectory_and_slice_dna(traj):
    """ Load trajectory's topology and slice DNA part """
    dna = traj.atom_slice(traj.top.select('resname DG DC DA DT'))
    return dna
    
def get_backbone_indices(dna, chainid, ref_atoms):

    indices = []
    # find torsions based on the epsilon and zeta atoms
    # finally map the torsions for all base steps 
    if chainid == 0:
        residues = dna.top._chains[chainid].residues
    else:
        residues = dna.top._chains[chainid]._residues
        
    for res in residues:
        for at in res.atoms:
            if at.name in ref_atoms:
                indices.append(at)
    return indices

def get_torsions(indices, ref_atoms):
    # Find the chunks based on ref_atoms
    torsions = []
    i = 0
    while i < len(indices):
        ref = [at.name for at in indices[i:i+len(ref_atoms)]]
        if ref == ref_atoms:
            torsions.append(indices[i:i+len(ref_atoms)])
            i += len(ref_atoms)
        else:
            i += 1
    return torsions

def get_torsion_indices(dna, chainid, ref_atoms):
    indices = get_backbone_indices(dna, chainid, ref_atoms)
    torsions = get_torsions(indices, ref_atoms)
    return torsions

def convert_torsion_indices_to_atom_indices(torsion_indices):
    atom_indices = []
    for torsion in torsion_indices:
        atom_indices.append([at.index for at in torsion])
    return atom_indices

def compute_BI_BII(traj,degrees=True):

    epsilon_atoms = ["C4'","C3'","O3'","P"] 
    zeta_atoms = ["C3'","O3'","P","O5'"]

    dna = load_trajectory_and_slice_dna(traj)

    epsi_0 = get_torsion_indices(dna, 0, epsilon_atoms)
    epsi_1 = get_torsion_indices(dna, 1, epsilon_atoms)
    zeta_0 = get_torsion_indices(dna, 0, zeta_atoms)
    zeta_1 = get_torsion_indices(dna, 1, zeta_atoms)

    print(len(epsi_0), len(epsi_1), len(zeta_0), len(zeta_1))

    e_torsion_indices = convert_torsion_indices_to_atom_indices(epsi_1)
    z_torsion_indices = convert_torsion_indices_to_atom_indices(zeta_1)

    epsi = md.compute_dihedrals(dna, e_torsion_indices)
    zeta = md.compute_dihedrals(dna, z_torsion_indices)

    if degrees:
        epsi = np.degrees(epsi)
        zeta = np.degrees(zeta)

    print(epsi.shape, zeta.shape)
    return epsi, zeta