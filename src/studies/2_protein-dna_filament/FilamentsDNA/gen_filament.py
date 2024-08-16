import numpy as np
import seaborn as sns
import mdtraj as md

# https://biopython.org/docs/1.74/api/Bio.SVDSuperimposer.html
from Bio.SVDSuperimposer import SVDSuperimposer
from numpy import array, dot, set_printoptions


def get_rot_and_trans(subtraj_A,subtraj_B):
    
    """ fit only works now on a single frame (mdtraj returns xyz with shape (n_frames, atoms, xyz) 
         even for single frame trajs so hence the xyz[0]"""
    
    # load super imposer
    sup = SVDSuperimposer()

    # Set the coords, y will be rotated and translated on x
    x = subtraj_A.xyz[0]
    y = subtraj_B.xyz[0]
    sup.set(x, y)

    # Do the leastsquared fit
    sup.run()

    # Get the rms
    rms = sup.get_rms()

    # Get rotation (right multiplying!) and the translation
    rot, tran = sup.get_rotran()
    
    # now we have the instructions to rotate B on A
    return rot,tran,rms

def apply_superimposition(traj, rot, tran):
    
    # get xyz coordinates
    xyz = traj.xyz[0]
    
    # rotate subject on target
    new_xyz = dot(xyz, rot) + tran

    # replace coordinates of traj
    traj.xyz = new_xyz
    return traj

def fit_B_on_A(A, B, overlap_A, overlap_B):
    # create trajs containing only the selections
    subtraj_A = A.atom_slice(overlap_A)
    subtraj_B = B.atom_slice(overlap_B)

    # obtain instructions to rotate and translate B on A based on substraj structures
    rot, tran, rms = get_rot_and_trans(subtraj_A,subtraj_B)

    # do the superimposition of B on A and subsitute old with new xyz of B
    return apply_superimposition(B, rot, tran)
    

def get_overlap_indices(top,n,chain=0,terminus=None):
    residues = np.array(top._chains[chain]._residues)
    if terminus == 'N_terminus': # get residues at end of chain
        s = residues[len(residues)-n*2:len(residues)]
        return [at.index for res in s for at in res.atoms]
    elif terminus == 'C_terminus': # get residues at beginning of chain
        s = residues[:n*2]
        return [at.index for res in s for at in res.atoms]
    else:
        print('No terminus')
        
def check_if_dimerization(site):
    if 's' in site:
        return True
    else:
        return False
    
def get_termini(site_x,site_y):
    chain_order = np.array(['s1','h3','s2','l2','dbd'])
    x = np.argwhere(chain_order==site_x)
    y = np.argwhere(chain_order==site_y)
    if x < y:
        return ['N_terminus','C_terminus']
    elif x > y:
        return ['C_terminus','N_terminus']

def check_overlaps(overlap_A,overlap_B):

    if len(overlap_A) != len(overlap_B):
        print(len(overlap_A),len(overlap_B))
        print('Something went wrong with finding the overlaps') 
    else:
        False

def remove_overlap(traj,overlap):
     return traj.atom_slice([at.index for at in traj.top.atoms if at.index not in overlap])
    
def split_chain_topology(traj,leading_chain):
    # split part of A in chain that is being extended and that is not
    traj_active = traj.atom_slice(traj.top.select(f'chainid {leading_chain}'))
    traj_passive = traj.atom_slice(traj.top.select(f'not chainid {leading_chain}'))
    return traj_active, traj_passive

def merge_chain_topology(A,B,keep_resSeq=True):
    C = A.stack(B,keep_resSeq=keep_resSeq)
    top = C.top
    # Merge two tops (with two chains or more) to a top of one chain 
    out = md.Topology()
    c = out.add_chain()
    for chain in top.chains:

        for residue in chain.residues:
            r = out.add_residue(residue.name, c, residue.resSeq, residue.segment_id)
            for atom in residue.atoms:
                out.add_atom(atom.name, atom.element, r, serial=atom.serial)
    #     for bond in top.bonds:
    #         a1, a2 = bond
    #         out.add_bond(a1, a2, type=bond.type, order=bond.order)
    out.create_standard_bonds() #rare manier om bonds te maken, maar werkt
    C.top = out 
    return C