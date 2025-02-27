import copy
from .utils import get_base_pair_dict, RigidBody, get_data_file_path
import numpy as np
from .geometry import ReferenceBase
import mdtraj as md
import copy
from mdtraj.core import element as elem



class Methylate:
    """
    Class to add methyl groups to the DNA structure
    Methylate DNA currently only does C and G methylation (no A or T) and G at the O6 oxygen and C at the C5 carbon
    In principle we could add an extra requirement for C to only methylate when it is in a CpG context

    Example:
    # This methylates the first base in the pdb file 
    methylator = mdna.Methylate(pdb, methylations=[0])
    methylated_traj = methylator.get_traj()
    """
    def __init__(self, traj, methylations=None, CpG=False, leading_strand=None):
        self.traj = copy.deepcopy(traj)
        if CpG and leading_strand is not None :
            print('Methylate all C in CpG context, superseeds methylations list.')
            self.baselist = self.find_CpGs(leading_strand)
            print('Methtylating:',self.baselist)
        elif CpG and leading_strand is None:
             print("Please provide chainid of `leading_strand` as argument.")
        elif len(methylations) is not None:
            self.baselist = methylations # List of resids that need to be methylated (so far only C and G)
        else:
            ValueError("Please provide either a list of resids to methylate or set CpG to True with chainid of `leading_strand` as argument.")
        self.apply_methylation()
    
    def apply_methylation(self):

        # For each residue that needs to be mutated
        for resid in self.baselist:
            
            # Current residue
            residue = self.traj.top.residue(resid)
            
            # Get residue code
            code = residue.name[1] # Assume DC DG DT DA 

            # Get dock and ref atom to add carbon methyl
            a, b  = self.get_atoms(residue, code)

            # Filter such that only C and G can get methylated
            if a is not None and b is not None:
                self.add_methyl(a, b, residue, code) # Add the methyl group to the residue 
                self.traj.top.residue(resid).name = f'D{code}M' # Update the residue name to reflect the mutation
            else:
                print(f"Residue {residue} with methylations index {resid} could not be methylated.")


    def find_CpGs(self,leading_strand=0):
        sequence = ''.join([res.name[1] for res in self.traj.top.chain(leading_strand)._residues])
        return [i for i in range(len(sequence) - 1) if sequence[i:i+2] == 'CG']

    def get_atoms(self, residue, code):
        # Get the atoms to add the methyl group to
        atom_a, atom_b  = None, None
        for at in residue.atoms:
            if str(at.name) == 'C5' and code == 'C': # The anchor atom for the methyl group (C5)
                atom_a = at
            elif str(at.name) == 'C2' and code == 'C': # Reference atom to create vector
                atom_b = at
            elif str(at.name) == 'O6' and code == 'G': # The anchor atom for the methyl group (C6)
                atom_a = at
            elif str(at.name) == 'N2' and code == 'G': # Reference atom to create vector 
                atom_b = at

        return atom_a, atom_b

    def add_methyl(self, a, b, residue, code):

        # Get the index of the atom to insert the new atom after
        index = a.index+1

        # Calculate new position with displacement of 0.138 nm in direction of a-b
        vec = self.traj.xyz[:,a.index] - self.traj.xyz[:,b.index]
        vec /= np.linalg.norm(vec)
        new_pos = self.traj.xyz[:,a.index] + 0.138 * vec
        
        # split xyz old in two at where the new atom will be inserted
        xyz1 = self.traj.xyz[:,:index,:]
        xyz2 = self.traj.xyz[:,index:,:]

        # Determine name
        if code == 'C':
            name = 'C5M' # Not cetain about this name
        elif code == 'G':
            name = 'C6M' # Nor this one hahaha

        # Insert atom 
        offset = residue._atoms[0].index # Get the index of the first atom in the residue
        self.traj.top.insert_atom(name=name, element=elem.carbon, residue=residue, index=index, rindex=index-offset, serial=None)
        # stack the two xyz arrays together with the new_pos in between
        self.traj.xyz = np.concatenate([xyz1, new_pos[:,None,:], xyz2], axis=1)
        
 


    def get_traj(self):
        return self.traj
    

class Hoogsteen:
    """ Hoogsteen base pair flip"""
    # should still update for all the other bases (non-canonical)
    def __init__(self, traj, fliplist, deg=180,verbose=False):
        self.traj = copy.deepcopy(traj)
        self.verbose = verbose
        self.fliplist = fliplist # List of resids that need to be flipped
        self.theta = np.deg2rad(deg) # Set the rotation angle to 180 degrees
        self.apply_flips()
        if verbose:
            print(f"Flipped residues {self.fliplist} by {self.theta} radians")

    def select_atom_by_name(self, name):
        # Select an atom by name returns shape (n_frames, 1, [x,y,z])
        return np.squeeze(self.traj.xyz[:,[self.traj.topology.select(f'name {name}')[0]],:],axis=1)


    def get_base_type(self, indices):
        # Identify whether base is a purine or pyrimidine based on presence of N1/N9
        res = self.traj.atom_slice(indices)
        for atom in res.topology.atoms:
            if atom.name == "N1":
                return "pyrimidine"
            elif atom.name == "N9":
                return "purine"
        raise ValueError("Cannot determine the base type from the PDB file.")

    def get_coordinates(self, base_type, resid):
        # Get the coordinates of key atoms based on the base type
        C1_coords = self.select_atom_by_name(f'"C1\'" and resid {resid}')
        if base_type == "pyrimidine":
            N_coords = self.select_atom_by_name(f"N1 and resid {resid}")
            # C_coords = self._select_atom_by_name("C2")
        elif base_type == "purine":
            N_coords = self.select_atom_by_name(f"N9 and resid {resid}")
            #C_coords = self.select_atom_by_name("C4") # changed this for HS testing from C4 to C5
        return C1_coords, N_coords #, C_coords

    def get_base_indices(self, traj, resid=0):

        # # Define the atoms that belong to the nucleotide
        # base_atoms = { 'C2','C4','C5','C6','C7','C8','C5M',
        #                'N1','N2','N3','N4','N6','N7','N9',
        #                'O2','O4','O6',
        #                'H1','H2','H3','H5','H6','H8',
        #                'H21','H22','H41','H42','H61','H62','H71','H72','H73'}
            
            # 'N9', 'N7', 'C8', 'C5', 'C4', 'N3', 'C2', 'N1', 
            #         'C6', 'C7','O6', 'N2', 'N6', 'O2', 'N4', 'O4', 'C5M',
            #         'H1','H2','H21','H22','H3','H41','H42','H5','H6','H61','H62','H71','H72','H73','H8'}
        # Select atoms that belong to the specified residue
        indices = traj.top.select(f'resid {resid}')
        offset = indices[0]  # Save the initial index of the residue
    
        # Create a subtrajectory containing only the specified residue
        subtraj = traj.atom_slice(indices)

        # Select the atoms that belong to the nucleotide
        #sub_indices = subtraj.top.select(f'name {" ".join(base_atoms)}')
        sub_indices = [atom.index for atom in subtraj.top.atoms if '\'' not in atom.name and 'P' not in atom.name]
        # Return the indices of the atoms that belong to the nucleotide
        return sub_indices + offset
    
    def apply_flips(self):
        # For each residue that needs to be mutated
        for resid in self.fliplist:
            
            # Get the indices of the atoms that need to be transformed
            nucleobase_selection = self.get_base_indices(self.traj, resid=resid)
            base_type = self.get_base_type(nucleobase_selection)

            # Get the coordinates of the atoms involved in the rotation
            # c1_prime_coords = self.select_atom_by_name(self.traj, f'"C1\'" and resid {resid}')
            # n9_coords = self.select_atom_by_name(self.traj, f"N9 and resid {resid}")
            
            c1_prime_coords, n_coords = self.get_coordinates(base_type, resid)
            # Calculate the Euler vector for the 180-degree rotation around the specified axis and normalize the axis vector
            rotation_axis = c1_prime_coords - n_coords
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Update the xyz of the nucleobase in base_A.xyz using the rotation
            relative_positions = self.traj.xyz[:, nucleobase_selection, :] - n_coords[:, None, :]

            # Apply the rotation to each atom's relative position
            rotated_positions = np.array([RigidBody.rotate_vector(v, rotation_axis[0], self.theta) for v in relative_positions[0]])

            # Translate the rotated positions back to the original coordinate system
            new_xyz = rotated_positions + n_coords[:, None, :]

            # Update the coordinates in the trajectory
            self.traj.xyz[:, nucleobase_selection, :] = new_xyz

    def get_traj(self):
        return self.traj


class Mutate:
    """ Class to mutate a DNA structure
    """
    def __init__(self, traj, mutations, complementary=True, verbose=False):
    
        self.traj = traj
        self.complementary = complementary
        self.verbose = verbose
        self.mutations = mutations
        self.current_resid = 0
        print('hello')
        self.mutate()
        
    def mutate(self):

        # Define the base pair map and the complementary mutant map
        #base_pair_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C','P':'T','D':'C'}
        base_pair_map = {'A':'T','T':'A','G':'C','C':'G','U':'A','D':'G','E':'T','L':'M','M':'L','B':'S','S':'B','Z':'P','P':'Z'}
        if self.complementary:
            # Update dict with the complementary mutations
            self.mutations = self.make_complementary_mutations(self.traj, self.mutations, base_pair_map)
    
        # TODO should also take into account the shift in atom indices due to atoms that are not deleted so that the new atoms are inserted at the correct position
        # Apply the mutations
        self.mutant_traj = self.apply_mutations(self.traj, self.mutations, base_pair_map)
        # Make sure indexing is also continious after mutation
        # Aka fix atom_indices are not monotonically increasing


    def get_base_indices(self, base_traj, resid=0):

        # Define the atoms that belong to the nucleotide
        #     base_atoms = { 'C2','C4','C5','C6','C7','C8','C5M',
        #                    'N1','N2','N3','N4','N6','N7','N9',
        #                    'O2','O4','O6',
        #                    'H1','H2','H3','H5','H6','H8',
        #                    'H21','H22','H41','H42','H61','H62','H71','H72','H73'}

        #    # noncanon_base_atoms = {'N9','C8',  'H8',  'N7',  'C5', 'C6', 'N7', 'N61', 'N62', 'N1', 'C2', 'H2', 'N3','C4','N1' ,'C2', 'O2', 'N3', 'C4', 'C6', 'C14', 'C13', 'N5', 'C11', 'S12', 'C7', 'C8', 'C9', 'C10'} # TC1
            
        base_atoms = {'C2','C4','C5','C6','C7','C8','C5M',
                       'N1','N2','N3','N4','N6','N7','N9',
                       'O2','O4','O6',
                       'H1','H2','H3','H5','H6','H8',
                       'H21','H22','H41','H42','H61','H62','H71','H72','H73','N9','C8','H8',  
                       'N7','C5','C6', 'N7', 'N61', 'N62','N1','C2','H2','N3','C4','N1','C2', 
                       'O2','N3','C4','C6','C14','C13','N5','C11','S12','C7','C8','C9','C10'} # TC1

        # Select atoms that belong to the specified residue
        indices = base_traj.top.select(f'resid {resid}')
        offset = indices[0]  # Save the initial index of the residue
    
        # Create a subtrajectory containing only the specified residue
        subtraj = base_traj.atom_slice(indices)
        
        # Select the atoms that belong to the nucleotide
        sub_indices = [atom.index for atom in subtraj.top.atoms if '\'' not in atom.name and 'P' not in atom.name]
        # Return the indices of the atoms that belong to the nucleotide
        return sub_indices + offset


    def make_complementary_mutations(self, traj, mutations, base_pair_map):
        
        # Get the basepair dictionary of the trajectory
        basepair_dict = get_base_pair_dict(traj)

        # Iterate over a static list of dictionary items to avoid RuntimeError
        for idx, base in list(mutations.items()):
            
            # Get the complementary base
            comp_base = basepair_dict[traj.top._residues[idx]]
            comp_mutant = base_pair_map[base]
        
            # Update mutations with the complementary base's mutation
            mutations[comp_base.index] = comp_mutant
            
        return mutations
    
    def _find_bonds_to_delete(self, traj, target_indices):
        pre_bonds = traj.top._bonds
        atoms = np.array(traj.top._atoms)[target_indices]
        bond_indices_to_delete = [] 
        for atom in atoms:
            for _,bond in enumerate(pre_bonds):
                if atom in bond:
                    bond_indices_to_delete.append(_)
        #print('bond indices to delete', set(bond_indices_to_delete))
        return list(set(bond_indices_to_delete))
    
    def _find_bonds_to_keep(self, traj, mutant_indices):
        pre_bonds = list(traj.top.bonds)
        bonds_to_keep= []
        atoms = np.array(traj.top._atoms)[mutant_indices]
        for atom in atoms:
            for _,bond in enumerate(pre_bonds):
                if atom in bond:
                    bonds_to_keep.append(bond)
        return bonds_to_keep


    def update_mutant_topology(self, traj, target_indices, mutant_indices, base, resid, mutation_traj):

        # Store pre-deletion atom names and indices for comparison
        pre_atoms = [(atom.name, atom.index) for atom in traj.top._residues[resid]._atoms]

        # Delete bonds that are no longer valid after the mutation
        bonds_to_delete = self._find_bonds_to_delete(traj, target_indices)
        for idx in bonds_to_delete:
            traj.top._bonds.pop(idx)

        print("Pre-deletion residue atoms :", [(atom.index,atom.name) for atom in traj.top._residues[resid]._atoms])
         
        # Delete target atoms from the topology
        self._delete_target_atoms(traj, target_indices)
   
        # Store post-deletion atom names and indices for offset calculation
        post_atoms = [(atom.name, atom.index) for atom in traj.top._residues[resid]._atoms]

        # Determine the insertion offset by comparing pre and post deletion atom names and indices
        offset, insert_id = self._find_insertion_offset(pre_atoms, post_atoms)

        print("Pre-insertion residue atoms:", [(atom.index,atom.name) for atom in traj.top._residues[resid]._atoms])

        # Insert new atoms into the topology at calculated positions
        self._insert_new_atoms(traj, resid, mutant_indices, mutation_traj, offset, insert_id)

        
        print("Post-insertion residue atoms:", [(atom.index,atom.name) for atom in traj.top._residues[resid]._atoms])
        
        # Update the residue name to reflect the mutation
        traj.top._residues[resid].name = f'D{base}'

        return traj

    def _delete_target_atoms(self, traj, target_indices):
        """
        Delete target atoms from the topology by sorting indices in reverse order
        to maintain index integrity after each deletion.
        """
        #print('og',traj.top.residue(self.current_resid)._atoms)
        #print('og',[at.index for at in traj.top.residue(self.current_resid)._atoms])

        for index in sorted(target_indices, reverse=True):
            traj.top.delete_atom_by_index(index)
            #print('del',index,traj.top.residue(self.current_resid)._atoms)
            #print('del',index,[at.index for at in traj.top.residue(self.current_resid)._atoms])

    def _find_insertion_offset(self, pre_atoms, post_atoms):
        """
        Determine the correct offset for new atom insertion by comparing
        pre- and post-deletion atom names and indices.
        """
        # Default to the last atom's index in the residue as the insertion point
        offset = post_atoms[-1][1]
        insert_id = len(post_atoms) - 1

        # Check for the actual offset where the first discrepancy in atom names occurs
        # loop over name,index pairs in pre_atoms and post_atoms
        for pre, post in zip(pre_atoms, post_atoms):
            if pre[0] != post[0]:
                offset = pre[1]
                insert_id = post_atoms.index(post)
                break

        return offset, insert_id
    
    def _insert_new_atoms(self, traj, resid, mutant_indices, mutation_traj, offset, insert_id):
        """
        Insert new atoms into the topology, accounting for edge cases when the insertion point
        is at the end of the topology.
        """

        #print('empty',traj.top.residue(resid)._atoms)
        #print('empty',[at.index for at in traj.top.residue(resid)._atoms])
        for idx, mutant_index in enumerate(mutant_indices, 1):
            if self.verbose:
                print('Processing residue',resid, 'atom',idx, 'index',mutant_index)
            atom = mutation_traj.top.atom(mutant_index)

            #print('target',offset+idx,atom)
            # Edge case: If the offset is the last atom in the topology, insert new atoms at the end
            if offset + idx >= traj.top.n_atoms:
                if self.verbose:
                    print('Edgecase: inserting at or beyond the last atom in the topology', offset, traj.top.n_atoms)
                traj.top.insert_atom(atom.name, atom.element, traj.top._residues[resid],
                                    index=traj.top.n_atoms + idx, rindex=insert_id + idx)
            else:
                # Regular case: insert new atoms at the calculated offset
                # rindex: the desired position for this atom within the residue
                # index: the desired position for this atom within the topology, Existing atoms with indices >= index will be pushed back.
                #print('idx, offset+idx, insert_id+idx')
                #print(idx, offset+idx, insert_id+idx)
                if self.verbose:
                    print('Inserting atom at index', offset + idx, 'rindex', insert_id + idx)
                traj.top.insert_atom(atom.name, atom.element, traj.top._residues[resid],
                                    index=offset + idx, rindex=insert_id + idx)

            #print('ins',idx+offset, traj.top.residue(resid)._atoms)
            #print('ins',idx+offset,[at.index for at in traj.top.residue(resid)._atoms])


    def get_base_transformation(self, mutant_reference,target_reference):

        # Collect the reference information of the mutation
        mutation_origin = mutant_reference.b_R[0]
        D = mutant_reference.b_D[0]
        L = mutant_reference.b_L[0]
        N = mutant_reference.b_N[0]
        mutation_basis = np.array([D,L,N])

        # Collect the reference information of the target to mutate
        target_ref = ReferenceBase(target_reference)
        target_origin = target_ref.b_R[0]
        target_basis = np.array([target_ref.b_D[0],target_ref.b_L[0],target_ref.b_N[0]])

        # Calculate the transformation 
        rot = np.linalg.solve(target_basis,mutation_basis)
        trans = target_origin - mutation_origin
        return rot, trans


    def apply_mutations(self, traj, mutations, base_pair_map):

        # Make a copy of the original trajectory
        traj = copy.deepcopy(traj)
        
        #reference_bases = {base: md.load_pdb(get_data_file_path(f'./atomic/NDB96_{base}.pdb')) for base in base_pair_map.keys()}
        reference_bases = {base: md.load_hdf5(get_data_file_path(f'./atomic/bases/BDNA_{base}.h5')) for base in base_pair_map.keys()}
        reference_frames = {letter: ReferenceBase(t) for letter,t in reference_bases.items()}

        # For each residue that needs to be mutated
        for resid,base in mutations.items():
            if self.verbose:
                print('Processing residue',resid, 'which corresponds to nucleotide',traj.top._residues[resid],' to mutation',base)
                
            #print('resid',resid,base)
            #print(traj.top, traj.top._residues)
            self.current_resid = resid
            # Get the mutant trajectory object
            mutation_traj = reference_bases[base] 

            # Get the indices of the atoms that need to be transformed
            mutant_indices = self.get_base_indices(mutation_traj, resid=0)
            target_indices = self.get_base_indices(traj, resid=resid)

            #print('n_mutant',len(mutant_indices))
            #print('n_target',len(target_indices))
            #print('diff', len(mutant_indices) - len(target_indices))
            #sub_m = mutation_traj.atom_slice(mutant_indices)
            #sub_w = traj.atom_slice(target_indices)

            #print('m',sub_m.top._residues[0],sub_m.top._atoms)
            #print('w',sub_w.top._residues[0],sub_w.top._atoms)
            
            # Get the transformation for the local reference frames from the mutant to the target
            mutant_reference = reference_frames[base]
            target_reference = traj.atom_slice(traj.top.select(f'resid {resid}'))
            # print(resid, target_reference.top,  target_reference.top._residues)
            rot, trans = self.get_base_transformation(mutant_reference, target_reference)
           
            # Transform the mutant atoms to the local reference frame of the target
            mutant_xyz = mutation_traj.xyz[:,mutant_indices,:]
            new_xyz = np.dot(mutant_xyz, rot.T) + trans    

            # Get the original xyz coordinates
            xyz = traj.xyz 
            #print('target_indices:',target_indices)
            # Split the xyz in 2 pieces, one before the indices that need to be replaced, and the indices after the indices that need to be replaced
            xyz1 = xyz[:,:target_indices[0],:] 
            xyz2 = xyz[:,target_indices[-1]+1:,]

            # Update the topology
            traj = self.update_mutant_topology(traj, target_indices, mutant_indices, base, resid, mutation_traj)

            print("Original trajectory shape:", traj.xyz.shape)
            print("xyz1 shape:", xyz1.shape)
            print("new_xyz shape:", new_xyz.shape)
            print("xyz2 shape:", xyz2.shape)
            print("Concatenated shape:", xyz1.shape[1] + new_xyz.shape[1] + xyz2.shape[1])


            # Concatenate the new xyz with the original xyz
            xyz = np.concatenate([xyz1, new_xyz, xyz2], axis=1)
            traj.xyz = xyz

        # Check if the atom indices are monotonically increasing
        #print('n_atoms',traj.top.n_atoms)
        #print('xyz', traj.xyz.shape)
        ats = [at.index for at in traj.top.atoms]
        #print('atdiff', np.diff(ats))


        # Return the mutated trajectory
        return traj

    def get_traj(self):
        return self.mutant_traj

# traj = md.load('/Users/thor/surfdrive/Data/h-ns/BacterialChromatin/FI/0_k/2_ApT/dry_0.pdb')
# traj.remove_solvent(inplace=True)
# traj = traj.atom_slice(traj.topology.select('not protein'))[0]
# # traj = traj.atom_slice(traj.topology.select('resid 0 23'))# and not element symbol H'))

# # Create a DNA object
# dna = mdna.NucleicFrames(traj)

# # Define the base pair map and the complementary mutant map
# base_pair_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}#,'P':'T','M':'C','H':'T'}


# # Get base frames from current DNA sequence
# base_frames = dna.frames

# # # Define the mutation to be performed
# complementary = True
# mutations = {0: 'A', 6: 'T'}

# if complementary:
#     # Update dict with the complementary mutations
#     mutations = make_complementary_mutations(traj, mutations, base_pair_map)

# print(mutations)
# sequence = mdna.utils.get_sequence_letters(traj)
# sequence_pairs = mdna.utils.get_base_pair_letters(traj)
# new_sequences = [mutations.get(idx, seq) for idx, seq in enumerate(sequence)]

# print('WT',sequence)
# # print(new_sequences)

# # Apply the mutations
# mutant_traj = apply_mutations(traj, mutations, base_pair_map)
    
# new_sequence = mdna.utils.get_sequence_letters(mutant_traj)
# print('M ',new_sequence)
# view = nv.show_mdtraj(mutant_traj)
# view.clear_representations()
# view.add_representation('ball+stick', selection='all')
# view
