import copy
from .utils import get_base_pair_dict, RigidBody, get_data_file_path
import numpy as np
from .geometry import ReferenceBase
import mdtraj as md
import copy
from mdtraj.core import element as elem
from .chemistry import (
    NUCLEOTIDE_COMPLEMENTS,
    copy_template_bonds,
    get_nucleotide_definition,
    template_relative_path,
)



class Methylate:
    """Add methyl groups to cytosine (C5) or guanine (O6) residues.

    Currently supports C-methylation at C5 and G-methylation at O6.
    Optionally restricts C-methylation to CpG dinucleotide contexts.

    Attributes:
        traj (md.Trajectory): Modified trajectory with methylated residues.
        baselist (list[int]): Residue indices that were methylated.

    Example:
        ```python
        methylator = Methylate(traj, methylations=[0, 5])
        methylated_traj = methylator.get_traj()
        ```
    """

    def __init__(self, traj, methylations=None, CpG=False, leading_strand=None):
        """Initialize methylation.

        Args:
            traj (md.Trajectory): Input DNA trajectory.
            methylations (list[int], optional): Residue indices to methylate.
            CpG (bool): If True, auto-detect and methylate all C in CpG contexts
                (overrides *methylations*).
            leading_strand (int, optional): Chain index of the leading strand.
                Required when *CpG* is True.
        """
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
        """Modify the trajectory in-place by adding a methyl carbon to each target residue."""

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


    def find_CpGs(self, leading_strand=0):
        """Return residue indices of C bases in CpG contexts on the leading strand.

        Args:
            leading_strand (int): Chain index.

        Returns:
            list[int]: Residue indices.
        """
        sequence = ''.join([res.name[1] for res in self.traj.top.chain(leading_strand)._residues])
        return [i for i in range(len(sequence) - 1) if sequence[i:i+2] == 'CG']

    def get_atoms(self, residue, code):
        """Get the anchor and reference atoms for methyl placement.

        Args:
            residue: MDTraj residue object.
            code (str): One-letter base code (``'C'`` or ``'G'``).

        Returns:
            tuple: ``(anchor_atom, reference_atom)`` or ``(None, None)``.
        """
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
        """Insert a methyl carbon atom into the topology and coordinates.

        Args:
            a: Anchor atom (C5 for C, O6 for G).
            b: Reference atom used to determine placement direction.
            residue: Target residue.
            code (str): One-letter base code.
        """

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
        """Return the methylated trajectory."""
        return self.traj
    

class Hoogsteen:
    """Apply a Hoogsteen base-pair flip by rotating nucleobases 180°.

    Rotates the nucleobase around the glycosidic bond (C1′–N) axis
    to produce a syn conformation.

    Attributes:
        traj (md.Trajectory): Modified trajectory.
        fliplist (list[int]): Residue indices that were flipped.

    Example:
        ```python
        hs = Hoogsteen(traj, fliplist=[3, 8])
        flipped_traj = hs.get_traj()
        ```
    """

    def __init__(self, traj, fliplist, deg=180, verbose=False):
        """Initialize Hoogsteen flip.

        Args:
            traj (md.Trajectory): Input DNA trajectory.
            fliplist (list[int]): Residue indices to flip.
            deg (float): Rotation angle in degrees.
            verbose (bool): Print diagnostic messages.
        """
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
        return [index + offset for index in sub_indices]
    
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
        """Return the flipped trajectory."""
        return self.traj


class Mutate:
    """Mutate one or more bases in a DNA structure.

    Replaces nucleobases by superimposing reference base coordinates
    onto the target residue frame.  Complementary-strand mutations are
    applied automatically when *complementary* is True.

    Attributes:
        mutant_traj (md.Trajectory): Trajectory with applied mutations.

    Example:
        ```python
        mutator = Mutate(traj, mutations={0: 'A', 6: 'T'})
        mutant_traj = mutator.get_traj()
        ```
    """

    def __init__(self, traj, mutations, complementary=True, verbose=False):
        """Initialize mutation.

        Args:
            traj (md.Trajectory): Input DNA trajectory.
            mutations (dict[int, str]): Mapping of residue index to desired
                one-letter base code.
            complementary (bool): Automatically mutate the complementary strand.
            verbose (bool): Print diagnostic messages.
        """
    
        self.traj = traj
        self.complementary = complementary
        self.verbose = verbose
        self.mutations = mutations
        self.current_resid = 0
        print('hello')
        self.mutate()
        
    def mutate(self):

        # Use the same registered complement map as the structure builder.
        base_pair_map = dict(NUCLEOTIDE_COMPLEMENTS)
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
        return [index + offset for index in sub_indices]


    def make_complementary_mutations(self, traj, mutations, base_pair_map):
        
        # Get the basepair dictionary of the trajectory
        basepair_dict = get_base_pair_dict(traj)
        mutations = {int(index): str(base).upper() for index, base in mutations.items()}

        # Iterate over a static list of dictionary items to avoid RuntimeError
        for idx, base in list(mutations.items()):
            get_nucleotide_definition(base)
            
            # Get the complementary base
            comp_base = basepair_dict[traj.top._residues[idx]]
            comp_mutant = base_pair_map[base]
        
            # Update mutations with the complementary base's mutation
            if comp_base.index in mutations and mutations[comp_base.index] != comp_mutant:
                raise ValueError(
                    f"Conflicting mutations for complementary residue {comp_base.index}: "
                    f"{mutations[comp_base.index]!r} and {comp_mutant!r}."
                )
            mutations[comp_base.index] = comp_mutant
            
        return mutations
    
    def get_base_transformation(self, mutant_reference, target_reference):
        """Return per-frame rigid transforms from a template to the target base frame."""
        mutation_origin = mutant_reference.b_R[0]
        mutation_basis = np.array([
            mutant_reference.b_D[0],
            mutant_reference.b_L[0],
            mutant_reference.b_N[0],
        ])
        target_ref = ReferenceBase(target_reference)
        rotations = []
        translations = []
        for frame in range(target_reference.n_frames):
            target_basis = np.array([
                target_ref.b_D[frame],
                target_ref.b_L[frame],
                target_ref.b_N[frame],
            ])
            rotation = np.linalg.solve(target_basis, mutation_basis)
            translation = target_ref.b_R[frame] - mutation_origin @ rotation.T
            rotations.append(rotation)
            translations.append(translation)
        return np.asarray(rotations), np.asarray(translations)

    def apply_mutations(self, traj, mutations, base_pair_map):
        mutations = {int(index): str(base).upper() for index, base in mutations.items()}
        invalid_residues = [
            index for index in mutations
            if index < 0 or index >= traj.topology.n_residues
        ]
        if invalid_residues:
            raise IndexError(f"Mutation residue indices are out of range: {invalid_residues}.")
        for base in mutations.values():
            get_nucleotide_definition(base)
        if not mutations:
            return copy.deepcopy(traj)

        reference_bases = {
            base: md.load_hdf5(get_data_file_path(template_relative_path(base)))
            for base in set(mutations.values())
        }
        reference_frames = {
            base: ReferenceBase(reference)
            for base, reference in reference_bases.items()
        }

        # Calculate replacements against untouched coordinates to avoid index drift.
        aligned_base_coordinates = {}
        for resid, base in mutations.items():
            mutation_traj = reference_bases[base]
            mutant_indices = self.get_base_indices(mutation_traj, resid=0)
            target_residue = traj.topology.residue(resid)
            target_reference = traj.atom_slice([atom.index for atom in target_residue.atoms])
            rotations, translations = self.get_base_transformation(
                reference_frames[base], target_reference
            )
            source_coordinates = mutation_traj.xyz[0, mutant_indices, :]
            aligned = np.stack([
                source_coordinates @ rotations[frame].T + translations[frame]
                for frame in range(traj.n_frames)
            ])
            aligned_base_coordinates[resid] = {
                atom_index: aligned[:, position, :]
                for position, atom_index in enumerate(mutant_indices)
            }

        new_topology = md.Topology()
        chain_map = {}
        old_atom_map = {}
        template_atom_maps = {}
        coordinate_columns = []

        def is_backbone_atom(atom):
            return "'" in atom.name or "P" in atom.name

        for residue in traj.topology.residues:
            old_chain = residue.chain
            if old_chain.index not in chain_map:
                chain_map[old_chain.index] = new_topology.add_chain(old_chain.chain_id)
            chain = chain_map[old_chain.index]
            base = mutations.get(residue.index)
            residue_name = (
                get_nucleotide_definition(base).residue_name
                if base is not None else residue.name
            )
            new_residue = new_topology.add_residue(
                residue_name, chain, resSeq=residue.resSeq,
                segment_id=residue.segment_id,
            )

            retained_by_name = {}
            for atom in residue.atoms:
                if base is not None and not is_backbone_atom(atom):
                    continue
                new_atom = new_topology.add_atom(
                    atom.name, atom.element, new_residue,
                    serial=new_topology.n_atoms + 1,
                    formal_charge=atom.formal_charge,
                )
                old_atom_map[atom] = new_atom
                retained_by_name[atom.name] = new_atom
                coordinate_columns.append(traj.xyz[:, atom.index, :])

            if base is None:
                continue

            mutation_traj = reference_bases[base]
            template_atom_map = {}
            for atom in mutation_traj.topology.residue(0).atoms:
                if is_backbone_atom(atom):
                    if atom.name in retained_by_name:
                        template_atom_map[atom] = retained_by_name[atom.name]
                    continue
                new_atom = new_topology.add_atom(
                    atom.name, atom.element, new_residue,
                    serial=new_topology.n_atoms + 1,
                    formal_charge=atom.formal_charge,
                )
                template_atom_map[atom] = new_atom
                coordinate_columns.append(
                    aligned_base_coordinates[residue.index][atom.index]
                )
            template_atom_maps[residue.index] = (mutation_traj, template_atom_map)

        # Preserve inter-residue links and all unmutated internal bonds.
        for bond in traj.topology.bonds:
            same_residue = bond.atom1.residue.index == bond.atom2.residue.index
            if same_residue and bond.atom1.residue.index in mutations:
                continue
            if bond.atom1 not in old_atom_map or bond.atom2 not in old_atom_map:
                continue
            new_topology.add_bond(
                old_atom_map[bond.atom1], old_atom_map[bond.atom2],
                type=getattr(bond, "type", None),
                order=getattr(bond, "order", None),
            )

        for mutation_traj, atom_map in template_atom_maps.values():
            copy_template_bonds(
                new_topology, mutation_traj.topology, atom_map, require_all=False
            )

        coordinates = np.stack(coordinate_columns, axis=1)
        return md.Trajectory(
            coordinates,
            new_topology,
            time=traj.time,
            unitcell_lengths=traj.unitcell_lengths,
            unitcell_angles=traj.unitcell_angles,
        )

    def get_traj(self):
        """Return the mutated trajectory."""
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
