import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import nglview as nv
import copy

# https://biopython.org/docs/1.74/api/Bio.SVDSuperimposer.html
from Bio.SVDSuperimposer import SVDSuperimposer

from pdbfixer import PDBFixer
from openmm.app import PDBFile

import os
import pandas as pd
import glob
import subprocess


from scipy.spatial.transform import Rotation as R
import numpy as np
import mdtraj as md
import nglview as nv


class Mover:

    def __init__(self):
        pass

    # Compute the optimal rotation and translation to align two trajectories
    def get_rot_and_trans(self, subtraj_A, subtraj_B):
        sup = SVDSuperimposer()  # Instantiate SVDSuperimposer object
        sup.set(subtraj_A.xyz[0], subtraj_B.xyz[0])  # Set two structures to be superimposed
        sup.run()  # Perform the superimposition
        rot, tran = sup.get_rotran()  # Retrieve rotation and translation
        return rot, tran, sup.get_rms()  # Return rotation, translation, and RMS

    # Apply superimposition to the given trajectory using the rotation and translation
    def apply_superimposition(self, traj, rot, tran):
        xyz = traj.xyz[0]  # Extract coordinates
        new_xyz = np.dot(xyz, rot) + tran  # Apply rotation and translation
        traj.xyz = new_xyz  # Update the coordinates
        return traj  # Return updated trajectory

    # Align two trajectories using superimposition
    def align(self,traj_A, traj_B, selection_A=None, selection_B=None, replace=None):
        # If no selection is provided, select all atoms
        if selection_A is None:
            selection_A = traj_A.top.select('all')
        if selection_B is None:
            selection_B = traj_B.top.select('all')

        # Raise exception if selections have different lengths
        if len(selection_A) != len(selection_B):
            raise ValueError('Selections have different lengths and need to be fixed')

        # Create sub-trajectories for alignment
        subtraj_A = traj_A.atom_slice(selection_A)
        subtraj_B = traj_B.atom_slice(selection_B)

        # Get rotation and translation to align A to B
        rot, tran, rms = self.get_rot_and_trans(subtraj_A, subtraj_B)
        # Apply superimposition to traj_B
        sup_B = self.apply_superimposition(traj_B, rot, tran)

        # If replace is specified, replace parts of traj_A with the superimposed traj_B
        if replace is not None:
            if not isinstance(replace, list):
                replace = [replace]  # make single int iterable
            # keep chains not in replace
            selection_to_keep_A = [at.index for at in traj_A.top.atoms if at.residue.chain.index not in replace]
            new_traj_A = traj_A.atom_slice(selection_to_keep_A)
            # stack the newly aligned structure with the rest of traj_A
            replaced_traj_A = new_traj_A.stack(sup_B)
            return replaced_traj_A, rms

        return sup_B, rms

    # Apply a translation vector to move one group of atoms relative to another
    def apply_translation_vector(self,traj, A_indices, B_indices, subset_A=None, subset_B=None,  direction=1, translation_magnitude=0.5):
        # If no subset specified, use indices as subset
        if subset_B is None:
            subset_B = B_indices
        if subset_A is None:
            subset_A = A_indices

        # Compute center of mass for each subset
        A_com = md.compute_center_of_mass(traj.atom_slice(subset_A))
        B_com = md.compute_center_of_mass(traj.atom_slice(subset_B))

        # Compute translation vector from A to B
        translation_vector = B_com - A_com
        # Normalize the translation vector and scale by the magnitude and direction (1 or -1 for forward or backward)
        translation_vector /= np.linalg.norm(translation_vector)
        translation_vector *= (translation_magnitude * direction)

        # Apply the translation vector to the atoms in A
        for frame in range(traj.n_frames):
            traj.xyz[frame, A_indices] -= translation_vector[frame]
        # Return the translated trajectory
        return traj
    
    # Align two trajectories using superimposition along a given axis
    def align_traj(self, traj, subset, align_to_axis=[0, 1, 0], superpose=False):

        # If superpose is True, superpose the trajectory to the first frame
        if superpose:
            traj = traj.superpose(traj, 0, atom_indices=subset)
            
        # Get the coordinates as a numpy array
        coords = traj.xyz.reshape(-1, 3)

        # Calculate the center of mass of the subset
        center_of_mass = np.mean(coords[subset], axis=0)

        # Translate coordinates so the center of mass is at the origin
        coords_centered = coords - center_of_mass

        # Calculate the covariance matrix of the subset
        cov = np.cov(coords_centered[subset].T)

        # Compute the singular value decomposition
        u, s, vh = np.linalg.svd(cov)

        # The principal axis corresponds to the largest singular value
        principal_axis = u[:, np.argmax(s)]

        # Compute the cross product of the principal axis and align_to_axis to get the rotation axis
        rotation_axis = np.cross(principal_axis, align_to_axis)

        # Compute the dot product of the principal axis and align_to_axis to get the cosine of the rotation angle
        cos_angle = np.dot(principal_axis, align_to_axis)

        # Compute the rotation angle
        rotation_angle = np.arccos(cos_angle)

        # Create the rotation matrix using scipy's Rotation module
        rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis)

        # Apply the rotation to all the atoms
        rotated_coords = rotation_matrix.apply(coords_centered)

        # Translate the rotated coordinates back
        rotated_coords += center_of_mass

        # Reshape the rotated coordinates to match the original shape
        rotated_coords = rotated_coords.reshape(traj.xyz.shape)

        # Create a new trajectory with the rotated coordinates
        aligned_traj = md.Trajectory(rotated_coords, traj.topology, time=traj.time, unitcell_lengths=traj.unitcell_lengths, unitcell_angles=traj.unitcell_angles)
        
        return aligned_traj


class Capper:

    def __init__(self,traj,leading_chainid, N_term=True, C_term=False):

        """

        Example Usage:

        traj = md.load('./pdbs/FI_HNS_Atr.pdb')
        capper = Capper(copy.deepcopy(traj),leading_chainid=2,N_term=True,C_term=False)
        print(capper.capped_traj.top)
        for c in capper.capped_traj.top.chains:
        print([res for res in c.residues])


        # Caps have been made as follows:
        peptide = md.load('./capped_peptide.pdb')
        ACE_indices = list(peptide.top.select('(resSeq 0) or (resSeq 1 and backbone)'))
        ACE = peptide.atom_slice(ACE_indices)
        ACE.save('ACE_cap.pdb')
        NME_indices = peptide.top.select('(resid 4) or (resid 3 and backbone)')
        NME = peptide.atom_slice(NME_indices)
        NME.save('NME_cap.pdb')
        """

        self.traj = traj
        self.top = traj.top
        self.leading_chainid = leading_chainid
        self.N_term = N_term
        self.C_term = C_term
    
        self.analyse_leading_chain()
        self.load_caps()
        self.fit_caps()
        self.add_caps()
        self.rebuild_system()

    
    def get_rot_and_trans(self,subtraj_A,subtraj_B):
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

    def apply_superimposition(self,traj, rot, tran):
        
        # get xyz coordinates
        xyz = traj.xyz[0]
        
        # rotate subject on target
        new_xyz = np.dot(xyz, rot) + tran

        # replace coordinates of traj
        traj.xyz = new_xyz
        return traj
    
    def fit_B_on_A(self,A, B, selection_A, selection_B):
        
        # create trajs containing only the selections
        subtraj_A = A.atom_slice(selection_A)
        subtraj_B = B.atom_slice(selection_B)

        # obtain instructions to rotate and translate B on A based on substraj structures
        rot, tran, rms = self.get_rot_and_trans(subtraj_A,subtraj_B)
        
        # do the superimposition of B on A and subsitute old with new xyz of B
        sup_B = self.apply_superimposition(B, rot, tran)

        # remove overlapping backbone atoms from B 
        selection_to_keep = [at.index for at in  B.top.atoms if at.index not in selection_B]

        new_B = sup_B.atom_slice(selection_to_keep)
        return new_B, rms

    def get_fit_indices(self,residue):
        res_indices = [at.index  for at in residue.atoms if at.name in ['N','CA','C','O']]
        return res_indices
            
    def analyse_leading_chain(self):

        # Get terminal residues of leading chain
        chain_top = self.top.chain(self.leading_chainid)
        self.leading_chain_residues = list(chain_top.residues)
        print('First and last residue of leading chain: ', self.leading_chain_residues[0], self.leading_chain_residues[-1])
        self.first_res_fit_indices = self.get_fit_indices(self.leading_chain_residues[0]) 
        self.last_res_fit_indices = self.get_fit_indices(self.leading_chain_residues[-1]) 

    def load_caps(self):
        # Load N and C terminal caps with backbone of the residues one after ACE and one before NME
        if self.N_term:
            self.ACE = md.load('./cap_pdbs/ACE_cap.pdb')
            self.ACE_fit_indices = list(self.ACE.top.select('backbone and not resname ACE'))
        if self.C_term:
            self.NME = md.load('./cap_pdbs/NME_cap.pdb')
            self.NME_fit_indices = list(self.NME.top.select('backbone and not resname NME'))

        if not self.N_term and not self.C_term:
            print('Please specify N_term and/or C_term')

    def fit_caps(self):
        if self.N_term:
            self.fitted_ACE, rms_ACE = self.fit_B_on_A(A=self.traj, B=self.ACE, selection_A=self.first_res_fit_indices, selection_B=self.ACE_fit_indices)
            print(f'RMS of fit ACE: {rms_ACE}')
        if self.C_term:
            self.fitted_NME, rms_NME = self.fit_B_on_A(A=self.traj, B=self.NME, selection_A=self.last_res_fit_indices, selection_B=self.NME_fit_indices)
            print(f'RMS of fit NME: {rms_NME}')
        
    def add_caps(self):

        protein = self.traj.atom_slice(self.traj.top.select(f'chainid {self.leading_chainid}'))
        cap_ace = copy.deepcopy(self.fitted_ACE) if self.N_term else None
        cap_nme = copy.deepcopy(self.fitted_NME) if self.C_term else None

        new_top = md.Topology()
        # set resSeq counter to one residue backwards from first residue of protein
        first_res = self.leading_chain_residues[0]
        s = first_res.resSeq - 1 if self.N_term else first_res.resSeq
        
        # Create empty chain
        c = new_top.add_chain()
        for chain in protein.top.chains:

            # add ACE cap at the beginning if add_ace is True
            if self.N_term:
                for residue in list(cap_ace.top.residues):
                    r = new_top.add_residue(residue.name, c, s, residue.segment_id)
                    s += 1
                    for atom in residue.atoms:
                        new_top.add_atom(atom.name, atom.element, r)
            
            # add the chain residues
            for residue in list(chain.residues):
                r = new_top.add_residue(residue.name, c, s, residue.segment_id)
                s += 1
                for atom in residue.atoms:
                    new_top.add_atom(atom.name, atom.element, r)

            # add NME cap at the end if add_nme is True
            if self.C_term:
                for residue in list(cap_nme.top.residues):
                    r = new_top.add_residue(residue.name, c, s, residue.segment_id)
                    s += 1
                    for atom in residue.atoms:
                        new_top.add_atom(atom.name, atom.element, r)

        # Create bonds
        new_top.create_standard_bonds()

        # Create a list of the xyz attributes to merge, only if the corresponding cap is not None
        xyzs_to_merge = []
        if cap_ace is not None:
            xyzs_to_merge.append(cap_ace.xyz)
        xyzs_to_merge.append(protein.xyz)
        if cap_nme is not None:
            xyzs_to_merge.append(cap_nme.xyz)
        
        # Merge the xyzs and create a new trajectory
        merged_xyz = np.concatenate(xyzs_to_merge, axis=1)
        self.capped_protein = md.Trajectory(merged_xyz, new_top)

    def rebuild_system(self):
        # Create a new empty topology
        new_top = md.Topology()

        # Initialize a list to store new coordinates
        new_coords = []

        # Iterate over the chains in the original trajectory
        for chain in self.traj.top.chains:
            new_chain = new_top.add_chain()
            # If it's the chain to cap, add the capped protein
            if chain.index == self.leading_chainid:
                for residue in self.capped_protein.top.residues:
                    new_res = new_top.add_residue(residue.name, new_chain, resSeq=residue.resSeq, segment_id=residue.segment_id)
                    for atom in residue.atoms:
                        new_top.add_atom(atom.name, atom.element, new_res)
                new_coords.append(self.capped_protein.xyz)
            else:
                # If it's not the chain to cap, copy it over
                sub_traj = self.traj.atom_slice(self.traj.top.select(f'chainid {chain.index}'))
                for residue in chain.residues:
                    new_res = new_top.add_residue(residue.name, new_chain, resSeq=residue.resSeq, segment_id=residue.segment_id)
                    for atom in residue.atoms:
                        new_top.add_atom(atom.name, atom.element, new_res)
                new_coords.append(sub_traj.xyz)
        
        # Create bonds
        new_top.create_standard_bonds()

        # Concatenate all the new coordinates
        new_xyz = np.concatenate(new_coords, axis=1)

        # This will become the new trajectory
        self.capped_traj = md.Trajectory(new_xyz, new_top)


class MutateDNA:
    
    """Atomic_2AP.pdb needs to be in this folder and the /Applications/x3dna.2.4/config/baselist.dat 
       needs to be edited to contain the new 2AP residue complementary to 'a'.
       
       
       Example:
       
       traj = md.load('./FI_highaff_capped.pdb')
        view = nv.show_mdtraj(traj)

        names = ['Atr','Gtr','GpT','GpC','ApT','GC-ana','3GC-ana','5GC-ana','cons','anit']

        sequences = ['GCGAAAAAAAGC',
                    'GCGGGGGGGGGC',
                    'GCGTGTGTGTGC',
                    'GCGCGCGCGCGC',
                    'GCATATATATGC',
                    'GCGGCGCGCCGC',
                    'GCGGCGTATTGC',
                    'GCAATACGCCGC']

        mutants = {}
        for n,s in zip(names,sequences):
            print(n,s)
            mutant = MutateDNA(traj,sense_letter='A',anti_letter='B',name=f'FI_{n}',loc='./mutants/')
            mutant.mutate(new_sequence=s, complementary=True)
            mutants[n] = mutant
    """

    def __init__(self, structure, sense_letter, anti_letter, name,loc='./',fixed_only=False):

        self.name = name
        self.structure = structure
        self.top = structure.topology
        self.loc = loc
        self.fixed_only = fixed_only
        self.sense_letter = sense_letter
        self.anti_letter = anti_letter
        self.map_chain_letters()

        self.collect_sequence_info()

    def mutate(self, new_sequence, complementary=True, pdbfix=True):

        # Save Wildtype
        self.structure.save(f'{self.loc}{self.name}_wildtype.pdb')
        
        # Check mutation for errors
        self.complementary = complementary
        self.new_sequence = new_sequence 
        self.check_mutation()
        
        # Mutate Wildtype to new sequence
        self.mutant_name = f'{self.name}_mutant'
        self.mutations = self.make_mutation_instructions()        
        self.apply_mutations(pdbfix)
        self.check_if_complementary(self.mutant_name)

    def map_chain_letters(self):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        chain_map = {letter:idx for letter,idx in zip(letters,range(0,26))}
        self.sense_idx = chain_map[self.sense_letter]
        self.anti_idx = chain_map[self.anti_letter]

    def collect_strand_info(self):
        # Collect residues of strands
        self.strands = {}
        self.strands['sense'] = self.top._chains[self.sense_idx]._residues
        self.strands['anti'] = self.top._chains[self.anti_idx]._residues

    def map_base_pairs(self):
        # Map complementary bases 
        self.base_pair_dict = {}
        self.base_pair_dict = {i:j for i,j in zip (self.strands['sense'],reversed(self.strands['anti']))}
        self.base_pair_dict.update({j:i for i,j in self.base_pair_dict.items()})
    
    def get_fasta(self, strand_type):
        sequence_list = [res.name[1:2] if res.name[1:2] in ['A','C','G','T'] else 'X' for res in self.strands[strand_type]]
        return ''.join(map(str, sequence_list))

    def collect_sequence_info(self):
        self.collect_strand_info()
        self.map_base_pairs()
        self.sequence = self.get_fasta('sense') 
        self.nbp = len(self.sequence)
        print(f"Current sequence of sense strand: ",self.sequence)
        print(f"Current sequence of anti-sense strand: { self.get_fasta('anti') }")

    def check_mutation(self):
        
        if len(self.new_sequence) != len(self.sequence) or self.new_sequence == self.sequence:
            print('New sequence should contain a mutation and of equal length as old sequence...')
            return ValueError
        else:
            print('Length of mutant sequence validated.')

    def make_mutation_instructions(self):
        
        # Complementary bases with X custom 2AP mutation
        c_base_dict = {'A':'T','T':'A','C':'G','G':'C','X':'X'}
        
        # Read mutations for x3DNA
        mutations = []
        self.mutant_indices = []

        residx = 0
        for base_wt, base_m in zip(self.sequence, self.new_sequence):

            # Wild type and mutant are not the same
            if base_wt != base_m:

                # Get wt residue to mutate 
                b_wt = self.strands['sense'][residx]

                # Check for 2AP label 
                if self.new_sequence[residx] == 'X':
                    base_m = '2AP'

                # Mutate base_pair.... at chain A, residue s to mutant
                #mutations.append(f'c={self.sense_letter} s={b_wt.index+1} m={base_m}\n')
                mutations.append(f'c={self.sense_letter} s={b_wt.resSeq} m={base_m}\n')
                self.mutant_indices.append(residx)

                if self.complementary:
                    # find complementary residue and mutate complementary base
                    b_wt_c = self.base_pair_dict[b_wt]
                    #mutations.append(f'c={self.anti_letter} s={b_wt_c.index+1-self.nbp} m={c_base_dict[base_m]}\n')
                    mutations.append(f'c={self.anti_letter} s={b_wt_c.resSeq} m={c_base_dict[base_m]}\n')
                    #mutations.append(f'c={self.anti_letter} s={b_wt_c.index+1} m={c_base_dict[base_m]}\n')
                    #mutations.append(f'c={self.anti_letter} s={b_wt_c.index+1} m={c_base_dict[base_m]}\n')

            residx +=1 
        
        # Write mutations to mutations.dat     
        with open(f'{self.loc}{self.name}_mutations.dat','w') as file:
            file.writelines(mutations)

        return mutations

    def do_pdbfix(self,structure_name):

        fixer = PDBFixer(filename=f'{self.loc}{structure_name}.pdb')
        
        # locate missing atoms/residues
        fixer.findNonstandardResidues()
        fixer.findMissingResidues()
        fixer.findMissingAtoms()

        for residue,atoms in fixer.missingAtoms.items():
            print(residue,len(atoms),'atom(s) :\n', atoms,'\n')

        # Add missing structure
        fixer.replaceNonstandardResidues()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        if not self.fixed_only:
            PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{self.loc}{structure_name}_fixed.pdb', 'w'))
        else:
            PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{self.loc}{structure_name}.pdb', 'w'))

    def apply_mutations(self,pdbfix):

        # Check if x3DNA is installed
        try:
            #location_x3dna = os.environ["$X3DNA"]

            location_x3dna = '/Users/thor/surfdrive/x3dna-v2.4'
            #print(location_x3dna)
        except KeyError as ex:
            print(f'The $X3DNA variable not found in the environment.\n Make sure X3DNA is installed and the environment ' \
                  f'variable $X3DNA is defined.')
        
        # Use mutations.dat to mutate wild type pdb 
        result = subprocess.getstatusoutput([f'{location_x3dna}/bin/mutate_bases -l {self.loc}{self.name}_mutations.dat {self.loc}{self.name}_wildtype.pdb {self.loc}{self.mutant_name}_raw.pdb']) 
        print('\n')
        for i in result[1:]:
            print(i[3:])

        # Open mutant pdb file
        file = open(f'{self.loc}{self.mutant_name}_raw.pdb', 'r')
        new_lines = []
        count = 0

        # Strips the newline character and rename residue name 
        for l in file.readlines():
            words = l.split()
            if words[0] =='END':
                break
            if words[3] in ['A','T','C','G']:
                l = l.replace(f' {words[3]} ',f'D{words[3]} ')
            new_lines.append(l)
            
        # And write everything back to a new pdb-file
        with open(f'{self.loc}{self.mutant_name}.pdb', 'w') as file:
            file.writelines(new_lines)
        

        print("\nCongratulations your created a mutant!")
        print(f'\nWild type file name:\n{self.name}.pdb\n')
        print(f'Mutant file name:\n{self.mutant_name}.pdb\n')
        
        if pdbfix:
            self.do_pdbfix(self.mutant_name)
            print(f'\nAlso generated {self.mutant_name}_fixed.pdb, which might do the trick... Or NOT!')

    def check_if_complementary(self, name):
    
        pdb_name = md.load(f'{self.loc}{name}.pdb')
        top = pdb_name.topology
        
        # Collect bp residues of chains 
        strand_A = top._chains[0]._residues
        strand_B = top._chains[1]._residues
        
        # Collect residue names
        a = [i.name[1:] for i in strand_A]
        b = [i.name[1:] for i in strand_B]

        pairs = ['AT','CG','GC','TA']
    
        # Check if complementary
        count, complementary, non_complementary = 0, 0, 0
        
        print('\nidx      A     B','\t M\n')
        for i,j in zip(a,reversed(b)):
            
            if i+j not in pairs:
                if count in self.mutant_indices:
                    print(count,'\t', i,'...',j,'\t y')
                else:
                    print(count,'\t', i,'...',j)
                non_complementary += 1
            else:
                if count in self.mutant_indices:
                    print(count,'\t', i,'---',j,'\t y')
                else:
                    print(count,'\t', i,'---',j)
                complementary += 1
            count += 1  
        
        print(f'\nTotal number of basepairs: {len(strand_A)}\nNumber of mutations: {len(self.mutant_indices)}\nNon-complementary basepairs: {non_complementary}\nComplementary basepairs: {complementary}')

    def show_mutations(self):
        print('residx  old    new')
        count = 0
        for i,j in zip(self.sequence,self.new_sequence):
            if i is not j:
                print(count,f'\t{i}  --> ',j)
            else:
                print(count,f'\t{i}')
            count +=1

    def view_mutant(self,atoms=False):

        view = nv.show_mdtraj(md.load(f'{self.loc}{self.mutant_name}_fixed.pdb'))
        view.clear()
        view.add_representation(repr_type = 'cartoon', selection='protein and not hydrogen and not Na and not Cl',color='teal')


        residx_sense = [at.index for res in  self.strands['sense'] for at in res.atoms]
        residx_anti = [at.index for res in  self.strands['anti'][:self.nbp-1] for at in res.atoms]

        view.add_representation(repr_type = 'base', selection=residx_sense,color='coral')
        view.add_representation(repr_type = 'cartoon', selection=residx_sense,color='coral')

        view.add_representation(repr_type = 'base', selection=residx_anti,color='cornflowerblue')
        view.add_representation(repr_type = 'cartoon', selection=residx_anti,color='cornflowerblue')


        m_residx_sense = [at.index for res in  self.strands['sense'] if res.index in self.mutant_indices for at in res.atoms]
        if not atoms:
            view.add_representation(repr_type='base',selection = m_residx_sense, color='springgreen',aspectRatio=1)
        else:
            view.add_representation(repr_type='ball+stick',selection = m_residx_sense,color='lightgreen',aspectRatio=3)

        if self.complementary:
            comp_mutant_indices = [self.base_pair_dict[res].index for res in self.strands['sense'] if res.index in self.mutant_indices]
            m_residx_anti = [at.index for res in  self.strands['anti'] if res.index in comp_mutant_indices for at in res.atoms]
            if not atoms:
                view.add_representation(repr_type='base',selection = m_residx_anti,color='springgreen',aspectRatio=1)
            else:
                view.add_representation(repr_type='ball+stick',selection = m_residx_anti,color='lightgreen',aspectRatio=3)

        return view