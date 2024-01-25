import numpy as np
import mdtraj as md
import random
from .geometry import NucleicFrames 

class SequenceGenerator:
    
    """Generates a mdtraj DNA trajectory and topology based on a provided sequence.
    
        # Some sanity checks to see if topology is properly created
        selection = DNA.top.select('(resname DG and chainid 1) and element type N')
        sliced = DNA.atom_slice(selection)
        import nglview as nv
        view = nv.show_mdtraj(sliced)
        view
        
        # Generate DNA trajectory and topology with dummy coordinates (aka everything based on standard bases)
        dna_generator = DNAGenerator(sequence='GCAATATATTGC', circular=False)
        dna = dna_generator.DNA

        sense_residues = dna.top._chains[0]._residues
        anti_residues = dna.top._chains[1]._residues

        print('sense',sense_residues)
        print('anti ', anti_residues)
        """

    def __init__(self, sequence=None, circular=False):
        self.base_pair_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        self.sequence = sequence
        self.circular = circular
        self.reference_bases = self._load_reference_bases()
        self.traj = self._make_DNA()

    def _load_reference_bases(self):
        """Loads reference bases from PDB files."""
        # import networkx as nx
        # G = DNA.top.to_bondgraph()
        # # check which nodes are not connected to any other node
        # isolated_nodes = list(nx.isolates(G))
        # print(isolated_nodes)
        # # somhow Thymine phospho-oxyo's are not connected to anything
        return {base: md.load_pdb(f'/Users/thor/surfdrive/Projects/genDNA/workspace/pymdna/atomic/BDNA_{base}.pdb') for base in self.base_pair_map.keys()}

    def _make_DNA(self):
        """Creates a DNA topology and trajectory based on the sequence."""
        trajectory = self._create_trajectory()
        topology = self._create_topology()
        return md.Trajectory(trajectory, topology)

    def _create_trajectory(self, translation=0.00001):
        """Creates a DNA trajectory based on the sequence."""
        trajectory_coords = []

        # Sense chain coordinates are added first starting from index 0 to N-1 sequence length
        trajectory_coords.extend(self._get_chain_coordinates(self.sequence, translation))

        # Antisense chain coordinates are addded next, coordinates are flipped, starting from N sequence length to N*2-1
        # Here the first base, N, is the complement of first base in the sense strand, this might not be ideal (have to check what is done in the literature)
        antisense_sequence = [self.base_pair_map[base] for base in self.sequence]
        trajectory_coords.extend(self._get_chain_coordinates(antisense_sequence[::-1], translation, antisense=True))

        return np.concatenate(trajectory_coords, axis=1)

    def _create_topology(self):
        """Creates a DNA topology based on the sequence."""
        topology = md.Topology()
        sense_chain = topology.add_chain()
        antisense_chain = topology.add_chain()

        # Sense chain residues
        self._add_chain_to_topology(self.sequence, topology, sense_chain)

        # Antisense chain residues
        antisense_sequence = [self.base_pair_map[base] for base in self.sequence]
        self._add_chain_to_topology(antisense_sequence[::-1], topology, antisense_chain)

        # Create standard bonds between atoms
        topology.create_standard_bonds()

        # Connect adjacent residues
        self._connect_chain(topology, sense_chain)
        self._connect_chain(topology, antisense_chain)

        # Make the DNA circular if specified
        if self.circular:
            residues_sense = list(sense_chain.residues)
            residues_antisense = list(antisense_chain.residues)
            self._connect_residues(topology, residues_sense[-1], residues_sense[0])
            self._connect_residues(topology, residues_antisense[-1], residues_antisense[0])
            
        return topology

    def _add_chain_to_topology(self, sequence, topology, chain):
        """Adds a chain and its residues to the given topology."""
        for _,base in enumerate(sequence):
            self._add_residue_to_topology(base, topology, chain, resSeq=_+1)

    def _add_residue_to_topology(self, base, topology, chain,resSeq):
        """Adds a residue and its atoms to the given topology."""
        residue = topology.add_residue(name='D' + base, chain=chain, resSeq=resSeq)
        self._copy_atoms_from_reference(base, residue, topology)
   
    def _copy_atoms_from_reference(self, base, residue, topology):
        """Copies atoms from the reference base to the topology."""
        for atom in self.reference_bases[base].topology.atoms:
            topology.add_atom(atom.name, atom.element, residue)

    def _connect_chain(self, topology, chain):
        """Connect adjacent residues of a given chain."""
        residues = list(chain.residues)
        for i in range(len(residues) - 1):
            self._connect_residues(topology, residues[i], residues[i+1])

    def _connect_residues(self, topology, residue1, residue2):
        """Connects two residues in the topology by adding a phosphodiester bond."""
        topology.add_bond(residue1.atom("O3'"), residue2.atom("P"))

    def get_basepair(self, base):
        """Get base pair trajectory based on complementary base."""
        return self.reference_bases[base], self.reference_bases[self.base_pair_map[base]]
    
    def _rotation_matrix(self,axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
    def _get_chain_coordinates(self, sequence, translation, antisense=False):
        """Adds (dummy) coordinates of residues to chains."""
        coords = []
        for base in sequence:
            base_coords = (self.reference_bases[base].xyz + translation) * -1
            if antisense:
                #base_coords = -base_coords # Flip coordinates
                axis = [1, 0, 0]  # Rotate about x-axis
                theta = np.pi  # Rotate by 180 degrees
                # Compute the rotation matrix
                R = self._rotation_matrix(axis, theta)
                # Rotate base pair A (assuming you have it in a variable `base_pair_A`)
                base_coords = np.dot(base_coords, R.T)

            coords.append(base_coords)
        return coords
    
class StructureGenerator:

    """Provides tools for generating and modifying DNA structures.

    Attributes:
    - circular (bool): Indicates whether the DNA is circular or not.
    - sequence (str): The DNA sequence. If not provided, a random sequence is generated.
    - spline (object): An object that defines the spline frames for the DNA.
    - dna (object): Represents the DNA structure based on the provided sequence.
    - traj (object): Represents the DNA trajectory and topology.
    - length (int): Number of base pairs in the DNA.

    Methods:
    - initialize(): Initializes the DNA trajectory and topology.
    - update_basepair_coordinates(old, new, basepairs, idx): Updates the coordinates of a base pair in the DNA trajectory.
    - apply_spline(): Transforms spline frames to coordinates in an mdtraj trajectory.
    - generate_letter_sequence(at_fraction=0.5): Generates a DNA sequence based on AT content fraction.
    - get_basepair_xyz(basepairs, idx): Fetches the xyz coordinates of a base pair.

    Note:
    - The 'initialize' method must be called before any other method to ensure the class attributes are set up correctly.
    """

    def __init__(self,spline=None,sequence=None, circular=False):

        self.circular = circular
        self.sequence = sequence
        self.spline = spline
        self.initialize()
        self.apply_spline()

    def initialize(self):
        """Initialize the DNA trajectory and topology."""
    
        # Number of base pairs
        self.length = self.spline.frames.shape[0]

        # Generate DNA sequence
        if self.sequence is None:
            # Fraction of AT content
            self.sequence = self.generate_letter_sequence(at_fraction=0.5)
            print(self.sequence)

        # Generate DNA trajectory and topology with dummy coordinates (aka everything based on standard bases)
        self.dna = SequenceGenerator(sequence=self.sequence, circular=self.circular)
        self.traj = self.dna.traj

    def update_basepair_coordinates(self, old, new, basepairs, idx):
        """Updates the coordinates of a base pair in the DNA trajectory."""

        # Get the origin of the old and new frames
        old_origin = old[0][0]
        new_origin = new[0]

        # Get basis of the old and new frames
        old_basis = old[0][1:]
        new_basis = np.array(new[1:]) # maybe need to flip the basis vectors here to get the right orientation

        # Get the rotation and translation
        rot = np.linalg.solve(old_basis,new_basis)
        # rot = np.dot(new_basis,old_basis.T) <-- this is the same as above
        trans = new_origin - old_origin

        # First collect xyz coordinates of the base pairs
        xyz, indices = self.get_basepair_xyz(basepairs,idx)

        # Apply the rotation and translation
        new_xyz = np.dot(xyz,rot) + trans

        # Set the new coordinates
        self.traj.xyz[:,indices,:] = new_xyz
        # return dna

    def apply_spline(self):
        """Transforms spline frames to coordinates in mdtraj trajectory."""
        
        # Generate mean reference frames based on dna trajectory (probably generates errors due to overlapping dna bases when computing the base pair parameters, so need to turn that off)
        _ = NucleicFrames(self.traj)

        # Get base pair topology of strands
        sense = self.traj.top._chains[0]._residues
        anti = self.traj.top._chains[1]._residues[::-1]
        basepairs = np.array(list(zip(sense, anti)))[::-1]

        # Set the new coordinates
        current_frames =  _.mean_reference_frames # has shape (n_basepairs, 1, 4, 3) base pairs, time, frames
        new_frames = self.spline.frames #spline.frames # has shape (n_basepairs, 4, 3) base pairs, frames

        # Loop over base pairs and apply the transformation to the base pair coordinates
        for idx,(old,new) in enumerate(zip(current_frames, new_frames)):
            self.update_basepair_coordinates(old, new, basepairs, idx)
        #return dna

    def generate_letter_sequence(self,at_fraction=0.5):
        # Ensure at_fraction is within valid bounds
        if not (0 <= at_fraction <= 1):
            raise ValueError("at_fraction must be between 0 and 1 inclusive")
        # Calculate the number of 'A' and 'T' bases based on the given fraction
        bases = ['A', 'T'] * int(self.length * at_fraction) + ['G', 'C'] * (self.length - int(self.length * at_fraction))
        return ''.join(random.sample(bases, self.length))
    
    def get_basepair_xyz(self,basepairs, idx):
        """Get xyz coordinates of a base pair."""
        indices = [at.index for at in basepairs[idx][0].atoms] + [at.index for at in basepairs[idx][1].atoms]
        return self.traj.xyz[:,indices,:], indices