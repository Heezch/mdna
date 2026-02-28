import numpy as np
import mdtraj as md
import random
from .geometry import NucleicFrames 
from .utils import get_data_file_path

class SequenceGenerator:
    """Build an MDTraj DNA trajectory and topology from a nucleotide sequence.

    Creates a double-stranded DNA structure using pre-loaded canonical and
    non-canonical base reference structures.  Each base is placed at a
    dummy position; the real coordinates are applied later by
    :class:`StructureGenerator`.

    Attributes:
        sequence (str): Sense-strand sequence.
        circular (bool): Whether the DNA is circular.
        traj (md.Trajectory): The generated trajectory with topology.

    Example:
        ```python
        gen = SequenceGenerator(sequence='GCAATATATTGC', circular=False)
        gen.traj  # MDTraj Trajectory with dummy coordinates
        ```
    """

    def __init__(self, sequence=None, circular=False):
        """Initialize the sequence generator.

        Args:
            sequence (str, optional): DNA sequence code (sense strand).
            circular (bool): If True, create a circular DNA topology.
        """
        #self.base_pair_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C','P':'T','D':'C','H':'T'}
        self.base_pair_map = {'A':'T','T':'A','G':'C','C':'G','U':'A','D':'G','E':'T','L':'M','M':'L','B':'S','S':'B','Z':'P','P':'Z'}
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

        return  {base: md.load_hdf5(get_data_file_path(f'atomic/bases/BDNA_{base}.h5')) for base in self.base_pair_map.keys()}

        #return {base: md.load_pdb(f'/Users/thor/surfdrive/Projects/mdna/mdna/atomic/BDNA_{base}.pdb') for base in self.base_pair_map.keys()}
        #return {base: md.load_pdb(f'../mdna/atomic/BDNA_{base}.pdb') for base in self.base_pair_map.keys()}
    
    def _make_DNA(self):
        """Create the combined DNA trajectory and topology.

        Returns:
            md.Trajectory: Trajectory with both sense and anti-sense chains.
        """
        trajectory = self._create_trajectory()
        topology = self._create_topology()
        return md.Trajectory(trajectory, topology)

    def _create_trajectory(self, translation=0.00001):
        """Create coordinate arrays for both strands.

        Args:
            translation (float): Small offset to avoid overlapping reference coordinates.

        Returns:
            numpy.ndarray: Concatenated coordinates for all atoms.
        """
        trajectory_coords = []

        # Sense chain coordinates are added first starting from index 0 to N-1 sequence length
        trajectory_coords.extend(self._get_chain_coordinates(self.sequence, translation))

        # Antisense chain coordinates are added next, coordinates are flipped, starting from N sequence length to N*2-1
        # Here the first base, N, is the complement of first base in the sense strand, this might not be ideal (have to check what is done in the literature)
        antisense_sequence = [self.base_pair_map[base] for base in self.sequence]
        trajectory_coords.extend(self._get_chain_coordinates(antisense_sequence[::-1], translation, antisense=True))

        return np.concatenate(trajectory_coords, axis=1)

    def _create_topology(self):
        """Build the MDTraj topology with sense and anti-sense chains.

        Returns:
            md.Topology: Complete double-stranded DNA topology.
        """
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
        """Add residues for a full strand to the topology.

        Args:
            sequence (str or list): Nucleotide sequence.
            topology (md.Topology): Target topology.
            chain: Chain object to add residues to.
        """
        for _,base in enumerate(sequence):
            self._add_residue_to_topology(base, topology, chain, resSeq=_+1)

    def _add_residue_to_topology(self, base, topology, chain, resSeq):
        """Add a single residue and its atoms to the topology.

        Args:
            base (str): One-letter base code.
            topology (md.Topology): Target topology.
            chain: Chain to add the residue to.
            resSeq (int): Residue sequence number.
        """
        residue = topology.add_residue(name='D' + base, chain=chain, resSeq=resSeq)
        self._copy_atoms_from_reference(base, residue, topology)
   
    def _copy_atoms_from_reference(self, base, residue, topology):
        """Copy atom definitions from the reference base to the new residue.

        Args:
            base (str): One-letter base code.
            residue: Target residue in the topology.
            topology (md.Topology): Target topology.
        """
        for atom in self.reference_bases[base].topology.atoms:
            topology.add_atom(atom.name, atom.element, residue)

    def _connect_chain(self, topology, chain):
        """Add phosphodiester bonds between adjacent residues in a chain.

        Args:
            topology (md.Topology): Target topology.
            chain: Chain whose residues should be connected.
        """
        residues = list(chain.residues)
        for i in range(len(residues) - 1):
            self._connect_residues(topology, residues[i], residues[i+1])

    def _connect_residues(self, topology, residue1, residue2):
        """Add an O3'–P phosphodiester bond between two residues.

        Args:
            topology (md.Topology): Target topology.
            residue1: Upstream residue (provides O3').
            residue2: Downstream residue (provides P).
        """
        topology.add_bond(residue1.atom("O3'"), residue2.atom("P"))

    def get_basepair(self, base):
        """Get reference base-pair trajectories for a base and its complement.

        Args:
            base (str): One-letter base code.

        Returns:
            tuple[md.Trajectory, md.Trajectory]: ``(base_traj, complement_traj)``.
        """
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
        """Generate coordinate arrays for a single strand.

        Args:
            sequence (str or list): Nucleotide sequence.
            translation (float): Small offset applied to reference coordinates.
            antisense (bool): If True, rotate coordinates 180° around the x-axis
                for the anti-sense strand.

        Returns:
            list[numpy.ndarray]: Per-residue coordinate arrays.
        """
        coords = []
        for base in sequence:
            base_coords = (self.reference_bases[base].xyz + translation) #* -1
            if antisense:
                #base_coords = -base_coords # Flip coordinates
                axis = [1, 0, 0]  # Rotate about x-axis
                theta = np.pi  # Rotate by 180 degrees
                # Compute the rotation matrix
                R = self._rotation_matrix(axis, theta)
                # Rotate base pair A (assuming you have it in a variable `base_pair_A`)
                base_coords = np.dot(base_coords, R.T) #* -1 

            coords.append(base_coords)
        return coords
    
class StructureGenerator:
    """Place nucleotide coordinates onto a spatial path defined by rigid-body frames.

    Takes an array of rigid-body frames (origin + three basis vectors per
    base pair) and a nucleotide sequence, then produces an MDTraj trajectory
    with atomic coordinates transformed to follow the path.

    Attributes:
        circular (bool): Whether the DNA is circular.
        sequence (str): Sense-strand sequence (auto-generated if not provided).
        spline (SplineFrames or None): Optional spline that supplies the frames.
        frames (numpy.ndarray): Shape ``(n_bp, 4, 3)`` rigid-body frames.
        length (int): Number of base pairs.
        traj (md.Trajectory): Resulting trajectory with positioned coordinates.

    Example:
        ```python
        from mdna import SplineFrames, StructureGenerator
        spline = SplineFrames(control_points)
        gen = StructureGenerator(spline=spline, sequence='ATCG')
        gen.traj  # MDTraj trajectory with coordinates on the spline
        ```
    """

    def __init__(self, spline=None, sequence=None, circular=False, frames=None):
        """Initialize the structure generator.

        Either *spline* or *frames* must be provided.  If both are given,
        *frames* takes precedence.

        Args:
            spline (SplineFrames, optional): Spline object that provides frames.
            sequence (str, optional): DNA sequence.  When ``None`` a random
                sequence with 50 %% AT content is generated.
            circular (bool): Generate circular DNA topology.
            frames (numpy.ndarray, optional): Rigid-body frames with shape
                ``(n_bp, 4, 3)``.

        Raises:
            ValueError: If neither *spline* nor *frames* is provided.
        """
        self.circular = circular
        self.sequence = sequence # Should deal with the fact if the sequence length does not match the spline length
        self.spline = spline
        if spline is not None:
            self.frames = spline.frames
        elif spline is None and frames is not None:
            self.frames = frames
        elif spline is not None and frames is not None:
            self.frames = frames
            print("Both spline and frames are provided, using frames")
        else:
            raise ValueError("Either spline or frames must be provided")
        
        self.length = self.frames.shape[0] # numer of base pairs 
        self.initialize()
        self.apply_spline()

    def initialize(self):
        """Build the initial trajectory with dummy coordinates.

        Generates a random sequence if none was provided, then creates
        the topology and reference coordinates via :class:`SequenceGenerator`.
        """

        # Generate DNA sequence
        if self.sequence is None:
            # Fraction of AT content
            self.sequence = self.generate_letter_sequence(at_fraction=0.5)
            print(self.sequence)

        # Generate DNA trajectory and topology with dummy coordinates (aka everything based on standard bases)
        self.dna = SequenceGenerator(sequence=self.sequence, circular=self.circular)
        self.traj = self.dna.traj


    def apply_spline(self):
        """Transform dummy base-pair coordinates to follow the spatial frames.

        Iterates over every base pair and applies a rotation and translation
        so that each base pair is positioned and oriented according to the
        corresponding entry in :attr:`frames`.
        """

        # Get base pair topology of strands
        sense = self.traj.top._chains[0]._residues
        anti = self.traj.top._chains[1]._residues[::-1]
        basepairs = np.array(list(zip(sense, anti)))
            
        # Set the dummy coordinates
        initial_frame = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        
        # Create a list of initial frames for each base pair
        basepair_frames = np.array([initial_frame for _ in range(len(basepairs))])  # Shape: (n_basepairs, 4, 3)

        # Add an empty axis to match expected dimensions (n_basepairs, 1, 4, 3)
        current_frames = basepair_frames[:, np.newaxis, :, :]
        new_frames = self.frames #spline.frames # has shape (n_basepairs, 4, 3) base pairs, frames

        # Loop over base pairs and apply the transformation to the base pair coordinates
        for idx,(old,new) in enumerate(zip(current_frames, new_frames)):
            self.update_basepair_coordinates(old, new, basepairs, idx)
            
    def update_basepair_coordinates(self, old, new, basepairs, idx):
        """Rotate and translate a single base pair to a new frame.

        Args:
            old (numpy.ndarray): Current frame for this base pair, shape ``(1, 4, 3)``.
            new (numpy.ndarray): Target frame, shape ``(4, 3)``.
            basepairs (numpy.ndarray): Array of ``(sense_residue, antisense_residue)`` pairs.
            idx (int): Index of the base pair to update.
        """

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

    def generate_letter_sequence(self, at_fraction=0.5):
        """Generate a random DNA sequence with a specified AT content.

        Args:
            at_fraction (float): Fraction of A/T bases (0–1).

        Returns:
            str: Random DNA sequence of length :attr:`length`.

        Raises:
            ValueError: If *at_fraction* is outside [0, 1].
        """
        # Ensure at_fraction is within valid bounds
        if not (0 <= at_fraction <= 1):
            raise ValueError("at_fraction must be between 0 and 1 inclusive")
        # Calculate the number of 'A' and 'T' bases based on the given fraction
        bases = ['A', 'T'] * int(self.length * at_fraction) + ['G', 'C'] * (self.length - int(self.length * at_fraction))
        return ''.join(random.sample(bases, self.length))
    
    def get_basepair_xyz(self, basepairs, idx):
        """Get atom coordinates for a single base pair.

        Args:
            basepairs (numpy.ndarray): Array of ``(sense_residue, antisense_residue)`` pairs.
            idx (int): Base-pair index.

        Returns:
            tuple[numpy.ndarray, list[int]]: ``(xyz, atom_indices)``.
        """
        indices = [at.index for at in basepairs[idx][0].atoms] + [at.index for at in basepairs[idx][1].atoms]
        return self.traj.xyz[:,indices,:], indices

    def get_traj(self, remove_terminal_phosphates: bool = False):
        """Return the DNA trajectory, optionally trimming terminal phosphates.

        Args:
            remove_terminal_phosphates (bool): If True and the structure is
                linear, remove the 5′ phosphate groups at the termini.

        Returns:
            md.Trajectory: DNA trajectory.
        """
        if self.circular or not remove_terminal_phosphates:
            return self.traj

        phosphor_termini = self.traj.top.select(f'name P OP1 OP2 and resid 0 {self.traj.top.chain(0).n_residues}')
        all_atoms = self.traj.top.select('all')
        return self.traj.atom_slice([at for at in all_atoms if at not in phosphor_termini])
