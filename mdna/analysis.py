import numpy as np
import matplotlib.pyplot as plt
try:
    from joblib import Parallel, delayed
    joblib_available = True
except ImportError:
    joblib_available = False
    print("joblib is not installed. Falling back to sequential computation.")
import mdtraj as md
import warnings
from .utils import get_sequence_letters, get_base_pair_letters
from scipy.interpolate import CubicSpline
import mdtraj as md
import matplotlib as mpl

def _compute_distance(xyz, pairs):
    """Compute Euclidean distance between atom pairs for every frame.

    Args:
        xyz (numpy.ndarray): Coordinates with shape ``(n_frames, n_atoms, 3)``.
        pairs (list[tuple[int, int]]): Atom-index pairs.

    Returns:
        numpy.ndarray: Distances with shape ``(n_frames, n_pairs)``.
    """
    delta = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
    return (delta ** 2.).sum(-1) ** 0.5

class GrooveAnalysis:
    """Compute minor and major groove widths from a DNA trajectory.

    Extracts phosphorus-atom positions from the two DNA strands, fits cubic
    splines through them, and measures inter-strand distances to derive
    groove widths for every frame.

    Attributes:
        minor_widths (numpy.ndarray): Minor-groove widths, shape ``(n_frames, n_points)``.
        major_widths (numpy.ndarray): Major-groove widths, shape ``(n_frames, n_points)``.
        sequence (list[str]): Sense-strand sequence letters.
        base_pairs (list[str]): Base-pair labels (e.g. ``['G-C', 'A-T', ...]``).

    Example:
        ```python
        import mdtraj as md
        traj = md.load('trajectory.xtc', top='topology.pdb')
        grooves = GrooveAnalysis(traj)
        fig, ax = plt.subplots()
        grooves.plot_groove_widths(ax=ax)
        ```
    """
    
    def __init__(self, raw_traj, points=None, parallel=joblib_available):
        """Initialize groove analysis.

        Args:
            raw_traj (md.Trajectory): Trajectory containing at least two DNA chains.
            points (int, optional): Number of spline interpolation points.
                Defaults to ``(n_bp - 1) * 4``.
            parallel (bool): Use joblib for parallel computation when available.
        """
        
        self.use_parallel = parallel
        self.points = points
        self.sequence = get_sequence_letters(raw_traj)
        self.base_pairs = get_base_pair_letters(raw_traj)
        self.get_phosphors_only_traj(raw_traj) 
        
        self.nbp = len(self.DNA_traj.top._residues)//2
        
        # Fit cubic spline curves based on phosphor atoms in strands
        if self.points:
            pass
        else:
            self.points = (self.nbp-1)*4
        
        self.fit_cubic_spline()
        self.compute_groove_widths()
    
    def describe(self):
        """Print a summary of the DNA trajectory (base-pair count and frames)."""
        print(f'Your DNA has {self.nbp} base pairs and contains {self.DNA_traj.n_frames} frames.')
     
    def get_phosphors_only_traj(self, raw_traj):
        """Extract phosphorus-only sub-trajectories for each strand.

        Populates :attr:`DNA_traj`, :attr:`phos_traj`, :attr:`strand_A`,
        and :attr:`strand_B`.

        Args:
            raw_traj (md.Trajectory): Full trajectory.
        """

        # Assuming first two chains are the DNA strands
        DNA_indices = [i.index for c in raw_traj.top._chains[:2] for i in c.atoms]
        self.DNA_traj = raw_traj.atom_slice(DNA_indices)

         # Select only phosphor atoms
        phos_indices = self.DNA_traj.top.select('name P')
        self.phos_traj = self.DNA_traj.atom_slice(phos_indices).center_coordinates()
        
        # Make new topology of chains
        phos_chain_A = self.phos_traj.topology._chains[0]
        phos_chain_B = self.phos_traj.topology._chains[1]
        
        # Split phos traj in two seperate traj for each strand
        self.strand_A = self.phos_traj.atom_slice([i.index for i in phos_chain_A.atoms])
        self.strand_B = self.phos_traj.atom_slice([i.index for i in phos_chain_B.atoms])
        
    def fit_cubic_spline(self):
        """Fit cubic splines through phosphorus positions and compute inter-strand distances.

        Populates :attr:`pairs`, :attr:`distances`, and :attr:`distance_matrices`.
        """
    
        # fit cubic spline curve
        curves_A = self.fit_curves(self.strand_A, self.points)
        curves_B = self.fit_curves(self.strand_B, self.points)

        # reshape back to trajectory xyz format
        a_curve_xyz = curves_A.T.swapaxes(1,2)
        b_curve_xyz = curves_B.T.swapaxes(1,2)

        # convert curves to xyz format
        curves = np.hstack((a_curve_xyz,b_curve_xyz))

        # Retrieve "predicted points/particles"
        n_points = curves.shape[1]
        points_strand_A = range(0,(int((n_points/2))))
        points_strand_B = range((int((n_points/2))),(int((n_points))))

        # Generate pairs between the two strands
        #self.pairs = list(itertools.product(points_strand_A, points_strand_B))
        self.pairs = [(point_A, point_B) for point_A in points_strand_A for point_B in points_strand_B]

        # Calculate pair distances
        self.distances = _compute_distance(curves, self.pairs)

        # Reshape pair distances to n x n matrix and account for vdWaals radii of P atoms
        s = self.distances.shape
        self.distance_matrices = self.distances.reshape(s[0],len(points_strand_A),len(points_strand_B))[::-1,:,::-1] - 0.58
    

    def fit_curves(self, traj, points):
        """Interpolate atom positions with a cubic spline.

        Args:
            traj (md.Trajectory): Single-strand phosphorus trajectory.
            points (int): Number of interpolation points.

        Returns:
            numpy.ndarray: Interpolated coordinates.
        """
        
        # make curve with shape[n_particles, xyz, time]
        xyz = traj.xyz.T.swapaxes(0,1)
        n_particles = len(xyz)
        particle_list = list(range(0,n_particles))

        # initialize predictor based on # of actual particles
        predictor = CubicSpline(particle_list,xyz)

        # points of curve to interpolate the particles 
        return predictor(np.linspace(0,len(xyz),points))
    
    @staticmethod
    def find_first_local_minimum(arr):
        """Return the first local minimum in *arr*, or ``NaN`` if none exists.

        Args:
            arr (numpy.ndarray): 1-D array of values.

        Returns:
            float: First local minimum value.
        """
        for i in range(1, len(arr) - 1):
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                return arr[i]  # Return the first local minimum found
        return np.NaN  # Return None if no local minimum is found

    @staticmethod
    def split_array(array):
        """Split an anti-diagonal slice into minor and major halves.

        Args:
            array (numpy.ndarray): Anti-diagonal slice.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: ``(minor_half, major_half)``.
        """
        # Compute the midpoint of the array
        midpoint = len(array) // 2
        
        # If the length of the array is odd, exclude the middle element. 
        # If it's even, split the array into two equal halves.
        # Return the two halves that correspond to lower and upper triangles of the distance matrix
        # Reverse the first array because the order of the anti diagonal's in the opposite direction of backone/trailing diagonal
        return array[:midpoint][::-1], array[midpoint + len(array) % 2:]

    @staticmethod
    def get_anti_diagonal_slices(matrix):
        """Extract all anti-diagonal slices from a square matrix.

        Args:
            matrix (numpy.ndarray): Square distance matrix.

        Returns:
            list[numpy.ndarray]: Anti-diagonal slices.
        """
        n = matrix.shape[0] # Get the size of the matrix and skip the first and last anti diagonal
        return [np.diagonal(np.flipud(matrix), offset) for offset in range(-(n - 2), n - 2)]


    def get_minor_major_widths(self, distance_matrix):
        """Compute minor and major groove widths from one distance matrix.

        Args:
            distance_matrix (numpy.ndarray): Inter-strand distance matrix for a single frame.

        Returns:
            tuple[list[float], list[float]]: ``(minor_widths, major_widths)``.
        """

        # Split the distance matrix into anti diagonal slices
        diagonal_slices = self.get_anti_diagonal_slices(distance_matrix)
        
        minor_widths, major_widths = [],[]
        for slice_ in diagonal_slices:
            minor, major = self.split_array(slice_)
            major_widths.append(self.find_first_local_minimum(major))  # Find local minimum in the major slice and add it to the list
            minor_widths.append(self.find_first_local_minimum(minor))  # Find local minimum in the minor slice and add it to the list 
        
        return minor_widths, major_widths

    def compute_groove_widths(self):
        """Compute groove widths for all frames (parallel or sequential).

        Populates :attr:`minor_widths` and :attr:`major_widths`.
        """
        if self.use_parallel:
            # Parallelize the computation and subtract 0.58 to account for the vdw radius of the phosphorus atoms
            results = Parallel(n_jobs=-1)(delayed(self.get_minor_major_widths)(distance_matrix) for distance_matrix in self.distance_matrices)
        else:
            # Compute sequentially
            results = [self.get_minor_major_widths(distance_matrix) for distance_matrix in self.distance_matrices]

        minor_widths_batch, major_widths_batch = zip(*results)
        self.minor_widths = np.array(minor_widths_batch)
        self.major_widths = np.array(major_widths_batch)


    def plot_width(self, ax, groove, color=None, std=True, ls='-', lw=0.5):
        """Plot mean groove width with optional standard-deviation shading.

        Args:
            ax (matplotlib.axes.Axes): Target axes.
            groove (numpy.ndarray): Groove-width array, shape ``(n_frames, n_points)``.
            color (str, optional): Line colour.
            std (bool): If True, shade the ±1σ region.
            ls (str): Line style.
            lw (float): Line width.
        """
        # Suppress warnings for mean of empty slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)    
            # Calculate the mean and standard deviation of major widths
            mean = np.nanmean(groove, axis=0)
            stds = np.nanstd(groove, axis=0)
    
        ax.plot(mean,color=color,ls=ls,lw=lw)
        if std:
            # Fill the area around the mean for widths
            ax.fill_between(range(len(mean)), mean - stds, mean + stds, alpha=0.25,color=color,ls='-',lw=lw)

    def plot_groove_widths(self, minor=True, major=True, std=True, color='k', c_minor=None, lw=0.5, c_major=None, ax=None, base_labels=True, ls='-'):
        """Plot minor and/or major groove widths.

        Args:
            minor (bool): Plot minor groove widths.
            major (bool): Plot major groove widths.
            std (bool): Show standard-deviation shading.
            color (str): Fallback colour when *c_minor*/*c_major* are not set.
            c_minor (str, optional): Minor-groove colour.
            c_major (str, optional): Major-groove colour.
            lw (float): Line width.
            ax (matplotlib.axes.Axes, optional): Target axes.  A new figure
                is created if ``None``.
            base_labels (bool): Label the x-axis with base-pair identifiers.
            ls (str): Line style.

        Returns:
            tuple or None: ``(fig, ax)`` when *ax* was ``None``.
        """

        # Create a figure and axes for plotting
        if ax is None:
            no_ax = True
            _, ax = plt.subplots()
        else:
            no_ax = False

        # If c_minor or c_major are not provided, use the general color
        if c_minor is None:
            c_minor = color
        if c_major is None:
            c_major = color
        if (minor and major) and (c_minor == c_major):
            c_minor = 'cornflowerblue'
            c_major = 'coral'

        if minor:
            self.plot_width(ax, self.minor_widths, std=std, color=c_minor,ls=ls,lw=lw)
        if major:
            self.plot_width(ax, self.major_widths, std=std, color=c_major,ls=ls,lw=lw)

        if base_labels:
            ax.set_xticks(np.linspace(0,len(self.major_widths[0]),len(self.base_pairs)).astype(float))
            ax.set_xticklabels(self.base_pairs,rotation=90)

        if no_ax:
            return _, ax

    def calculate_groove_depths(self):
        # Having defined a groove width by a minimal distance at some point along the nucleic acid fragment,
        # we have to calculate the corresponding groove depth. 

        # At a base pair level, this is defined as the distance from the centre of 
        # the backbone-to-backbone width vector to the mid-point of a vector defining the corresponding base pair. 
        # This vector is constructed using the C8 atom of purines and the C6 atom of pyrimidines 

        # For groove depths half-way between base pair levels,
        # we use the average of the corresponding base pair vector mid-points. 
        pass

    def calculate_groove_depths(self):
        
        # Calculate the C8 and C6 atom positions of purines and pyrimidines
        c8_indices = self.DNA_traj.top.select('name C8')
        c6_indices = self.DNA_traj.top.select('name C6')
        c8_positions = self.DNA_traj.xyz[:, c8_indices]
        c6_positions = self.DNA_traj.xyz[:, c6_indices]

        # Calculate mid-points of vectors defining base pairs
        base_pair_mid_points = 0.5 * (c8_positions + c6_positions)

        # Calculate backbone-to-backbone width vector centres
        backbone_width_vector_centres = 0.5 * (self.strand_A.xyz + self.strand_B.xyz)

        # Calculate groove depths at base pair level
        groove_depths_base_pair_level = np.linalg.norm(base_pair_mid_points - backbone_width_vector_centres, axis=-1)

        # For groove depths half-way between base pair levels, we use the average of the corresponding base pair vector mid-points
        groove_depths_half_way = 0.5 * (groove_depths_base_pair_level[:-1] + groove_depths_base_pair_level[1:])

        return groove_depths_base_pair_level, groove_depths_half_way

class TorsionAnalysis:
    """Compute backbone torsion angles and BI/BII conformational states.

    Calculates the epsilon and zeta dihedral angles of the DNA backbone
    and classifies each base step as BI or BII based on the sign of
    epsilon − zeta.

    Attributes:
        epsilon (numpy.ndarray): Epsilon torsion angles, shape ``(n_frames, n_steps)``.
        zeta (numpy.ndarray): Zeta torsion angles, shape ``(n_frames, n_steps)``.
        B_state (numpy.ndarray): Fraction of BII per base step.

    Example:
        ```python
        torsions = TorsionAnalysis(traj)
        torsions.B_state  # BII fraction per base step
        ```
    """

    def __init__(self, traj, degrees=True, chain=0):
        """Initialize torsion analysis.

        Args:
            traj (md.Trajectory): DNA-containing trajectory.
            degrees (bool): Report angles in degrees (default) or radians.
            chain (int): Chain index to analyse (0 = sense, 1 = anti-sense).
        """
        self.chain = chain
        self.dna = self.load_trajectory_and_slice_dna(traj)
        self.epsilon, self.zeta = self.compute_BI_BII(degrees=degrees)
        self.B_state = self.get_B_state(self.epsilon - self.zeta)

    def load_trajectory_and_slice_dna(self, traj):
        """Slice a trajectory to keep only canonical DNA residues.

        Args:
            traj (md.Trajectory): Input trajectory.

        Returns:
            md.Trajectory: Sub-trajectory containing DG, DC, DA, and DT residues.
        """
        dna = traj.atom_slice(traj.top.select('resname DG DC DA DT'))
        return dna
        
    def get_backbone_indices(self, chainid, ref_atoms):
        """Collect backbone atom objects matching *ref_atoms* from a chain.

        Args:
            chainid (int): Chain index.
            ref_atoms (list[str]): Atom names to select (e.g. ``["C3'", "O3'"]``).

        Returns:
            list: Atom objects.
        """
        indices = []
        # find torsions based on the epsilon and zeta atoms
        # finally map the torsions for all base steps 
        if chainid == 0:
            residues = self.dna.top._chains[chainid].residues
        else:
            residues = self.dna.top._chains[chainid]._residues
            
        for res in residues:
            for at in res.atoms:
                if at.name in ref_atoms:
                    indices.append(at)
        return indices

    def get_torsions(self, indices, ref_atoms):
        """Group sequential atoms into torsion-angle quartets.

        Args:
            indices (list): Ordered backbone atom objects.
            ref_atoms (list[str]): Reference atom-name pattern for one torsion.

        Returns:
            list[list]: Each element is a four-atom group defining one torsion.
        """
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

    def get_torsion_indices(self, chainid, ref_atoms):
        """Get torsion-angle atom groups for a chain.

        Args:
            chainid (int): Chain index.
            ref_atoms (list[str]): Atom-name pattern.

        Returns:
            list[list]: Torsion atom groups.
        """
        indices = self.get_backbone_indices(chainid, ref_atoms)
        torsions = self.get_torsions(indices, ref_atoms)
        return torsions

    def convert_torsion_indices_to_atom_indices(self, torsion_indices):
        """Convert atom objects to integer indices for MDTraj.

        Args:
            torsion_indices (list[list]): Torsion atom-object groups.

        Returns:
            list[list[int]]: Integer atom indices.
        """
        atom_indices = []
        for torsion in torsion_indices:
            atom_indices.append([at.index for at in torsion])
        return atom_indices

    def compute_BI_BII(self, degrees=True):
        """Compute epsilon and zeta backbone torsion angles.

        Args:
            degrees (bool): Return angles in degrees.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: ``(epsilon, zeta)`` arrays.
        """

        epsilon_atoms = ["C4'","C3'","O3'","P"] 
        zeta_atoms = ["C3'","O3'","P","O5'"]

        epsi_0 = self.get_torsion_indices(0, epsilon_atoms)
        epsi_1 = self.get_torsion_indices(1, epsilon_atoms)
        zeta_0 = self.get_torsion_indices(0, zeta_atoms)
        zeta_1 = self.get_torsion_indices(1, zeta_atoms)

        print(len(epsi_0), len(epsi_1), len(zeta_0), len(zeta_1))

        # From here only the antisense strand is used
        if self.chain == 1:
            e_torsion_indices = self.convert_torsion_indices_to_atom_indices(epsi_1)
            z_torsion_indices = self.convert_torsion_indices_to_atom_indices(zeta_1)
        else:
            e_torsion_indices = self.convert_torsion_indices_to_atom_indices(epsi_0)
            z_torsion_indices = self.convert_torsion_indices_to_atom_indices(zeta_0)

        epsi = md.compute_dihedrals(self.dna, e_torsion_indices)
        zeta = md.compute_dihedrals(self.dna, z_torsion_indices)

        if degrees:
            epsi = np.degrees(epsi)
            zeta = np.degrees(zeta)

        print(epsi.shape, zeta.shape)
        return epsi, zeta
    
    def get_B_state(self, diff):
        """Classify each base step as BI (0) or BII (1) from epsilon−zeta.

        Args:
            diff (numpy.ndarray): ``epsilon - zeta`` array.

        Returns:
            numpy.ndarray: BII fraction per base step.
        """
        state = np.zeros_like(diff)
        state[diff < 0] = 0  # BI
        state[diff > 0] = 1  # BII
        return np.round(np.sum(state,axis=0)/state.shape[0],2)
    
    # def place_holder(self):
    #     def get_B_state(diff):
    #         state = np.zeros_like(diff)
    #         state[diff < 0] = 0  # BI
    #         state[diff > 0] = 1  # BII
    #         return np.round(np.sum(state,axis=0)/state.shape[0],2)

    #     from matplotlib.lines import Line2D

    #     fig,ax = plt.subplots(ncols=2,nrows=epsi_t_haff.T.shape[0],figsize=(6,12),sharex=True,sharey=True)

    #     for _,(e,z)in enumerate(zip(epsi_t_haff.T,zeta_t_haff.T)):
    #         d = e-z
    #         sns.kdeplot(d,ax=ax[_][0],fill=True,color='navy',label=get_B_state(d))

    #     for _,(e,z)in enumerate(zip(epsi_d_haff.T,zeta_d_haff.T)):
    #         d = e-z
    #         sns.kdeplot(d,ax=ax[_][0],fill=True,color='cornflowerblue',label=get_B_state(d))

    #     for _,(e,z)in enumerate(zip(epsi_t_gca.T,zeta_t_gca.T)):
    #         d = e-z
    #         sns.kdeplot(d,ax=ax[_][1],fill=True,color='darkred',label=get_B_state(d))

    #     for _,(e,z)in enumerate(zip(epsi_d_gca.T,zeta_d_gca.T)):
    #         d = e-z
    #         sns.kdeplot(d,ax=ax[_][1],fill=True,color='coral',label=get_B_state(d))

    #         ax[_][0].axvline(0,color='gray',ls=':')
    #         ax[_][1].axvline(0,color='gray',ls=':')

    #     for _ in range(epsi_t_haff.T.shape[0]):
    #         ax[_][1].legend()
    #         ax[_][0].legend()
    #         ax[_][0].set_ylabel(f'Step {_}')

    #     ax[_][1].set_xticks([-90,0,90])
    #     ax[_][0].set_xlim(-181,181)
    #     ax[0][0].set_title('High Affinity')
    #     ax[0][1].set_title('GC-analogue')

    #     # Define custom legend
    #     legend_elements = [Line2D([0], [0], color='cornflowerblue', lw=2, label='DNA-haff'),
    #                     Line2D([0], [0], color='coral', lw=2, label='DNA-gca'),
    #                     Line2D([0], [0], color='navy', lw=2, label='FI-haff'),
    #                     Line2D([0], [0], color='darkred', lw=2, label='FI-gca')]


    #     # Add the custom legend to the figure (NOT the subplot)
    #     fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 0.89),title='BII fraction')
    #     fig.suptitle('Denstities of Anti Strand')
    #     fig.savefig('Anti_BII_densities.png',dpi=300,bbox_inches='tight')

class ContactCount:
    """Compute protein–DNA contacts using a smooth switching function.

    Measures distances between user-specified protein donor atoms and
    DNA groove acceptor atoms, then applies a rational switching function
    to obtain a continuous contact count.

    Attributes:
        contacts (numpy.ndarray): Contact values for all pairs, shape ``(n_frames, n_pairs)``.
        contact_matrix (numpy.ndarray): Contacts reshaped to ``(n_frames, n_protein, n_dna)``.

    Example:
        ```python
        protein_donors = {'GLN112': ['N', 'NE2'], 'ARG114': ['N', 'NE', 'NH1', 'NH2']}
        minor_acceptors = {'DA': ['N3'], 'DT': ['O2'], 'DC': ['O2'], 'DG': ['N3']}
        contacts = ContactCount(traj, protein_donors, minor_acceptors)
        ```
    """
    
    def __init__(self, traj, protein_queue, dna_haystack, d0=0.25, r0=0.4, nn=2, mm=4):
        """Initialize contact analysis.

        Args:
            traj (md.Trajectory): Trajectory with protein and DNA.
            protein_queue (dict[str, list[str]]): Protein residue–atom mapping,
                e.g. ``{'GLN112': ['N', 'NE2']}``.
            dna_haystack (dict[str, list[str]]): DNA residue–atom mapping for
                groove atoms, e.g. ``{'DA': ['N3']}``.
            d0 (float): Contact distance offset (nm).
            r0 (float): Contact distance scale (nm).
            nn (int): Exponent in the numerator of the switching function.
            mm (int): Exponent in the denominator of the switching function.
        """
        # store trajectory and topology information
        self.traj = traj
        self.top = traj.topology
        self.atom_names = [at for at in map(str, self.top.atoms)]
        self.protein_queue = protein_queue
        self.dna_haystack = dna_haystack
        
        # store parameters
        self.d0=d0
        self.r0=r0
        self.nn=nn
        self.mm=mm        

        # get indices of protein and dna atoms
        self.protein_indices = self.get_protein_indices()
        self.groove_indices = self.get_groove_indices()
        
        # compute distances and contacts
        self.pairs =  np.array([[k,l]for k in self.protein_indices for l in self.groove_indices])
        self.distances = md.geometry.compute_distances(self.traj, self.pairs)
        self.contacts = self.compute_contacts()
        self.contact_matrix = self.get_contact_matrix()

        # collect sections of protein  (for plotting)
        self.collect_sections()
    
    def get_protein_indices(self):
        """Get atom indices for the specified protein donor atoms.

        Returns:
            list[int]: Atom indices.
        """
        # Find protein indices corresponding to queue (Restype-Atomtype) 
        return [self.atom_names.index(res+'-'+i) for res,at in self.protein_queue.items() for i in at]
    
    def get_groove_indices(self):
        """Get atom indices for the specified DNA groove acceptor atoms.

        Returns:
            list[int]: Sorted atom indices.
        """
        # Find selection of atom types for each nucleobase
        return sorted(sum([list(self.top.select(f"resname {res} and {' '.join(['name '+i for i in at])}")) for res,at in self.dna_haystack.items()],[]))
    
    def smooth_contact(self, r):
        """Evaluate the rational switching function for distances *r*.

        Args:
            r (numpy.ndarray): Distance array.

        Returns:
            numpy.ndarray: Continuous contact values in [0, 1].
        """
        # Compute contact based on distance smoothing function
        return ((1 - ((r-self.d0)/self.r0)**self.nn ) / (1 - ( (r-self.d0)/self.r0)**self.mm) )

    def compute_contacts(self):
        """Apply the switching function to all pair distances.

        Returns:
            numpy.ndarray: Contact values, shape ``(n_frames, n_pairs)``.
        """
        # Check where first condition holds
        ones = np.where(self.distances-self.d0 <= 0)
        # Apply second condition
        contacts = np.where(self.distances-self.d0 >= 0, self.smooth_contact(self.distances),self.distances)
        # Apply second condition (...)
        contacts[ones] = np.ones(ones[0].shape)
        return contacts

    def get_total_contacts(self):
        """Sum contacts over all pairs per frame.

        Returns:
            numpy.ndarray: Total contacts per frame.
        """
        return np.sum(self.contacts,axis=1)
    
    def get_protein_names(self):
        """Return atom names for the selected protein donors."""
        return [self.atom_names[idx] for idx in self.protein_indices]

    def get_dna_names(self):
        """Return atom names for the selected DNA acceptors."""
        return [self.atom_names[idx] for idx in sorted(self.groove_indices)]

    def get_distance_matrix(self):
        """Reshape pair distances into a protein × DNA matrix.

        Returns:
            numpy.ndarray: Shape ``(n_frames, n_protein, n_dna)``.
        """
        # Reshape pair distances to n x n matrix
        s = self.distances.shape
        return self.distances.reshape(s[0],len(self.protein_indices),len(self.groove_indices))
    
    def get_contact_matrix(self):
        """Reshape contact values into a protein × DNA matrix.

        Returns:
            numpy.ndarray: Shape ``(n_frames, n_protein, n_dna)``.
        """
        # Reshape pair distances to n x n matrix
        s = self.contacts.shape
        return self.contacts.reshape(s[0],len(self.protein_indices),len(self.groove_indices))
    
    def collect_sections(self):
        """Compute section boundaries for splitting contacts per protein residue."""
        section_ends = []
        count = 0
        for residue in self.protein_queue.keys():
            count += len(self.protein_queue[residue])
            section_ends.append(count)
        self.sections = section_ends[:-1]

    def split_data(self):
        """Split the contact matrix by protein residue.

        Returns:
            list[numpy.ndarray]: One sub-matrix per protein residue.
        """
        return np.split(self.contact_matrix,self.sections,axis=1)
        
    def get_contacts_per_residue(self):
        """Total contacts per protein residue across all DNA atoms.

        Returns:
            numpy.ndarray: Shape ``(n_residues, n_frames)``.
        """
        return np.array([np.sum(d,axis=(1,2)) for d in self.split_data()])
    
    def get_contacts_per_residue_per_base(self):
        """Contacts per protein residue resolved by DNA base.

        Returns:
            numpy.ndarray: Shape ``(n_residues, n_frames, n_bases)``.
        """
        return np.array([np.sum(d,axis=1) for d in self.split_data()])
            
    def get_contacts_per_base(self):
        """Total contacts per DNA base (summed over protein residues).

        Returns:
            numpy.ndarray: Shape ``(n_bases, n_frames)``.
        """    
        contacts_per_residue_per_base = self.get_contacts_per_residue_per_base()
        return np.sum(contacts_per_residue_per_base,axis=0).T     
                         
    def get_contacts_per_bp(self):
        """Total contacts per base pair (sense + anti-sense combined).

        Returns:
            numpy.ndarray: Shape ``(n_bp, n_frames)``.
        """
        contacts_per_base = self.get_contacts_per_base()
        n_bases = len(contacts_per_base)
        return np.array([a+b for a,b in zip(contacts_per_base[:n_bases//2],contacts_per_base[n_bases//2:][::-1])])
    
    def get_contacts_per_residue_per_bp(self):
        """Contacts per protein residue per base pair.

        Returns:
            numpy.ndarray: Shape ``(n_residues, n_frames, n_bp)``.
        """
        contacts_per_residue_per_base = self.get_contacts_per_residue_per_base()
        n_bases = len(contacts_per_residue_per_base.T)
        return np.array([a+b for a,b in zip(contacts_per_residue_per_base.T[:n_bases//2], contacts_per_residue_per_base.T[n_bases//2:][::-1])]).T

    def check_axis(self,ax):
        if ax is None:
            fig,ax = plt.subplots(figsize=(8,8))
        return fig,ax
    
    def plot_contact_map(self, ax=None, frame=-1):
        """Plot the contact map as a heatmap.

        Args:
            ax (matplotlib.axes.Axes, optional): Target axes.
            frame (int): Frame index to plot.  Use ``-1`` for the time-averaged map.
        """
        fig,ax = self.check_axis(ax)
        contact_matrices = self.get_contact_matrix()
        if frame == -1:
            im = ax.imshow(np.mean(contact_matrices,axis=0),vmin = np.min(self.contacts), vmax = np.max(self.contacts),aspect='auto')
        else:
            im = ax.imshow(contact_matrices[frame],vmin = np.min(self.contacts), vmax = np.max(self.contacts),aspect='auto')


        protein_labels = self.get_protein_names()
        dna_labels = self.get_dna_names()

        ax.set_yticks(range(0,len(protein_labels)))
        ax.set_yticklabels(protein_labels)

        ax.set_xticks(range(0,len(dna_labels)))
        ax.set_xticklabels(dna_labels)
        ax.tick_params(axis="x", rotation=80)
        ax.set_title(f'Contact map of frame {frame}')
        plt.colorbar(im,ax=ax,label="$C_{Protein}$")
        
    # def plot_contact_distribution(self,ax=None,c='Red'):
    #     fig,ax = self.check_axis(ax)
    #     total_contacts = self.get_total_contacts()
    #     df = pd.DataFrame(total_contacts)

    #     data = pd.DataFrame({
    #             "idx": np.tile(df.columns, len(df.index)),
    #             "$C_{Protein-DNA}$": df.values.ravel()})

    #     sns.kdeplot(
    #        data=data, y="$C_{Protein-DNA}$", legend = False, color=c,#hue="idx",
    #        fill=True, common_norm=False, palette="Reds",
    #        alpha=.5, linewidth=1, ax=ax)
        
    def ns_to_steps(self,ns=1):
        # assume a timestep of 2 fs
        return int((ns*1000)/0.002)