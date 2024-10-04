import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import nglview as nv
import networkx as nx

# https://biopython.org/docs/1.74/api/Bio.SVDSuperimposer.html
# conda install conda-forge::biopython
from Bio.SVDSuperimposer import SVDSuperimposer


class SiteMapper:

    def __init__(self, s1s1, s2s2, segments=dict, k=100):
        
        self.segments = segments
        self.s1s1 = s1s1[::k]
        self.s2s2 = s2s2[::k]
        #self.get_site_map()
        
    def get_site_map(self):
        
        # get structures of the different sites
        s1 = self.get_site_structures(self.s1s1,site='s1')
        s2_from_s1 = self.get_site_structures(self.s1s1, site='s2')
        s2 = self.get_site_structures(self.s2s2,site='s2')

        h3_s1s1 = self.get_segment_structures(self.s1s1,site='h3')
        h3_s2s2 = self.get_segment_structures(self.s2s2,site='h3')
        h3 = md.join([h3_s1s1,h3_s2s2])

        l2_s1s1 = self.get_segment_structures(self.s1s1,site='l2')
        l2_s2s2 = self.get_segment_structures(self.s2s2,site='l2')
        l2 = md.join([l2_s1s1,l2_s2s2])

        dbd_s1s1 = self.get_segment_structures(self.s1s1,site='dbd')
        dbd_s2s2 = self.get_segment_structures(self.s2s2,site='dbd')
        dbd = md.join([dbd_s1s1,dbd_s2s2])

        # map of the different sites
        site_map = {'s1':s1,
                'h3':h3,
                's2':s2,
                's2_end' : s2_from_s1,
                'l2':l2,
                'dbd':dbd}
        
        # fix resSeq numbering for second chain of s1 and s2
        for c in site_map['s1'].top.chains:
            if c.index == 1:
                for res in c.residues:
                    res.resSeq = res.resSeq - 137

        for c in site_map['s2'].top.chains:
            if c.index == 1:
                for res in c.residues:
                    res.resSeq = res.resSeq - 137
                
        return site_map

    def check_selection(self,top,selection):
        if selection == 'CA':
            indices = top.select('name CA')
        elif selection == 'backbone':
            indices = top.select('backbone')
        elif selection == 'sidechain':
            indices = top.select('sidechain')
        else:
            indices = top.select('all')   
        return indices 
    
    def get_monomer_domain_indices(self,top,domain,chain=0,selection=None):
        residues = np.array(top._chains[chain]._residues)
        indices = self.check_selection(top,selection)
        return [at.index for res in residues[domain] for at in res.atoms if at.index in indices]

    def get_segment_structures(self,traj,site='dbd'):
        chain_a = self.get_monomer_domain_indices(top=traj.top, domain=self.segments[site], chain=0, selection=None)
        chain_b = self.get_monomer_domain_indices(top=traj.top, domain=self.segments[site], chain=1, selection=None)
        A = traj.atom_slice(chain_a)
        B = traj.atom_slice(chain_b)
        return md.join([A,B])

    def get_site_structures(self, traj,site='s1'):
        chain_a = self.get_monomer_domain_indices(top=traj.top, domain=self.segments[site], chain=0, selection=None)
        chain_b = self.get_monomer_domain_indices(top=traj.top, domain=self.segments[site], chain=1, selection=None)
        return traj.atom_slice(np.sort(chain_a+chain_b))

    def show_domain(self, system, domains, domain):
        """"Not working yet, need to fix the selection of the atoms in the domain."""
        # shows first frame
        top = system.top
        view = nv.show_mdtraj(system[0])
        view.clear()
        indices = self.get_monomer_domain_indices(top, domains[domain], chain=0)
        view.add_representation('cartoon',selection=[i for i in  top.select('all') if i not in indices],color='cornflowerblue')
        top = system.topology
        chain_id = 0
        indices = self.get_monomer_domain_indices(top, domains[domain], chain=chain_id)
        view.add_representation('cartoon',selection=indices,color='gold')
        top = system.topology
        chain_id = 1
        indices = self.get_monomer_domain_indices(top, domains[domain], chain=chain_id)
        view.add_representation('cartoon',selection=indices,color='red')
        return view
    
class Superimposer:

    def __init__(self, A, B, overlap_A, overlap_B):
        self.A = A
        self.B = B
        self.overlap_A = overlap_A
        self.overlap_B = overlap_B

    def get_rot_and_trans(self, subtraj_A,subtraj_B):
        
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

    def apply_superimposition(self, traj, rot, tran):
        
        # get xyz coordinates
        xyz = traj.xyz[0]
        
        # rotate subject on target
        new_xyz = np.dot(xyz, rot) + tran

        # replace coordinates of traj
        traj.xyz = new_xyz
        return traj

    def fit_B_on_A(self):
        # create trajs containing only the selections
        subtraj_A = self.A.atom_slice(self.overlap_A)
        subtraj_B = self.B.atom_slice(self.overlap_B)

        # obtain instructions to rotate and translate B on A based on substraj structures
        rot, tran, _ = self.get_rot_and_trans(subtraj_A,subtraj_B)

        # do the superimposition of B on A and subsitute old with new xyz of B
        return self.apply_superimposition(self.B, rot, tran)

class Helper:
    
    @staticmethod
    def check_if_dimerization(site):
        if 's' in site:
            return True
        else:
            return False
        
    @staticmethod
    def get_termini(site_x,site_y):
        chain_order = np.array(['s1','h3','s2','l2','dbd'])
        x = np.argwhere(chain_order==site_x)
        y = np.argwhere(chain_order==site_y)
        if x < y:
            return ['N_terminus','C_terminus']
        elif x > y:
            return ['C_terminus','N_terminus']

    @staticmethod
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

    @staticmethod
    def check_overlaps(overlap_A,overlap_B):

        if len(overlap_A) != len(overlap_B):
            print(len(overlap_A),len(overlap_B))
            print('Something went wrong with finding the overlaps') 
        else:
            False

    @staticmethod
    def remove_overlap_old(traj, overlap):
        """Works fine but is slow for large number of atoms... because top and traj get reinitialized"""
        return traj.atom_slice([at.index for at in traj.top.atoms if at.index not in overlap])
    
    @staticmethod
    def remove_overlap(traj, overlap):
        indices_to_keep = [at.index for at in traj.top.atoms if at.index not in overlap]
        xyz = np.array(traj.xyz[:, indices_to_keep], order='C')
        for _,index in enumerate(overlap):
            traj.top.delete_atom_by_index(index-_)
        traj.xyz = xyz
        return traj

    @staticmethod    
    def split_chain_topology(traj, leading_chain):
        # split part of A in chain that is being extended and that is not
        traj_active = traj.atom_slice(traj.top.select(f'chainid {leading_chain}'))
        traj_passive = traj.atom_slice(traj.top.select(f'not chainid {leading_chain}'))
        return traj_active, traj_passive

    @staticmethod
    def merge_chain_topology(A, B, keep_resSeq=True):
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

class Fixer:

    def __init__(self, traj):

        segments = {'s1':np.arange(0,41),
                     's2':np.arange(53,82)}
        
        # Figure out which chains are connected 
        G = self.compute_interaction_graph(traj, segments)
            
        # Traverse over graph for new chain assignements
        chain_mapping = self.traverse_from_endpoint(G)

        # Update chain order in topology
        new_topology, atom_mapping  = self.update_chain_topology(traj, chain_mapping)

        # Update xyz coordinates
        new_xyz =  traj.xyz[:,atom_mapping]

        # Create new trajectory with corrected order and adjust xyz as well
        new_traj = md.Trajectory(new_xyz,new_topology)
        self.new_traj = new_traj

    def get_updated_traj(self):
        return self.new_traj

    def traverse_from_endpoint(self, G):

        # Find all nodes with degree 1 (endpoints)
        endpoints = [node for node, degree in G.degree() if degree == 1]

        # Choose the first endpoint as the start node
        start_node = endpoints[0] if endpoints else None
        # Initialize a dictionary to store the number of steps to each node
        # chain_mapping = {node: float('inf') for node in G.nodes()}
        D = []
        for node in G.nodes:
            d = nx.shortest_path_length(G, source=start_node, target=node)
            D.append(d)
        chain_mapping = {i:j for i,j in zip(G.nodes, np.argsort(D))}
        return chain_mapping

    def compute_chain_centers(self, traj, domain):
        top = traj.top
        coms = []
        lens = []
        ids =   []
        for c in top.chains:
            try:
                selection = top.select(f'chainid {c.index} and resSeq {domain[0]} to {domain[-1]}')
                com = md.compute_center_of_mass(traj.atom_slice(selection))
                coms.append(com)
                lens.append(len(selection))
                ids.append(c.index)
            except:
                pass
        coms = np.squeeze(np.array(coms))
        lens = np.array(lens)
        ids = np.array(ids)
        return coms, lens, ids

    def compute_interaction_graph(self, traj, segments):

        # Compute COMS of each chain domain and get chain labels
        s1_centers = self.compute_chain_centers(traj, segments['s1'])
        s2_centers = self.compute_chain_centers(traj, segments['s2'])


        # Initialize graph
        G = nx.Graph()
        labels = {}
        for c in traj.top.chains:
            G.add_node(c.index,label=c.index)
            labels[c.index] = c.index

        # Loop over centers
        for idx,center in enumerate([s1_centers, s2_centers]):
            coms = center[0]
            ids = center[2]

            # Computer distance between coms
            D = np.zeros((len(coms),len(coms)))
            for i,ci in enumerate(coms):
                for j,cj in enumerate(coms):
                    d = np.linalg.norm(ci-cj)
                    D[i,j] = d

            # Use closest pairs to collect edges
            edges = []
            for _,d in enumerate(D):
                pair = np.sort([ids[_],ids[np.argsort(d)[1]]])
                edges.append(pair)
            
            # Filter pairs for redudancy
            edges = np.unique(edges,axis=0)
            for pair in edges:
                G.add_edge(pair[0],pair[1])
        return G

    def update_chain_topology(self, traj, chain_mapping):

        # Initialize empty top
        new_top = md.Topology()

        # Collect current chains
        chains = list(traj.top.chains)
        atom_mapping = []

        # Loop over chain mapping 
        for new, current in chain_mapping.items():
            new_chain = new_top.add_chain() # add empty chain
            chain = chains[current]
            for res in chain.residues: # fill chain with resdues
                new_res = new_top.add_residue(res.name, new_chain, res.resSeq,res.segment_id)
                for atom in res.atoms: # fill residue with atoms
                    new_top.add_atom(atom.name, atom.element, new_res, serial=atom.serial)
                    atom_mapping.append(atom.index) # keep track of new index order 
                    
        # Return mapping and top
        return new_top, atom_mapping


class Assembler:

    def __init__(self, site_map, n_overlap : int = 2):
        
        self.traj = None
        self.chain_id = 0
        self.site_map = site_map
        self.n = n_overlap
        self.n_dimers = 0
        self.n_dna = 0
        self.first = True
        self.s1_pairs = [['s1','h3'],['h3','s2'],['s2','l2'],['l2','dbd']]
        self.s2_pairs = [['s2','h3'],['h3','s1'],['s2','l2'],['l2','dbd']]
        self.traj_history = []
        self.cleaned = False

    def get_traj(self):
        if self.cleaned:
            return self.traj
        else:
            self.clean_traj()
            return self.traj
        # if self.n_dna != 0:
        #     subtraj_dna = self.traj.atom_slice(self.traj.top.select('resname DG DC DA DT'))
        #     subtraj_protein  = self.traj.atom_slice(self.traj.top.select('protein'))
        #     traj = subtraj_protein.atom_slice(subtraj_protein.top.select(f'chainid 0 to {(self.n_dimers*2)-1}'))
        #     return traj.stack(subtraj_dna)
        # else:
        #     return self.traj.atom_slice(self.traj.top.select(f'chainid 0 to {(self.n_dimers*2)-1}'))

    def add_dimer(self, verbose: bool = False, segment: str = 'random'):
        """
        Adds a dimer to the trajectory by iterating over s2_pairs first and then s1_pairs, with specific increments to chain_id.
        """
        self.n_dimers += 1
        # print('Start processing s2 pairs')
        # print(self.s2_pairs)
        self.traj = self.process_pairs(self.s2_pairs, self.chain_id, verbose, segment)
        # print('Start processing s1 pairs')
        # print(self.s1_pairs)
        self.traj = self.process_pairs(self.s1_pairs, self.chain_id + 2,verbose, segment)
        self.chain_id += 2
        return self.traj

    def process_pairs(self, pairs, chain_id, verbose, segment):
        """
        Processes a sequence of pairs, adding each to the trajectory.
        """        
        for idx, pair in enumerate(pairs):
            leading_chain = chain_id if idx == 0 else 0
            self.traj = self.add_pair(pair, leading_chain=leading_chain, verbose=verbose, segment=segment)
            
        return self.traj

    # Get segments based on 'fixed' or 'random' segment criteria
    def get_segments(self, site_a, site_b, segment):
        if not self.traj:
            if segment == 'fixed':
                x, y = 40, 90
            elif segment == 'random':
                k = len(self.site_map[site_a])
                l = len(self.site_map[site_b])
                x, y = np.random.randint(0, k), np.random.randint(0, l)
            A = self.site_map[site_a][x]
            B = self.site_map[site_b][y]
        else:
            if segment == 'fixed':
                z = 20
            elif segment == 'random':
                k = len(self.site_map[site_b])
                z = np.random.randint(0, k)
            A = self.traj
            B = self.site_map[site_b][z]
        return A, B

    # Check for dimerization and print if verbose
    def check_dimerization(self, site_a, site_b):
        dimer_a = Helper.check_if_dimerization(site_a)
        dimer_b = Helper.check_if_dimerization(site_b)
        return dimer_a, dimer_b

    # Determine growth direction based on terminus
    def determine_terminus_direction(self, site_a, site_b):
        terminus_a, terminus_b = Helper.get_termini(site_a, site_b)
        reverse = terminus_a == 'C_terminus'
        return reverse, terminus_a, terminus_b

    # Manage overlaps and perform superimposition
    def manage_overlaps(self, A, B, leading_chain, adding_chain, terminus_a, terminus_b):
       
        overlap_A = Helper.get_overlap_indices(A.top, self.n, chain=leading_chain, terminus=terminus_a)
        overlap_B = Helper.get_overlap_indices(B.top, self.n, chain=adding_chain, terminus=terminus_b)
        check = Helper.check_overlaps(overlap_A, overlap_B)

        if check:
            return check
        
        superimposer = Superimposer(A, B, overlap_A, overlap_B)
        new_B = superimposer.fit_B_on_A()

        # Instead of atom slice I should use pop/delete to remove the atoms?
        new_A = Helper.remove_overlap(A, overlap_A)
        return new_A, new_B

    # Manipulate topologies: split, merge, and stack components
    def manipulate_topology(self, A, B, leading_chain, adding_chain, reverse, keep_resSeq, dimer_b):
        A_active, A_passive = Helper.split_chain_topology(A, leading_chain)
        if dimer_b:
            B_active, B_passive = Helper.split_chain_topology(B, adding_chain)
            temp = Helper.merge_chain_topology(B_active if reverse else A_active, A_active if reverse else B_active, keep_resSeq)
            C_temp = temp.stack(A_passive, keep_resSeq)
            C = C_temp.stack(B_passive, keep_resSeq)
        else:
            temp = Helper.merge_chain_topology(B if reverse else A_active, A_active if reverse else B, keep_resSeq)
            C = temp.stack(A_passive, keep_resSeq)
        return C

    # Refactored method to add a pair of sites
    def add_pair(self, pair, leading_chain=0, adding_chain=0, verbose=False, reverse=False, segment='fixed'):
        if verbose:
            #print(f'Adding pair: {pair} of dimer {self.n_dimers}')
            pass
        site_a, site_b = pair

        # Get segments based on 'fixed' or 'random' segment criteria
        A, B = self.get_segments(site_a, site_b, segment)

        # Check for dimerization (aka if site is s1 or s2)
        dimer_a, dimer_b = self.check_dimerization(site_a, site_b)

        # Determine growth direction based on terminus
        reverse, terminus_a, terminus_b = self.determine_terminus_direction(site_a, site_b)

        # Manage overlaps and perform superimposition
        new_A, new_B = self.manage_overlaps(A, B, leading_chain, adding_chain, terminus_a, terminus_b)

        # Manipulate topologies: split, merge, and stack components, and return new trajectory
        C = self.manipulate_topology(A=new_A, B=new_B, leading_chain=leading_chain, adding_chain=adding_chain, reverse=reverse, keep_resSeq=True, dimer_b=dimer_b)
        self.traj_history.append(C)
        return C

    def clean_traj(self):
        self.raw_traj = self.traj
        fixer = Fixer(self.traj)
        self.traj = fixer.get_updated_traj()
        # Remove leftover s2 segment domains at the ends of the filament
        self.traj = self.traj.atom_slice(self.traj.top.select(f'chainid 1 to {(self.n_dimers*2)}'))
        self.cleaned = True

    def add_dna(self, chainid=0, frame_idx=1):
        
        if not self.cleaned:
            self.clean_traj()
        
        # Select frame of DNA - DBD complex
        dna_complex = self.site_map['complex'][frame_idx]

        # Get selection of dbd residues for fit of only backbone
        indices_dbd_complex = dna_complex.top.select(f'resSeq 95 to 137 and backbone') # SALMONELA
        indices_dbd_traj = self.traj.top.select(f'(chainid {chainid} and resSeq 95 to 137) and backbone') # Ecoli

        # Fit the dbd with DNA to the loction of the dbd in the filament at chainid
        imposer = Superimposer(self.traj,dna_complex,indices_dbd_traj,indices_dbd_complex)
        new_dbd_complex =  imposer.fit_B_on_A()

        # Remove indices of DBD from complex
        dna = new_dbd_complex.atom_slice(new_dbd_complex.top.select('not protein'))
        # Add DNA to the filament traj
        self.traj = self.traj.stack(dna)
        self.n_dna += 1
    