import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

class ContactCount:

    """

    Example Usage:

    protein_donors = {'GLN112':['N','NE2'],
                  'GLY113':['N'],
                  'ARG114':['N','NE','NH1','NH2']}

    minor_acceptors = {'DA':['N3'],
                    'DT':['O2'],
                    'DC':['O2'],
                    'DG':['N3']}
    
    major_acceptors = {'DA':['N7'],
                   'DT':['"O4'"'],
                   'DC':['None'],
                   'DG':['"O4'"','N7']}
    
    # Here you can adjust the parameters nn and mm to change the shape of the smoothing function
    contacts = ContactCount(traj,protein_donors,minor_acceptors,d0=0.25,r0=0.4,nn=2,mm=4)
    """
    
    def __init__(self, traj, protein_queue, dna_haystack, d0=0.25,r0=0.4,nn=2,mm=4):
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
        # Find protein indices corresponding to queue (Restype-Atomtype) 
        return [self.atom_names.index(res+'-'+i) for res,at in self.protein_queue.items() for i in at]
    
    def get_groove_indices(self):
        # Find selection of atom types for each nucleobase
        return sorted(sum([list(self.top.select(f"resname {res} and {' '.join(['name '+i for i in at])}")) for res,at in self.dna_haystack.items()],[]))
    
    def smooth_contact(self, r):
        # Compute contact based on distance smoothing function
        return ((1 - ((r-self.d0)/self.r0)**self.nn ) / (1 - ( (r-self.d0)/self.r0)**self.mm) )

    def compute_contacts(self):
        # Check where first condition holds
        ones = np.where(self.distances-self.d0 <= 0)
        # Apply second condition
        contacts = np.where(self.distances-self.d0 >= 0, self.smooth_contact(self.distances),self.distances)
        # Apply second condition (...)
        contacts[ones] = np.ones(ones[0].shape)
        return contacts

    def get_total_contacts(self):
        return np.sum(self.contacts,axis=1)
    
    def get_protein_names(self):
        return [self.atom_names[idx] for idx in self.protein_indices]
    
    def get_dna_names(self):
        return [self.atom_names[idx] for idx in sorted(self.groove_indices)]
    
    def get_distance_matrix(self):
        # Reshape pair distances to n x n matrix
        s = self.distances.shape
        return self.distances.reshape(s[0],len(self.protein_indices),len(self.groove_indices))
    
    def get_contact_matrix(self):
        # Reshape pair distances to n x n matrix
        s = self.contacts.shape
        return self.contacts.reshape(s[0],len(self.protein_indices),len(self.groove_indices))
    
    def collect_sections(self):
        section_ends = []
        count = 0
        for residue in self.protein_queue.keys():
            count += len(self.protein_queue[residue])
            section_ends.append(count)
        self.sections = section_ends[:-1]

    def split_data(self):
        return np.split(self.contact_matrix,self.sections,axis=1)
        
    def get_contacts_per_residue(self):
        return np.array([np.sum(d,axis=(1,2)) for d in self.split_data()])
    
    def get_contacts_per_residue_per_base(self):
        return np.array([np.sum(d,axis=1) for d in self.split_data()])
            
    def get_contacts_per_base(self):    
        contacts_per_residue_per_base = self.get_contacts_per_residue_per_base()
        return np.sum(contacts_per_residue_per_base,axis=0).T     
                         
    def get_contacts_per_bp(self):
        contacts_per_base = self.get_contacts_per_base()
        n_bases = len(contacts_per_base)
        return np.array([a+b for a,b in zip(contacts_per_base[:n_bases//2],contacts_per_base[n_bases//2:][::-1])])
    
    def get_contacts_per_residue_per_bp(self):
        contacts_per_residue_per_base = self.get_contacts_per_residue_per_base()
        n_bases = len(contacts_per_residue_per_base.T)
        return np.array([a+b for a,b in zip(contacts_per_residue_per_base.T[:n_bases//2], contacts_per_residue_per_base.T[n_bases//2][::-1])]).T

    def check_axis(self,ax):
        if ax is None:
            fig,ax = plt.subplots(figsize=(8,8))
        return fig,ax
    
    def plot_contact_map(self,ax=None,frame=-1):
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
        
    def plot_contact_distribution(self,ax=None,c='Red'):
        fig,ax = self.check_axis(ax)
        total_contacts = self.get_total_contacts()
        df = pd.DataFrame(total_contacts)

        data = pd.DataFrame({
                "idx": np.tile(df.columns, len(df.index)),
                "$C_{Protein-DNA}$": df.values.ravel()})

        sns.kdeplot(
           data=data, y="$C_{Protein-DNA}$", legend = False, color=c,#hue="idx",
           fill=True, common_norm=False, palette="Reds",
           alpha=.5, linewidth=1, ax=ax)
        
    def ns_to_steps(self,ns=1):
        # assume a timestep of 2 fs
        return int((ns*1000)/0.002)

    # def make_plumed_smd(self, filename=)



#     def make_plumed_metad(self, filename='plumed.dat', write=False):

#         output = f"""UNITS LENGTH=nm TIME=ps ENERGY=kj/mol\n"""
#         output += f"""MOLINFO MOLTYPE=protein STRUCTURE=system.pdb
# WHOLEMOLECULES ENTITY0={self.top._atoms[0].index+1}-{self.top._atoms[-1].index+1}\n
# cmap: CONTACTMAP ...\n"""
            
#         for idx,p in enumerate(self.pairs):
#             output += f'\tATOMS{idx+1}={p[0]+1},{p[1]+1}\n'
                
#         output+=f'\tSWITCH={{RATIONAL R_0={self.r0} D_0={self.d0} NN={self.nn} MM={self.mm}}}\n'
#         output+='\n\t\tSUM \n\t...\n\n'

#         output += f"""metad: METAD ARG=cmap PACE=500 HEIGHT=1.2 SIGMA=1 FILE=HILLS
# PRINT ARG=* FILE=COLVAR STRIDE=50"""

#         if write:
#             with open(filename, "w") as f:
#                 f.write(output)
      
#         print(output)

#     def make_plumed_smd(self, filename='plumed.dat', write=False, start=0, end=1000, step=10, axis='z', ref=0.0, k=1000.0, rate=0.0001):
#         # assume a timestep of 2 fs
#         start = self.ns_to_steps(start)
#         end = self.ns_to_steps(end)
#         step = self.ns_to_steps(step)
        
#         output = f"""UNITS LENGTH=nm TIME=ps ENERGY=kj/mol\n"""
#         output += f"""MOLINFO MOLTYPE=protein STRUCTURE=system.pdb
# WHOLEMOLECULES ENTITY0={self.top._atoms[0].index+1}-{self.top._atoms[-1].index+1}\n"""


#         output += f"""\ncmap: CONTACTMAP ...\n""")

#         for idx, p in enumerate(self.pairs):
#             output += f'\tATOMS{idx+1}={p[0]+1},{p[1]+1}\n'

#         output += f'\tSWITCH={{RATIONAL R_0={self.r0} D_0={self.d0} NN={self.nn} MM={self.mm}}}\n'
#         output += '\n\t\tSUM \n\t...\n\n'

#         output += f"""\nMOVINGRESTRAINT ...
#     ARG=cmap
#     STEP0=0         AT0=0\tKAPPA0=0
#     STEP1={preste}    AT1=0\tKAPPA1={k}
#     STEP2={end}  AT2=0\tKAPPA2={k}
    
#     print('\ncmap: CONTACTMAP ...',file=f)
    
#     for idx,p in enumerate(pairs):
#         print(f'\tATOMS{idx+1}={p}',file=f)
        
#     print(f'\tSWITCH={{RATIONAL R_0={r0} D_0={d0} NN={nn} MM={mm}}}',file=f)
#     print('\n\tSUM',file=f)
#     print('    ...',file=f)

#     print(f"""\nMOVINGRESTRAINT ...
#     \tARG=cmap
#     \tSTEP0=0         AT0={contact_i}\tKAPPA0=0
#     \tSTEP1={presteps}    AT1={contact_i}\tKAPPA1={kappa}
#     \tSTEP2={steps+presteps}  AT2={contact_f}\tKAPPA2={kappa}    
#     ...

#     PRINT ARG=* FILE=COLVAR STRIDE=50""",file=f)
    
# def ns_to_steps(ns=1):
#     return int((ns*1000)/0.002)

# def make_plumed_cmap(traj,contact_i,contact_f,d0,r0,nn,mm,set_A,set_B,kappa=500,ns_production=50,ns_eq=1,save=True):
    
#     top = traj.topology
#     pairs = [f'{a+1},{b+1}' for a in set_A for b in set_B] #plumed works with 1 indexing
#     presteps = ns_to_steps(ns_eq)
#     steps = ns_to_steps(ns_production)
    
#     name = 'system'
#     atom_indices = [at.index+1 for at in top.atoms]
    
#     if save:
#         with open('plumed.dat', 'w') as f:
#             print_plumed_input(name, atom_indices,pairs,d0,r0,nn,mm,contact_i,contact_f,presteps,steps,f)   
#     else:
#         print_plumed_input(name, atom_indices,pairs,d0,r0,nn,mm,contact_i,contact_f,presteps,steps,f=None)