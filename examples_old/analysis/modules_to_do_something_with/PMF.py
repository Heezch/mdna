import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from ContactCount import *


class DataProcessor:
    """

    Requires ContactCount.py

    Example usage:

    # Create an instance of the class with your data location and number of files
    data = DataProcessor('../data/smd/0_highaff/1_lcmap/2_k500_108_to_65_100ns/', 20)

    # Plot work profiles
    fig, ax = plt.subplots()
    data.plot_profiles(ax=ax,palette="Blues_r")

    # Plot PMF
    fig, ax = plt.subplots()
    data.plot_pmf(average=True, std=True,profiles=False,cumulant=True,ax=ax[0],color='cornflowerblue',palette='Blues_r')
    ax[0].set_xlim(0,1)
    ax[0].invert_xaxis()
    print('$\Delta$W = ',np.max(data.bavg)-np.min(data.bavg))

    # Plot 2D PMF
    fig, ax = plt.subplots()
data.plot_2d_pmf(ax=ax[0],palette='Blues_r',vmax=60,paths=True)

    """
    def __init__(self, location, n_files=None,unit='kcal',ns=100,blacklist=None,id=None):
        self.unit = unit
        self.location = location
        self.ns = ns
        self.n_files = n_files
        self.temperature = 298
        self.blacklist = blacklist
        self.id = id
        self.load_trajectories()
        self.load_colvars()
        self.compute_contacts() 
        
    
    def check_pulling_coordinate(self):
        # check if pulling coordinate follows steering plan 
        pass

    def compute_contacts(self):


        minor_acceptors = {'DA':['N1'],
                    'DT':['N3'],
                    'DC':['N3'],
                    'DG':['N1']}

        # Top 5 residues
        protein_stack = {'GLU228': ['N', 'OE1', 'OE2'],
                        'ARG232': ['N', 'NE', 'NH1', 'NH2'],
                        'ALA233': ['N'],
                        'ARG235': ['N', 'NE', 'NH1', 'NH2'],
                        'ASN236': ['N', 'OD1', 'ND2']}

        # Top 8 residues
        protein_stack = {'LYS219': ['N', 'NZ'],
                        'GLU228': ['N', 'OE1', 'OE2'],
                        'ARG232': ['N', 'NE', 'NH1', 'NH2'],
                        'ALA233': ['N'],
                        'ARG235': ['N', 'NE', 'NH1', 'NH2'],
                        'ASN236': ['N', 'OD1', 'ND2'],
                        'LYS239': ['N', 'NZ'],
                        'LYS249': ['N', 'NZ']}

        # Top 11 residues
        protein_donors = {'LYS219': ['N', 'NZ'],
                        'MET225': ['N'],
                        'GLU228': ['N', 'OE1', 'OE2'],
                        'LYS229': ['N', 'NZ'],
                        'ALA231': ['N'],
                        'ARG232': ['N', 'NE', 'NH1', 'NH2'],
                        'ALA233': ['N'],
                        'ARG235': ['N', 'NE', 'NH1', 'NH2'],
                        'ASN236': ['N', 'OD1', 'ND2'],
                        'LYS239': ['N', 'NZ'],
                        'LYS249': ['N', 'NZ']}

        self.contacts = {}
        # Here you can adjust the parameters nn and mm to change the shape of the smoothing function
        self.C = [ContactCount(traj,protein_donors,minor_acceptors,d0=0.25,r0=0.4,nn=2,mm=4) for traj in self.trajs]
        contacts_per_residue = np.array([c.get_contacts_per_residue() for c in self.C]).swapaxes(0,1)
        for _,key in enumerate(protein_donors.keys()):
            self.contacts[key] = contacts_per_residue[_]
        self.contacts['C'] = np.array([c.get_total_contacts() for c in self.C])
        self.protein_donors = protein_donors

    def load_trajectories(self):
        self.trajs = [md.load(self.location + f'dry_{i}.xtc',top=self.location+f'dry_{i}.pdb') for i in range(self.n_files)]
        self.n_frames = self.trajs[0].n_frames


    # def load_trajectories(self):
    #     self.index_map = {}
    #     self.trajs = []
    #     idx = 0
    #     for i in range(self.n_files):
    #         key = self.id+'_'+str(i)
    #         if key not in self.blacklist:
    #             traj = md.load(self.location + f'dry_{i}.xtc', top=self.location + f'dry_{i}.pdb')
    #             self.trajs.append(traj)
    #             self.index_map[idx] = i
    #             idx += 1

    #     if self.trajs:
    #         self.n_frames = self.trajs[0].n_frames

    # def load_trajectories(self):
    #     self.index_map = {}
    #     self.trajs = []
    #     idx = 0
    #     for i in range(self.n_files):
    #         try:
    #             traj = md.load(self.location + f'dry_{i}.xtc', top=self.location + f'dry_{i}.pdb')
    #             #if traj.n_frames != 2011:
    #             if traj.n_frames < 1600:                    
    #                 continue
    #             else:
    #                 #self.trajs.append(traj)
    #                 self.trajs.append(traj[:1600])
    #                 self.index_map[idx] = i
    #         except OSError:
    #             print(f"File dry_{i}.xtc or dry_{i}.pdb not found in {self.location}. Skipping...")
    #             continue
    #         idx += 1
    #     if self.trajs:
    #         self.n_frames = self.trajs[0].n_frames
    #     else:
    #         print("No valid trajectories loaded.")


    # def load_colvars(self):
    #     self.headers = ['time','cmap','bias','force2','cmap_cntr','cmap_work','cmap_kappa','work'] # headers for the colvars file
    #     self.colvars = np.array([np.loadtxt(self.location + f'COLVAR_{i}', comments='#', usecols=(2,6,7)) for i in range(self.n_files)]).swapaxes(0,1).T
    #     self.colvars = self.colvars[:,:,:89000]
    #     print(self.colvars.shape)
    #     ddK = np.diff(np.diff(self.colvars[1][0])) # second derivative of the force constant 
    #     self.start = np.argmin(ddK) # find the point where the force constant is at maximum (aka start of the pulling)
    #     self.correct_work_profiles() # correct the work profiles to start at 0 work at the start of the pulling
    #     self.get_order() # order the works for plotting
    #     self.compute_pmf()
    #     self.match_frames() 

    # def load_colvars(self):
    #     self.headers = ['time','cmap','bias','force2','cmap_cntr','cmap_work','cmap_kappa','work'] # headers for the colvars file
    #     colvars_list = []
    #     for i in range(self.n_files):
    #         key = self.id+'_'+str(i)
    #         if key not in self.blacklist:
    #             colvar_data = np.loadtxt(self.location + f'COLVAR_{i}', comments='#', usecols=(2,6,7))
    #             colvars_list.append(colvar_data)

    #     if colvars_list:
    #         self.colvars = np.array(colvars_list).swapaxes(0,1).T

    def load_colvars(self):
        self.headers = ['time','cmap','bias','force2','cmap_cntr','cmap_work','cmap_kappa','work'] # headers for the colvars file
        colvars_list = []

        for i in range(self.n_files):
            try:
                colvar_data = np.loadtxt(self.location + f'COLVAR_{i}', comments='#', usecols=(1,6,7))
                #print(colvar_data.shape)
                # print(len(colvar_data))
                #if len(colvar_data) != 100500:
                # if len(colvar_data) < 80000:
                #     print(i,len(colvar_data))
                #     continue
                # else:
                colvars_list.append(colvar_data)
                
            except FileNotFoundError:
                print(f"File COLVAR_{i} not found in {self.location}. Skipping...")
                continue

        if colvars_list:
            
            # print(self.colvars.shape)
            #self.colvars = self.colvars[:,:,:80000]
            self.colvars = np.array(colvars_list).swapaxes(0,1).T
            
        else:
            print("No valid COLVAR files loaded.")
        
        ddK = np.diff(np.diff(self.colvars[1][0])) # second derivative of the force constant 
        self.start = np.argmin(ddK) # find the point where the force constant is at maximum (aka start of the pulling)
        self.correct_work_profiles() # correct the work profiles to start at 0 work at the start of the pulling
        self.get_order() # order the works for plotting
        self.compute_pmf()
        self.match_frames() 


    def match_frames(self):
        # match the frames of the colvars to the trajectories
        self.time = np.linspace(0, self.ns, num=self.n_frames)
        self.progress = np.linspace(1, 0, num=self.n_frames)

        # Define a helper function for resizing
        def resize(old_array, new_length):
            old_length = len(old_array)
            old_indices = np.arange(old_length)
            new_indices = np.linspace(0, old_length - 1, new_length)
            return np.interp(new_indices, old_indices, old_array)

        # Resize the arrays to match the length of self.time/self.n_frames
        self.bavg = resize(self.bavg, self.n_frames)
        self.average = resize(self.average, self.n_frames)
        self.cumulant = resize(self.cumulant, self.n_frames)
        self.std = resize(self.std, self.n_frames)
        self.works = np.array([resize(work, self.n_frames) for work in self.works])


    def correct_work_profiles(self):
        # correct the work profiles to start at 0 work at the start of the pulling
        for i in range(len(self.colvars[2])):
            self.colvars[2][i] -= self.colvars[2][i][self.start]
            self.colvars[2][i][:self.start] = 0
        self.works_kJ = self.colvars[2]
        self.convert_units()

    def convert_units(self, to_unit='kcal'):
        conversion_factor = {'kJ': 1, 'kcal': 0.239005736}
        self.works_kcal = self.works_kJ * conversion_factor[to_unit]

        if self.unit == 'kcal':
            self.works = self.works_kcal    
        elif self.unit == 'kJ':
            self.works = self.works_kJ

    def get_order(self):
        # Order the works for plotting
        wmin = np.min(self.works, axis=1)
        wmax = np.max(self.works, axis=1)
        self.order = np.argsort(wmax - wmin)

    def compute_pmf(self):  
        # Set units
        kB = {'kcal': 1.987204259e-3, 'kJ': 8.314462618e-3}
        self.beta = 1.0 / (kB[self.unit] * self.temperature) 

        # Jarzynski's equality to find exponential average of work profiles to determine PMF
        self.bavg = -1 / self.beta * np.log(np.mean(np.exp(-self.beta * self.works), axis=0))

        # Cumulant expansion to measure variance of work profiles 
        self.average = np.mean(self.works, axis=0)
        self.cumulant = self.average - (self.beta/2 * np.var(self.works, axis=0))

        # Standard deviation of work profiles beta-weighted
        self.std = np.std(self.works, axis=0) * self.beta/2 # aka second cumulant sqrt of variance

    def set_xaxis(self, xaxis='progress'):

        if xaxis == 'progress':
            x = self.progress
            xlabel = 'Progress (a.u.)'
        elif xaxis == 'time':
            x = self.time
            xlabel = 'Time (ns)'
        
        self.x, self.xlabel = x, xlabel

    def plot_profiles(self, ax=None, palette='viridis',xaxis='progress'):
        if ax is None:
            _, ax = plt.subplots()
        cmap = sns.color_palette(palette, n_colors=self.n_files)

        self.set_xaxis(xaxis)

        for i in range(len(self.order)):
            ax.plot(self.x,self.works[self.order[i]], c=cmap[i])
        ax.set_ylabel(f'Work ({self.unit})')
        ax.set_xlabel(self.xlabel)

    def plot_pmf(self, ax=None, color=None, palette='viridis', cumulant=False, average=False, std=True,profiles=False,legend=False,labels=False,xaxis='progress'):
        if ax is None:
            _, ax = plt.subplots()
        if color is None:
            color = sns.color_palette(palette, n_colors=1)[0]

        self.set_xaxis(xaxis)

        # Plot the exponential average
        ax.plot(self.x,self.bavg, color=color,label='Exponential Average')

        if std:
            # Plot the standard deviation as a fill between around the exponential average
            ax.fill_between(self.x,(self.bavg - self.std), (self.bavg + self.std), color=color, alpha=.1)

        if average:
            # Plot the average
            ax.plot(self.x,self.average, label='Average',ls=':',color=color)
        if cumulant:
            # Plot the cumulant expansion
            ax.plot(self.x,self.cumulant, label='Cumulant Expansion',ls='--',color=color)

        if profiles:
            # Plot the work profiles
            self.plot_profiles(ax=ax, palette=palette,xaxis=xaxis)

        # Set labels and title
        if labels:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(f'PMF ({self.unit}/mol)')
        if legend:
            ax.legend()

    @staticmethod
    def denormalize_y(norm, data):
        return norm * (min(data) - max(data)) + max(data)

    @staticmethod
    def denormalize_x(norm, data):
        return norm * (max(data) - min(data)) + min(data)

    @staticmethod
    def get_basis(x, y, max_order=8):
        basis = []
        for i in range(max_order+1):
            for j in range(max_order - i +1):
                basis.append(x**j * y**i)
        return basis

    def smooth_and_fit_2d_grid(self, Z, order=20, nan_ratio=1.1,vmax=None): # nan_ratio default value is 110% of maximum value
        
        # Calculate the max value and the replacement for NaNs
        max_value = np.nanmax(Z) 
        
        if vmax is not None:
            nan_replacement = vmax
        else:
            nan_replacement = max_value * nan_ratio
        Z = np.nan_to_num(Z, nan=nan_replacement)

        print('Max value =', max_value, 'NaN replacement =', nan_replacement)
        row = np.linspace(0,1,Z.shape[0])
        X,Y = np.meshgrid(row,row)

        # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
        x, y = X.ravel(), Y.ravel()
        # Maximum order of polynomial term in the basis.
        max_order = order
        basis = self.get_basis(x, y, max_order)
        # Linear, least-squares fit.
        A = np.vstack(basis).T
        b = Z.ravel()
        # Replace NaNs with the replacement value
        b = np.nan_to_num(b, nan=nan_replacement)
        c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        print('Fitted parameters:')
        
        # Calculate the fitted surface from the coefficients, c.
        fit = np.sum(c[:, None, None] * np.array(self.get_basis(X, Y, max_order))
                        .reshape(len(basis), *X.shape), axis=0)

        rms = np.sqrt(np.mean((Z - fit)**2))
        print('RMS residual =', rms)
        
        return X, Y, fit.T, max_value

    def boltz(self,works):
        beta = 1/(298*0.0083145)
        epsilon = 1e-10 # offset to avoid taking log of zero or negative
        bavg = -1 / beta * np.log(np.mean(np.exp(-beta * works), axis=0)+epsilon)
        return bavg 

    def generate_2D_projected_PMF(self,x, y, z, N=100):
        xedges = np.linspace(np.min(x),np.max(x),N)
        yedges = np.linspace(np.min(y),np.max(y),N)
        xchunks = [np.argwhere((x>=i) & (x<=j)).flatten() for i, j in zip(xedges, xedges[1:])]
        ychunks = [np.argwhere((y>=i) & (y<=j)).flatten() for i, j in zip(yedges, yedges[1:])]
        chunkgrid = [np.intersect1d(xi,yi) for xi in xchunks for yi in ychunks[::-1]]
        Z = np.array([self.boltz(z[g]) for g in chunkgrid]).reshape(N-1,N-1)
        return Z

    def make_2d_pmf_plot(self,X, Y, Zi, x, y, vmax=55, levels=20, palette='Blues_r',ax=None,cbar=False):
        colors = sns.color_palette(palette, levels)
        colors[-1] = 'white'
        cmap = mpl.colors.ListedColormap(colors, "", len(colors))

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 5))
        _ = ax.contourf(self.denormalize_x(X, x), self.denormalize_y(Y, y), Zi, cmap=cmap, vmin=-1, vmax=vmax, levels=levels)
        ax.invert_xaxis()
        ax.invert_yaxis()
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_clim(0, vmax)
        if cbar:
            plt.colorbar(m, ax=ax, ticks=np.linspace(0, vmax, 6), boundaries=np.linspace(0, vmax, levels)).set_label(label=f'Work [{self.unit}/mol]', size=8)

    def lpfilter(self,input_signal, win):
        # Low-pass linear Filter
        # (2*win)+1 is the size of the window that determines the values that influence 
        # the filtered result, centred over the current measurement
        # http://scotthosking.com/notebooks/smooth_timeseries/
        from scipy import ndimage
        kernel = np.lib.pad(np.linspace(1,3,win), (0,win-1), 'reflect') 
        kernel = np.divide(kernel,np.sum(kernel)) # normalise
        output_signal = ndimage.convolve(input_signal, kernel) 
        return output_signal


    def plot_paths(self,ax,x=None,y=None,n_paths=4,interval=30,window=50,color='gold'):
        # plot paths from lowest to highest work
        for o in self.order[:n_paths]:
            ax.plot(self.lpfilter(x[o],window)[::interval],self.lpfilter(y[o],window)[::interval],c=color,lw=0.5)
            ax.scatter(self.lpfilter(x[o],window)[::interval],self.lpfilter(y[o],window)[::interval],c=color,s=0.8)


    def plot_2d_pmf(self,x=None, y=None, palette='Greys_r',N=100, vmax=None, paths=False,levels=20, order=20,cbar=False, nan_ratio=1.1,window=50,interval=30,n_paths=4,path_color='gold',ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 5))
       
        if x is None:
            x_ = self.contacts['ARG114']
            x = x_.flatten()
        if y is None:
            y_ = self.contacts['GLN112']
            y = y_.flatten()

        z = self.works.flatten()

        Z = self.generate_2D_projected_PMF(x, y, z, N=N)
        X, Y, Zi, max_value = self.smooth_and_fit_2d_grid(Z, order=order,nan_ratio=nan_ratio,vmax=vmax)

        self.make_2d_pmf_plot(X, Y, Zi, x, y, vmax=max_value if vmax is None else vmax, levels=levels,palette=palette,ax=ax,cbar=cbar)
        if paths:
            self.plot_paths(ax=ax,x=x_,y=y_,window=window,interval=interval,n_paths=n_paths,color=path_color)




