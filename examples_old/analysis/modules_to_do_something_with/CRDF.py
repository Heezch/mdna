from BaseParameters import *   
from scipy.stats import gaussian_kde
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Canion:

    def __init__(self,traj,selection='name NA',mask=None):
        # Initialize the dictionary that will store the computed densities
        self.mask = None
        self.densities = {}
        self.traj = traj
        self.top = traj.topology
        self.mean_reference_frames = self.get_mean_reference_frames()
        self.particles = self.get_ref_particles(selection)
        self.local_coordinates = self.compute_local_coordinates(spherical=False, cylindrical=True)
        self.cmap = sns.color_palette("Spectral_r", as_cmap=True)

    def get_mean_reference_frames(self):
        self.F = Frames(self.traj)
        return self.F.mean_reference_frames
    
    def get_ref_particles(self,selection):
        if type(selection) == str:
            selection = self.top.select(selection)
        elif type(selection) == list:
            selection = np.array(selection)
        else:
            raise TypeError('Selection must be a string of MDtraj atom selection language or a list of atom indices')
        return self.traj.atom_slice(selection)
    
    def compute_local_coordinates(self, spherical=False, cylindrical=True):

        # Get the reference frames for each base pair
        frames = self.mean_reference_frames  # (frames, time, origin and frame vectors, xyz)

        # Get the particle coordinates
        particles = self.particles.xyz.swapaxes(0, 1)  # (n_particles, time, xyz)

        # Compute the local spherical coordinates for each frame
        local_spherical_coordinates = []  # list to hold the spherical coordinates of each frame
        local_cylindrical_coordinates = []  # list to hold the cylindrical coordinates of each frame

        # Loop over each frame
        for idx in range(len(frames)):
            frame = frames[idx]  # (time, origin and frame vectors, xyz)

            # Get the origin and frame vectors from the frame
            origin = frame[:, 0, :]  # (time, xyz)
            frame_vectors = frame[:, 1:, :]  # (time, frame vectors, xyz)

            # Compute the relative positions of the particles to the origin
            relative_pos = particles - origin[np.newaxis, :, :]  # (n_particles, time, xyz)

            # Rotate the relative positions based on the frame vectors
            rotated_pos = np.einsum('tij,ktj->kti', frame_vectors, relative_pos)

            # Compute the radial distance, r
            r = np.sqrt(np.sum(rotated_pos ** 2, axis=-1))  # (n_particles, time)

            if spherical:
                # Compute the polar angle, theta
                theta = np.arccos(rotated_pos[..., 2] / r)  # (n_particles, time)

            # Compute the azimuthal angle, phi
            phi = np.arctan2(rotated_pos[..., 1], rotated_pos[..., 0])  # (n_particles, time)

            if spherical:
                # Add the computed spherical coordinates to the list
                local_spherical_coordinates.append((r, theta, phi))

            if cylindrical:
                # Compute the cylindrical coordinates
                z = rotated_pos[..., 2]  # (n_particles, time)
                cylindrical_r = np.sqrt(rotated_pos[..., 0] ** 2 + rotated_pos[..., 1] ** 2)  # (n_particles, time)
                # Add the computed coordinates to the list
                local_cylindrical_coordinates.append((cylindrical_r, z, phi))

        # Convert the list to a numpy array for easier manipulation
        if spherical:
            local_spherical_coordinates = np.array(local_spherical_coordinates)  # shape -> (frames, 3, n_particles, time)
        if cylindrical:
            local_cylindrical_coordinates = np.array(local_cylindrical_coordinates)  # shape -> (frames, 3, n_particles, time)
        
        #  Return the local coordinates of the particles for each base pair
        if cylindrical:
            return local_cylindrical_coordinates 
        elif spherical:
            return local_spherical_coordinates
        elif cylindrical and spherical:
            return local_cylindrical_coordinates, local_spherical_coordinates
    
    def create_radial_grid(self,radius, num):
        # Create a grid of radial and angular (theta) coordinates
        if self.mask == 'inner':
            r = np.linspace(1.025, radius, num)
        else:
            r = np.linspace(0, radius, num)
        theta = np.linspace(0, 2*np.pi, num)
        r_grid, theta_grid = np.meshgrid(r, theta)
        xi = r_grid * np.cos(theta_grid)
        yi = r_grid * np.sin(theta_grid)
        return xi, yi
    
    def create_density_estimate(self,local_coordinates, radius=np.pi, num=60, cut=0.165, k=1,bw='scott'):
        
        # Unpack the local coordinates
        r = local_coordinates[0].flatten()
        z = local_coordinates[1].flatten()
        phi = local_coordinates[2].flatten()

        # Slice the coordinates based on z value to obtain cylindrical slice around base pair
        if cut:
            mask = np.where(np.abs(z) < cut)
            r = r[mask] 
            z = z[mask] 
            phi = phi[mask] 

        # Convert cylindrical coordinates to cartesian
        x = r[::k] * np.cos(phi)[::k]
        y = r[::k] * np.sin(phi)[::k]

        # Estimate the density of the points
        xy = np.vstack([x, y])
        try:
            d = gaussian_kde(xy,bw_method=bw)(xy)

        except ValueError as e:
            print("Error:", e)
            print("Value of xy:", xy)
            print("Value of r:", r)
            return np.zeros((num,num)), np.zeros((num,num)), np.zeros((num,num)), 0

        # Remove points with low density
        mask = d > 0.01
        x = x[mask]
        y = y[mask]
        z = z[mask]

        # Create a spherical grid based on radius for the density estimate
        if radius is None:
            radius = np.sqrt(x.max()**2 + y.max()**2)
        xi, yi = self.create_radial_grid(radius, num)
        xy_grid = np.vstack([xi.flatten(), yi.flatten()])
        zi = gaussian_kde(xy,bw_method=bw)(xy_grid)
        zi = zi.reshape(xi.shape)
        # Return the cartesian coordinates and the density estimate as well as the maximum density
        return xi, yi, zi, zi.max()
    
    def compute_density(self, dna=True, bp_index=1, radius=np.pi, num=100, cut=0.165, k=1,bw='scott'):

        # Get the local coordinates depending on whether the user wants to plot the dna or a single base pair
        if dna:
            coords = self.local_coordinates.swapaxes(0, 1)
            key = 'dna'
        else:
            coords = self.local_coordinates[bp_index]
            key = bp_index  # Use base pair index as key

        # Create a radial grid and compute the density estimate
        x, y, d, dmax = self.create_density_estimate(coords, radius=radius, num=num, cut=cut, k=k,bw=bw)

        # Store the computed values in the dictionary
        self.densities[key] = {
            'xi': x,
            'yi': y,
            'dens': d,
            'dens_max': dmax
        }
    
    def compute_all_densities(self, num=100, radius=np.pi, cut=0.165, k=1):
        # Loop over all base pair indices and compute density for each
        for bp_index in range(len(self.local_coordinates)):
            self.compute_density(dna=False, bp_index=bp_index, num=num, radius=radius, cut=cut, k=k)
    
    def get_global_vmin_vmax(self):
        # Initialize vmin and vmax to extreme values
        vmin, vmax = np.inf, -np.inf
        # Loop over all densities
        for key in self.densities:
            # Skip the 'dna' entry
            if key == 'dna':
                continue
            dens_max = self.densities[key]['dens_max']
            dens_min = self.densities[key]['dens'].min()
            # Update vmin and vmax
            if dens_max > vmax:
                vmax = dens_max
            if dens_min < vmin:
                vmin = dens_min
        return vmin, vmax

    def decorate_axis(self,ax,lw=1,alpha=0.5,color='white',radius=np.pi,labels=True):
        radii = [1.025, 2.825]
        for rr in radii:
            circle = plt.Circle((0,0),rr,fill=False, color=color,linewidth=1,ls='--',alpha=alpha)
            ax.add_artist(circle)
            #ax.text(x=rr*np.sin(np.pi/4), y=-rr*np.sin(-np.pi/4), s=str(rr), rotation=np.pi/4,color='gray', fontsize=10)
            #ax.text(x=-0.1,y=rr+0.1,s=str(rr),color=color, fontsize=10)
            if labels:
                ax.text(x=0, y=rr+0.15 , ha='center', va='center', s=str(rr*10)+' $\AA$', color=color, fontsize=10)
        if labels:
            ax.text(x=0, y=-radius - 0.15, ha='center', va='center', s='Minor', rotation=0, color='gray', fontsize=12)
            ax.text(x=0, y=radius + 0.15, ha='center', va='center', s='Major', rotation=0, color='gray', fontsize=12)
        
    def add_colorbar(self,ax,dens):
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)

            fig = ax.figure  # get the figure of the Axes
            fig.colorbar(dens, cax=cax,label='Density')
            # Maintain aspect ratio
            ax.set_aspect('equal')  # This will make the x and y scales equal, so your plot remains undistorted
            
    def plot_density(self, dna=True, bp_index=1, ax=None, levels=20, cmap=None, vmin=0, num=100, radius=np.pi, cut=0.165, k=1,vmax=None, antialiased=True, decoration=True, cbar=False,bw=0.1,labels=True,imreturn=False):
       
        # Check if the appropriate density has been computed
        key = 'dna' if dna else bp_index
        if key not in self.densities:
            # If not, compute it
            self.compute_density(dna=dna, bp_index=bp_index,num=num,radius=radius,cut=cut,k=k,bw=bw)

        # Now, get the density data
        density_data = self.densities[key]

        # Set the colorbar limits
        if vmax is None:
            vmax = density_data['dens_max']

        if ax is None:
            _, ax = plt.subplots(figsize=(4,4))

        # Plot the density estimate
        if cmap is None:
            cmap = self.cmap
            
        dens = ax.contourf(density_data['yi'],density_data['xi'], density_data['dens'], levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, antialiased=antialiased)
        ax.axis('off') # Remove the axis
        
        # Decorate the axis with radial and angular grid lines
        if decoration:
            self.decorate_axis(ax,radius=radius,labels=labels)

        # Add a colorbar
        if cbar:
            self.add_colorbar(ax,dens)

        if imreturn:
            return dens

    def plot_all_densities(self, levels=20, cmap=None, num=100, radius=np.pi, cut=0.165, k=1, antialiased=True, decoration=True, cbar=False,size=1,labels=True):
        # Compute all densities
        self.compute_all_densities(num=num, radius=radius, cut=cut, k=k)
        # Get global vmin and vmax
        vmin, vmax = self.get_global_vmin_vmax()

        # Get the number of base pairs
        n_bp = len(self.local_coordinates)

        # Determine the layout of the subplots
        ncols = int(np.ceil(np.sqrt(n_bp)))
        nrows = int(np.ceil(n_bp / ncols))

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=[size * ncols, size * nrows])

        # Plot density for each base pair index
        for i, ax in enumerate(axes.flat):
            if i < n_bp:
                self.plot_density(dna=False,bp_index=i, ax=ax, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, num=num, radius=radius, cut=cut, k=k, antialiased=antialiased, decoration=decoration, cbar=cbar,labels=labels)
            else:
                ax.axis('off')  # Hide unused subplots

        # Adjust the spacing between subplots
        fig.tight_layout()

        return fig, axes