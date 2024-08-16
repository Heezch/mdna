
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
from pyntcloud import PyntCloud
from heapq import heappush, heappop
import itertools


class Voxelizer:
    #  Recieves xyz coordinates of a protein and voxelizes it
    #  Determine if voxelizations is done with a voxel dimension of number of grid points
    #  Afterwards atom indices or centers of mass or other points can be mapped to the voxel grid
    #  We need to create function that returns for each point the voxel it belongs to and stores them in a dynamic list
    #  Next for each control point we need to decide if it is allowed to be in the voxel or not
    #  If it is not allowed we need to find the nearest zero voxel and move it there
    #  Finally we need to return the voxel grid with the control points both in the voxel grid and the original coordinates
    #  Additionally we can add some plotting/vizualization functions
    """Voxelizer class for voxelizing a protein based on its XYZ coordinates.

        Args:
            xyz (numpy.ndarray): XYZ coordinates of the protein (n_atoms, 3).
            n (int): Number of voxels in each dimension of the voxel grid.

        Attributes:
            xyz (numpy.ndarray): XYZ coordinates of the protein (n_atoms, 3).
            n (int): Number of voxels in each dimension of the voxel grid.
            voxelgrid (pyntcloud.structures.VoxelGrid): Voxel grid object created using PyntCloud.
            binary_voxel_array (numpy.ndarray): Binary voxel array (n, n, n).

        Methods:
            xyz_to_voxel(plot=False): Converts XYZ coordinates to voxel grid object.
            to_binary(): Converts the voxel grid to a binary voxel array.
            get_voxel_indices(points): Maps points to the voxel grid and returns voxel indices and voxel numbers.
        """

    # 1. Voxelization of the protein
    def __init__(self, xyz, n=10):
        self.xyz = xyz # xyz coordinates of the protein (n_atoms, 3)
        self.n = n # number of voxels in each dimension of the voxel grid, int

        # Creates a voxel grid using PyntCloud
        self.xyz_to_voxel() # converts xyz coordinates to voxel grid object
        self.binary_voxel_array = self.to_binary() # binary voxel array (n,n,n)


    def xyz_to_voxel(self, plot=False):
        """"Converts point cloud as N,3 numpy array of XYZ coordinates
            with n the number of voxels in each dimension of the voxel grid
            https://medium.com/analytics-vidhya/3d-cad-to-binary-voxel-numpy-array-b538d00d97da"""

        # create a dataframe with the atomic coordinates
        df = pd.DataFrame(data=self.xyz, columns=['x','y','z'])

        # create a PyntCloud object with the dataframe
        cloud = PyntCloud(df)

        # plot the point cloud  
        voxelgrid_id = cloud.add_structure("voxelgrid",n_x=self.n, n_y=self.n, n_z=self.n)
        self.voxelgrid = cloud.structures[voxelgrid_id]

        if plot:
            self.voxelgrid.plot(d=3, mode="density", cmap="hsv")

    def to_binary(self):
        # get the voxel grid as a numpy array as a binary array
        return self.voxelgrid.get_feature_vector(mode="binary")

    # 2. Mapping of points to the voxel grid
    def get_voxel_indices(self, points):
        
        # Calculate voxel indices based on the voxel grid segments 
        voxel_x = np.clip(np.searchsorted(self.voxelgrid.segments[0], points[:, 0]) - 1, 0, self.voxelgrid.x_y_z[0])
        voxel_y = np.clip(np.searchsorted(self.voxelgrid.segments[1], points[:, 1]) - 1, 0, self.voxelgrid.x_y_z[1])
        voxel_z = np.clip(np.searchsorted(self.voxelgrid.segments[2], points[:, 2]) - 1, 0, self.voxelgrid.x_y_z[2])

        # Convert voxel indices to voxel numbers, list of voxel numbers
        voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], self.voxelgrid.x_y_z)

        # Get the unique voxel numbers of the points
        voxel_indices = np.array([voxel_x, voxel_y, voxel_z]).T # (n_points, 3)

        # Return the voxel indices and voxel numbers
        return voxel_indices, voxel_n

    # 3. Check if points are allowed to be in the voxel grid
    def find_nearest_zero_voxel(self, voxel_index):
        """WARNING this function might needs improvement to move the point 
          to the nearest zero voxel based on lowest DENSITY or 
          based on a specific vector direction away from the voxel grid"""
        
        # Check if the given voxel is 1
        # print(voxel_array[voxel_index[0], voxel_index[1], voxel_index[2]])

        # Get the coordinates of all voxels with value 0
        zero_voxels = np.argwhere(self.binary_voxel_array == 0)

        # Calculate the distances from the given voxel to all zero voxels
        distances = cdist([voxel_index], zero_voxels)

        # Find the index of the nearest zero voxel
        nearest_zero_voxel_index = zero_voxels[np.argmin(distances)]

        return tuple(nearest_zero_voxel_index)
    
    def process_points(self, points):
        # Checks if the points are allowed to be in the voxel grid
        # And if it is not allowed it finds the nearest zero voxel and moves it there
        # Get the voxel indices and voxel numbers of the points
        voxel_indices, voxel_n = self.get_voxel_indices(points)

        control_points,ids = [], []
        for _, index in enumerate(voxel_indices):
            value = self.binary_voxel_array[index[0], index[1], index[2]]
            if value == 1:
                # point is inside voxel space, move it to nearest zero voxel
                print(f'Point {index} is inside voxel space. Moving to nearest zero voxel.')
                new_index = self.find_nearest_zero_voxel((index))    
                new_id = np.ravel_multi_index([new_index[0], new_index[1], new_index[2]], self.voxelgrid.x_y_z)
                control_points.append(new_index)
                ids.append(new_id)
                print(f'Point {index}, {voxel_n[_]} is nearest zero voxel {new_index}, {new_id}')
            else:
                # point is outside voxel space
                control_points.append(index)  
                ids.append(voxel_n[_])

        # Return the control points and their voxel numbers
        self.control_points = np.array(control_points)
        self.voxel_ids = ids

    def voxel_to_xyz(self, voxel_indices):
        """Converts voxel indices to XYZ coordinates."""
        # Get the voxel centers
        voxel_centers = self.voxelgrid.voxel_centers
        # Get the voxel IDs
        ids = [np.ravel_multi_index([index[0], index[1], index[2]], self.voxelgrid.x_y_z) for index in voxel_indices]
        # Return the XYZ coordinates of the voxel centers
        return voxel_centers[ids]

    # 4. Find path between two points in the voxel grid
    def heuristic(self, a, b):
        """Function to calculate the Euclidean distance (heuristic) between two points"""
        # Calculate the Euclidean distance between points 'a' and 'b'
        return np.sqrt(sum((np.array(a) - np.array(b)) ** 2))


    def astar(self, start, goal):
        """A* algorithm implementation"""
        array = self.binary_voxel_array  # Get the voxel array
        neighbors = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]

        close_set = set()  # Set to store explored nodes
        came_from = {}     # Dictionary to store the path for each node
        gscore = {start: 0}  # Dictionary to store the cost to reach each node from the start
        fscore = {start: self.heuristic(start, goal)}  # Dictionary to store the total estimated cost for each node
        oheap = []  # Priority queue to store nodes to be explored

        heappush(oheap, (fscore[start], start))  # Add the start node to the priority queue
        
        while oheap:

            current = heappop(oheap)[1]  # Get the node with the lowest estimated cost from the priority queue

            if current == goal:
                # Reconstruct the path from the goal node to the start node
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data

            close_set.add(current)  # Mark the current node as explored
            for i, j, k in neighbors:
                neighbor = current[0] + i, current[1] + j, current[2] + k  # Calculate the coordinates of the neighbor node
                
                # Calculate the tentative cost to reach the neighbor node from the current node
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                
                if 0 <= neighbor[0] < array.shape[0]:
                    if 0 <= neighbor[1] < array.shape[1]:
                        if 0 <= neighbor[2] < array.shape[2]:
                            if array[neighbor[0]][neighbor[1]][neighbor[2]] == True:
                                continue
                        else:
                            # Skip neighbors outside of the array bounds in the z direction
                            continue
                    else:
                        # Skip neighbors outside of the array bounds in the y direction
                        continue
                else:
                    # Skip neighbors outside of the array bounds in the x direction
                    continue
                    
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    # Skip this neighbor if it has already been explored or if the tentative cost is higher
                    continue
                    
                if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    # Update the path and cost information for the neighbor node
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))  # Add the neighbor to the priority queue
        print('No path found to the goal node')         
        return False  # Return False if no path is found

    def find_path(self, ordered=True):
        """Finds the path between the control points in the voxel grid
           either in order or using the traveling salesman problem (TSP)
           which could be adjusted to account for that the DNA cannot cross itself but should also avoid itself;
           this could be done by adding a penalty to the cost function of the A* algorithm based on the existing path
           or temporarily add a value of 1 at the excisting path to the voxel array for the duration of the A* algorithm """
        if ordered:
            # Existing logic for ordered path
            path = []
            for i in range(len(self.control_points)-1):
                start = self.control_points[i]
                goal = self.control_points[i+1]
                path.extend(self.astar(tuple(start), tuple(goal)))
            return path
        else:
            # Logic for TSP with unique paths
            shortest_path = []
            shortest_distance = float('inf')

            for permutation in itertools.permutations(self.control_points[1:]):
                current_path = []
                current_distance = 0
                is_valid_path = True
                previous_point = self.control_points[0]

                for point in permutation:
                    segment = self.astar(tuple(previous_point), tuple(point))
                    if self.is_segment_repeated(current_path, segment):
                        is_valid_path = False
                        break
                    current_path.extend(segment)
                    current_distance += self.calculate_segment_distance(segment)
                    previous_point = point

                if is_valid_path and current_distance < shortest_distance:
                    shortest_distance = current_distance
                    shortest_path = current_path

            return shortest_path

    def is_segment_repeated(self, current_path, segment):
        """Check if a segment is repeated in the current path"""
        segment_pairs = set(zip(segment, segment[1:]))
        current_path_pairs = set(zip(current_path, current_path[1:]))
        return not segment_pairs.isdisjoint(current_path_pairs)

    def calculate_segment_distance(self, segment):
        """Calculate the distance of a segment"""
        distance = 0
        for i in range(len(segment) - 1):
            distance += self.heuristic(segment[i], segment[i+1])
        return distance

    def plot(self, ax=None, control_points=None,path=None):
        """Plots the voxel grid."""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        
        # Plot the binary protein voxel array
        ax.voxels(self.binary_voxel_array, edgecolor='navy', shade=True, facecolor='lightblue', alpha=0.95, linewidth=0.2)
        if control_points is not None:
            # Plot the control points
            C = np.array([True if idx in self.voxel_ids else False for idx,_ in enumerate(self.voxelgrid.voxel_centers)]).reshape(self.n,self.n,self.n)
            ax.voxels(C, edgecolor='red', shade=True, facecolor='coral')
        if path is not None:
            ax.scatter3D(path[:,0], path[:,1], path[:,2], 'red', linewidth=2)




    # def find_path(self, ordered=True):
    #     
    #     import itertools
    #     if ordered:
    #         # Existing logic for ordered path
    #         path = []
    #         for i in range(len(self.control_points)-1):
    #             start = self.control_points[i]
    #             goal = self.control_points[i+1]
    #             path.extend(self.astar(tuple(start), tuple(goal)))
    #         return path
    #     else:
    #         # Logic for traveling salesman problem (TSP)
    #         shortest_path = []
    #         shortest_distance = float('inf')

    #         # Start from the first control point
    #         start_point = self.control_points[0]
    #         for permutation in itertools.permutations(self.control_points[1:]):
    #             current_path = [start_point]
    #             current_distance = 0

    #             # Calculate path and distance for this permutation
    #             for i in range(len(permutation)):
    #                 current_path.append(permutation[i])
    #                 current_distance += self.heuristic(current_path[-2], current_path[-1])

    #             # Compare with the shortest path found so far
    #             if current_distance < shortest_distance:
    #                 shortest_distance = current_distance
    #                 shortest_path = current_path

    #         # Generate the actual path using A* between the points in shortest_path
    #         final_path = []
    #         for i in range(len(shortest_path)-1):
    #             final_path.extend(self.astar(tuple(shortest_path[i]), tuple(shortest_path[i+1])))

    #         return final_path

































    # def __init__(self, xyz, voxel_size=0.1):
    #     self.voxel_size = voxel_size

    # def __call__(self, points, normals):
    #     return self.voxelize(points, normals)

    # def voxelize(self, points, normals):    
    #     pass

    # def xyz_to_binary_voxel(self, point_cloud, n=30, plot=False):
    #     """"Point cloud as N,3 numpy array of XYZ coordinates
    #         with n the number of voxels in each dimension of the voxel grid"""

    #     # create a dataframe with the atomic coordinates
    #     df = pd.DataFrame(data=point_cloud, columns=['x','y','z'])

    #     # create a PyntCloud object with the dataframe
    #     cloud = PyntCloud(df)

    #     # plot the point cloud  
    #     voxelgrid_id = cloud.add_structure("voxelgrid",n_x=n, n_y=n, n_z=n)
    #     voxelgrid = cloud.structures[voxelgrid_id]

    #     if plot:
    #         voxelgrid.plot(d=3, mode="density", cmap="hsv")

    #     # get the voxel grid as a numpy array as a binary array
    #     binary_voxel_array = voxelgrid.get_feature_vector(mode="binary")

    #     return voxelgrid, binary_voxel_array

    # def get_COM_selections(traj, QGR_resids):
    #     coms = []
    #     for ids in QGR_resids:
    #         selection = traj.top.select(f'resid {ids[0]} {ids[1]} {ids[2]}')
    #         com = md.compute_center_of_mass(traj.atom_slice(selection))
    #         coms.append(com)
    #     return np.array([list(l) for l in list(np.array(coms).swapaxes(0,1)[0])])


    # def get_voxel_indices(points, voxelgrid):
    #     # Calculate voxel indices based on the voxel grid segments 
    #     voxel_x = np.clip(np.searchsorted(voxelgrid.segments[0], points[:, 0]) - 1, 0, voxelgrid.x_y_z[0])
    #     voxel_y = np.clip(np.searchsorted(voxelgrid.segments[1], points[:, 1]) - 1, 0, voxelgrid.x_y_z[1])
    #     voxel_z = np.clip(np.searchsorted(voxelgrid.segments[2], points[:, 2]) - 1, 0, voxelgrid.x_y_z[2])

    #     # Convert voxel indices to voxel numbers
    #     voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], voxelgrid.x_y_z)

    #     # Get the unique voxel numbers of coms
    #     voxel_indices = np.array([voxel_x, voxel_y, voxel_z]).T

    #     return voxel_indices, voxel_n

    # def visualize_voxels(voxelgrid, voxel_n, ids, n):
    #     """
    #     Visualizes the voxels in a 3D plot.

    #     Args:
    #         voxelgrid (VoxelGrid): The voxel grid object.
    #         voxel_n (list): List of voxel indices.
    #         ids (list): List of voxel IDs.

    #     Returns:
    #         None
    #     """

    #     # Create boolean arrays to mark the presence of voxels
    #     # C: This boolean array marks the presence of all voxels in the voxel grid.
    #     # C_: This boolean array marks the presence of specific voxels identified by the voxel_n parameter.
    #     # C__: This boolean array marks the presence of specific voxels identified by the ids parameter.

    #     C = np.array([True if idx in voxelgrid.voxel_n else False for idx,_ in enumerate(voxelgrid.voxel_centers)]).reshape(n,n,n)
    #     C_ = np.array([True if idx in voxel_n else False for idx,_ in enumerate(voxelgrid.voxel_centers)]).reshape(n,n,n)
    #     C__ = np.array([True if idx in ids else False for idx,_ in enumerate(voxelgrid.voxel_centers)]).reshape(n,n,n)

    #     # Create a 3D plot
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')

    #     # Plot the voxels
    #     ax.voxels(C, edgecolor='navy', shade=True, facecolor='cornflowerblue', alpha=0.5, linewidth=0.01)
    #     ax.voxels(C_, edgecolor='navy', shade=True, facecolor='coral')
    #     ax.voxels(C__, edgecolor='navy', shade=True, facecolor='green')


    # def find_nearest_zero_voxel(voxel_array, voxel_index):
    #     # Check if the given voxel is 1
    #     # print(voxel_array[voxel_index[0], voxel_index[1], voxel_index[2]])

    #     # Get the coordinates of all voxels with value 0
    #     zero_voxels = np.argwhere(voxel_array == 0)

    #     # Calculate the distances from the given voxel to all zero voxels
    #     distances = cdist([voxel_index], zero_voxels)

    #     # Find the index of the nearest zero voxel
    #     nearest_zero_voxel_index = zero_voxels[np.argmin(distances)]

    #     return tuple(nearest_zero_voxel_index)

    # def get_control_points(voxel_indices, binary_voxels, voxelgrid, voxel_n):
    #     # Checks if the center of mass of the QGR residues is inside a voxel or not
    #     # And if it is move it to the nearest zero voxel in the voxel grid

    #     control_points = []
    #     ids = []
    #     for idx,xyz in enumerate(voxel_indices):
    #         value = binary_voxels[xyz[0], xyz[1], xyz[2]]
    #         print('\n')
    #         if value == 1:
    #             # com is inside voxel space
    #             #print("COM {} is inside voxel".format(xyz))
    #             new_index = find_nearest_zero_voxel(binary_voxels, (xyz))
    #             #new_values = binary_voxels[new_index[0], new_index[1], new_index[2]]
    #             new_id = np.ravel_multi_index([new_index[0], new_index[1], new_index[2]], voxelgrid.x_y_z)
    #             #print("COM {} is nearest zero voxel {}".format(xyz, new_index))
    #             control_points.append(new_index)
    #             ids.append(new_id)
    #         else:
    #             # com is outside voxel space
    #             #print("COM {} is outside voxel".format(xyz))
    #             control_points.append(xyz)
    #             ids.append(voxel_n[idx])

    #     control_points = np.array(control_points)
    #     return control_points, ids