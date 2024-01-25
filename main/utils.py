
import numpy as np

class RigidBody:

    @staticmethod
    def extract_omega_values(R):
        """
        Extracts Euler angle (omega) values from a rotation matrix.

        Args:
            R (np.ndarray): Rotation matrix of shape (n, 3, 3).

        Returns:
            np.ndarray: Euler angles (omegavec) of shape (n, 3).
        """
        if np.any(np.trace(R, axis1=-2, axis2=-1) + 1 < 1e-12):
            print("Invalid Trace!")
            print(R)
            return np.zeros((R.shape[0], 3)) 

        #  add check for RuntimeWarning: invalid value encountered in arccos
        omega = np.arccos(0.5 * (np.trace(R, axis1=-2, axis2=-1) - 1))
        if np.isnan(omega).any():
            print("Nan value found!")
            print(omega)
            print("++++++++++++++++++++++++")

        omegavec = np.where(omega[..., None] < 1e-12, 
                            np.zeros((R.shape[0], 3)), 
                            omega[..., None] * 1 / (2 * np.sin(omega[..., None])) * np.stack([
                                R[..., 2, 1] - R[..., 1, 2],
                                R[..., 0, 2] - R[..., 2, 0],
                                R[..., 1, 0] - R[..., 0, 1]
                            ], axis=-1))

        return omegavec
    
    @staticmethod
    def get_rotation_matrix(omega):
        """
        Returns the rotation matrix corresponding to the Euler vector (omega).

        Args:
            omega (np.ndarray): Euler vector of shape (n, 3).

        Returns:
            np.ndarray: Rotation matrix of shape (n, 3, 3).
        """
        # Normalize the Euler vector
        omega_norm = np.linalg.norm(omega, axis=-1, keepdims=True)

        # Add check for RuntimeWarning: invalid value encountered in divide
        omega_normalized = np.where(omega_norm != 0, omega / omega_norm, omega)

        # Compute the angle of rotation
        theta = omega_norm.squeeze()

        # Compute the components of the Euler vector
        wx, wy, wz = omega_normalized.T

        # Compute intermediate values
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        one_minus_cos_theta = 1 - cos_theta

        # Compute the rotation matrix
        rotation_matrix = np.array([
            [cos_theta + wx**2 * one_minus_cos_theta, wx * wy * one_minus_cos_theta - wz * sin_theta, wx * wz * one_minus_cos_theta + wy * sin_theta],
            [wx * wy * one_minus_cos_theta + wz * sin_theta, cos_theta + wy**2 * one_minus_cos_theta, wy * wz * one_minus_cos_theta - wx * sin_theta],
            [wx * wz * one_minus_cos_theta - wy * sin_theta, wy * wz * one_minus_cos_theta + wx * sin_theta, cos_theta + wz**2 * one_minus_cos_theta]
        ]).transpose((2, 0, 1))

        return rotation_matrix
    
    @staticmethod
    def rotate_vector(v, k, theta):
        """
        Rotate vector v around axis k by angle theta using Euler vector representation.
        :param v: np.array, vector to be rotated
        :param k: np.array, axis of rotation (unit vector)
        :param theta: float, angle of rotation in radians
        :return: np.array, rotated vector
        """
        # Compute the Euler vector
        omega = k * theta
        # Reshape the Euler vector to add one dimension for batching
        omega_batched = omega[None, :]
        # Compute the rotation matrix using the Euler vector
        R = RigidBody.get_rotation_matrix(omega_batched)
        # Apply the rotation and squeeze to remove the additional dimension
        return np.dot(R[0], v)
    

class Shapes:
    
    def __init__(self, parametric_function, t_values=None, num_points=100):
        self.num_points = num_points
        self.parametric_function = parametric_function
        self.points = self._generate_points(t_values)
    
    def _generate_points(self,t_values=None):
        x_values, y_values, z_values = self.parametric_function(t_values)
        return np.stack((x_values, y_values, z_values), axis=1)

    @classmethod
    def circle(cls, radius=1, t_values=None, num_points=100):
        if t_values is None:
            t_values = np.linspace(0, 2 * np.pi, num=num_points)
        parametric_function = lambda t_values: (
            radius * np.cos(t_values),
            radius * np.sin(t_values),
            np.zeros_like(t_values)
        )
        return cls(parametric_function, t_values, num_points=num_points).points

    @classmethod
    def line(cls, length=1, num_points=100):
        t_values = np.linspace(0, 1, num=num_points)
        parametric_function = lambda t_values: (
            t_values * length,
            np.zeros_like(t_values),
            np.zeros_like(t_values)
        )
        return cls(parametric_function, t_values, num_points=num_points).points
    
    @classmethod
    def helix(cls, radius=1, pitch=1, height=1, num_turns=1, num_points=100):
        t_values = np.linspace(0, num_turns * 2 * np.pi, num=num_points)
        parametric_function = lambda t_values: (
            radius * np.cos(t_values),
            radius * np.sin(t_values),
            height * t_values / (2 * np.pi) - pitch * num_turns * t_values / (2 * np.pi)
        )
        return cls(parametric_function, t_values, num_points=num_points).points
    
    @classmethod
    def spiral(cls, radius=1, pitch=1, height=1, num_turns=1, num_points=100):
        t_values = np.linspace(0, num_turns * 2 * np.pi, num=num_points)
        parametric_function = lambda t_values: (
            radius * t_values * np.cos(t_values),
            radius * t_values * np.sin(t_values),
            height * t_values / (2 * np.pi) * pitch
        )
        return cls(parametric_function, t_values, num_points=num_points).points
    
    @classmethod
    def mobius_strip(cls, radius=1, width=0.5, num_twists=1, t_values=None, num_points=100):
        if t_values is None:
            t_values = np.linspace(0, 2 * np.pi, num=num_points)
        u_values = np.linspace(0, width, num=num_points)
        u, t = np.meshgrid(u_values, t_values)
        x_values = (radius + u * np.cos(t / 2) * np.cos(num_twists * t)) * np.cos(t)
        y_values = (radius + u * np.cos(t / 2) * np.cos(num_twists * t)) * np.sin(t)
        z_values = u * np.sin(t / 2) * np.cos(num_twists * t)
        parametric_function = lambda t_values: (
            x_values.flatten(),
            y_values.flatten(),
            z_values.flatten()
        )
        return cls(parametric_function, t_values, num_points=num_points).points
    
    @classmethod
    def square(cls, side_length=1,t_values=None,num_points=100):
        if t_values is None:
            t_values = np.linspace(0, 1, num=num_points)
        parametric_function = lambda t_values: (
            side_length * (2 * (t_values < 0.25) - 1),
            side_length * (2 * (t_values >= 0.25) & (t_values < 0.5)) - side_length,
            np.zeros_like(t_values)
        )
        return cls(parametric_function, t_values).points

    @classmethod
    def trefoil(cls, radius=1, num_turns=3,t_values=None,num_points=100):
        if t_values is None:
            t_values = np.linspace(0, num_turns * 2 * np.pi, num=num_points)
        x_values = np.sin(t_values) + 2 * np.sin(2 * t_values)
        y_values = np.cos(t_values) - 2 * np.cos(2 * t_values)
        z_values = -np.sin(3 * t_values)
        parametric_function = lambda t_values: (
            radius * x_values,
            radius * y_values,
            radius * z_values
        )
        return cls(parametric_function, t_values).points
    
    @classmethod
    def square(cls, side=1, t_values=None,num_points=100):
        if t_values is None:
            t_values = np.linspace(0, 4, num=num_points)
        # Calculate x and y coordinates based on t_values
        x_values = np.zeros_like(t_values)
        y_values = np.zeros_like(t_values)
        for i, t in enumerate(t_values):
            if 0 <= t < 1:
                x_values[i] = t * side
                y_values[i] = 0
            elif 1 <= t < 2:
                x_values[i] = side
                y_values[i] = (t - 1) * side
            elif 2 <= t < 3:
                x_values[i] = (3 - t) * side
                y_values[i] = side
            elif 3 <= t <= 4:
                x_values[i] = 0
                y_values[i] = (4 - t) * side
        z_values = np.zeros_like(t_values)
        parametric_function = lambda t_values: (
            x_values,
            y_values,
            z_values
        )
        return cls(parametric_function, t_values).points
    
    @classmethod
    def heart(cls, a=1, b=1, c=1,t_values=None,num_points=100):
        if t_values is None:
            t_values = np.linspace(-np.pi, np.pi, num=num_points)
        x_values = a * np.sin(t_values) ** 3
        y_values = b * np.cos(t_values) - c * np.cos(2 * t_values)
        z_values = np.zeros_like(t_values)
        parametric_function = lambda t_values: (x_values, y_values, z_values)
        return cls(parametric_function, t_values).points

    @classmethod
    def ellipse(cls, a=1, b=1, t_values=None,num_points=100):
        if t_values is None:
            t_values = np.linspace(0, 2 * np.pi, num=num_points)
        x_values = a * np.cos(t_values)
        y_values = b * np.sin(t_values)
        z_values = np.zeros_like(t_values)
        parametric_function = lambda t_values: (x_values, y_values, z_values)
        return cls(parametric_function, t_values).points

    @classmethod
    def lemniscate_of_bernoulli(cls, a=1, b=1, t_values=None,num_points=100):
        if t_values is None:
            t_values = np.linspace(0, 2 * np.pi, num=num_points)
        x_values = a * np.sqrt(2) * np.cos(t_values) / (np.sin(t_values) ** 2 + 1)
        y_values = b * np.sqrt(2) * np.cos(t_values) * np.sin(t_values) / (np.sin(t_values) ** 2 + 1)
        z_values = np.zeros_like(t_values)
        parametric_function = lambda t_values: (x_values, y_values, z_values)
        return cls(parametric_function, t_values).points
    
    @classmethod
    def torus_helix(cls, R=1, r=2, num_windings=3, t_values=None, num_points=100):
        if t_values is None:
            t_values = np.linspace(0, 2 * np.pi, num=num_points)

        parametric_function = lambda t_values: (
            (R + r * np.cos(num_windings*t_values)) * np.cos( t_values),
            (R + r * np.cos(num_windings*t_values)) * np.sin( t_values),
            r * np.sin(t_values)
        )
        return cls(parametric_function, t_values, num_points=num_points).points
    
    @classmethod
    def bonus(cls, t_values=None,num_points=100):
        "https://www.geogebra.org/m/pH8wD3rW, Author:Simona Riva"
        if t_values is None:
            t_values = np.linspace(0, 2 * np.pi, num=num_points)
        t = t_values
        parametric_function = lambda t: (
                                        -(721*np.sin(t))/4 + 196/3*np.sin(2*t) - 86/3*np.sin(3*t) - 131/2*np.sin(4*t) + 477/14*np.sin(5*t) 
                                        + 27*np.sin(6*t) - 29/2*np.sin(7*t) + 68/5*np.sin(8*t) + 1/10*np.sin(9*t) + 23/4*np.sin(10*t) 
                                        - 19/2*np.sin(12*t) - 85/21*np.sin(13*t) + 2/3*np.sin(14*t) + 27/5*np.sin(15*t) + 7/4*np.sin(16*t) 
                                        + 17/9*np.sin(17*t) - 4*np.sin(18*t) - 1/2*np.sin(19*t) + 1/6*np.sin(20*t) + 6/7*np.sin(21*t) 
                                        - 1/8*np.sin(22*t) + 1/3*np.sin(23*t) + 3/2*np.sin(24*t) + 13/5*np.sin(25*t) + np.sin(26*t) 
                                        - 2*np.sin(27*t) + 3/5*np.sin(28*t) - 1/5*np.sin(29*t) + 1/5*np.sin(30*t) + (2337*np.cos(t))/8 
                                        - 43/5*np.cos(2*t) + 322/5*np.cos(3*t) - 117/5*np.cos(4*t) - 26/5*np.cos(5*t) - 23/3*np.cos(6*t) 
                                        + 143/4*np.cos(7*t) - 11/4*np.cos(8*t) - 31/3*np.cos(9*t) - 13/4*np.cos(10*t) - 9/2*np.cos(11*t) 
                                        + 41/20*np.cos(12*t) + 8*np.cos(13*t) + 2/3*np.cos(14*t) + 6*np.cos(15*t) + 17/4*np.cos(16*t) 
                                        - 3/2*np.cos(17*t) - 29/10*np.cos(18*t) + 11/6*np.cos(19*t) + 12/5*np.cos(20*t) + 3/2*np.cos(21*t) 
                                        + 11/12*np.cos(22*t) - 4/5*np.cos(23*t) + np.cos(24*t) + 17/8*np.cos(25*t) - 7/2*np.cos(26*t) 
                                        - 5/6*np.cos(27*t) - 11/10*np.cos(28*t) + 1/2*np.cos(29*t) - 1/5*np.cos(30*t),
                                        -(637/2)*np.sin(t) - (188/5)*np.sin(2*t) - (11/7)*np.sin(3*t) - (12/5)*np.sin(4*t) + (11/3)*np.sin(5*t)
                                        - (37/4)*np.sin(6*t) + (8/3)*np.sin(7*t) + (65/6)*np.sin(8*t) - (32/5)*np.sin(9*t) - (41/4)*np.sin(10*t)
                                        - (38/3)*np.sin(11*t) - (47/8)*np.sin(12*t) + (5/4)*np.sin(13*t) - (41/7)*np.sin(14*t) - (7/3)*np.sin(15*t)
                                        - (13/7)*np.sin(16*t) + (17/4)*np.sin(17*t) - (9/4)*np.sin(18*t) + (8/9)*np.sin(19*t) + (3/5)*np.sin(20*t)
                                        - (2/5)*np.sin(21*t) + (4/3)*np.sin(22*t) + (1/3)*np.sin(23*t) + (3/5)*np.sin(24*t) - (3/5)*np.sin(25*t)
                                        + (6/5)*np.sin(26*t) - (1/5)*np.sin(27*t) + (10/9)*np.sin(28*t) + (1/3)*np.sin(29*t) - (3/4)*np.sin(30*t)
                                        - (125/2)*np.cos(t) - (521/9)*np.cos(2*t) - (359/3)*np.cos(3*t) + (47/3)*np.cos(4*t) - (33/2)*np.cos(5*t)
                                        - (5/4)*np.cos(6*t) + (31/8)*np.cos(7*t) + (9/10)*np.cos(8*t) - (119/4)*np.cos(9*t) - (17/2)*np.cos(10*t)
                                        + (22/3)*np.cos(11*t) + (15/4)*np.cos(12*t) - (5/2)*np.cos(13*t) + (19/6)*np.cos(14*t) + (7/4)*np.cos(15*t)
                                        + (31/4)*np.cos(16*t) - np.cos(17*t) + (11/10)*np.cos(18*t) - (2/3)*np.cos(19*t) + (13/3)*np.cos(20*t)
                                        - (5/4)*np.cos(21*t) + (2/3)*np.cos(22*t) + (1/4)*np.cos(23*t) + (5/6)*np.cos(24*t) + (3/4)*np.cos(26*t)
                                        - (1/2)*np.cos(27*t) - (1/10)*np.cos(28*t) - (1/3)*np.cos(29*t) - (1/19)*np.cos(30*t),
                                        np.zeros_like(t))
        return cls(parametric_function, t_values).points*0.1