#!/usr/bin/env python3

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt

# from hw1.groups import Group, GroupElement, RepresentationGroup, RepresentationGroupElement


class TangentVector:
    def __init__(self, value, configuration) -> None:
        self.value = np.array(value).reshape(-1, 1)  # Ensure it's a column vector
        self.configuration = configuration

    def __add__(self, other):
        if isinstance(other, TangentVector):
            if self.configuration == other.configuration:
                return TangentVector(self.value + other.value, self.configuration)
            else:
                raise ValueError("Tangent vectors must be at the same configuration to be added.")
        else:
            raise TypeError("Addition is only supported between TangentVector instances.")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, TangentVector):
            if self.configuration == other.configuration:
                return TangentVector(self.value - other.value, self.configuration)
            else:
                raise ValueError("Tangent vectors must be at the same configuration to be subtracted.")
        else:
            raise TypeError("Subtraction is only supported between TangentVector instances.")
    
    def __rsub__(self, other):
        if isinstance(other, TangentVector):
            return other.__sub__(self)
        else:
            raise TypeError("Subtraction is only supported between TangentVector instances.")
        
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return TangentVector(self.value * other, self.configuration)
        elif isinstance(other, np.ndarray):
            # Matrix multiplication: matrix * vector
            if other.shape[1] == self.value.shape[0]:
                return TangentVector(other @ self.value, self.configuration)
            else:
                raise ValueError("Matrix dimensions must match for left multiplication (matrix should have same number of columns as vector's rows).")
        else:
            raise TypeError("Multiplication is only supported for scalars or matrices with appropriate dimensions.")
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return self.__mul__(other)
        elif isinstance(other, np.ndarray):
            # Right matrix multiplication: vector * matrix
            if self.value.shape[0] == other.shape[0]:
                return TangentVector(self.value.T @ other, self.configuration)
            else:
                raise ValueError("Matrix dimensions must match for right multiplication (matrix should have same number of rows as vector's rows).")
        else:
            raise TypeError("Multiplication is only supported for scalars or matrices with appropriate dimensions.")
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            # Scalar division
            return TangentVector(self.value / other, self.configuration)
        elif isinstance(other, np.ndarray):
            # Left matrix division: vector / matrix -> vector * inv(matrix)
            if other.shape[0] == other.shape[1] == self.value.shape[0]:
                try:
                    inv_matrix = np.linalg.inv(other)
                    return TangentVector(inv_matrix @ self.value, self.configuration)
                except np.linalg.LinAlgError:
                    raise ValueError("Matrix inversion failed. The matrix may not be invertible.")
            else:
                raise ValueError("Matrix must be square and match vector dimensions for division.")
        else:
            raise TypeError("Division is only supported for scalars or square matrices with matching dimensions.")
    
    def __rtruediv__(self, other):
        if isinstance(other, np.ndarray):
            # Right matrix division: matrix / vector -> inv(vector) * matrix
            if self.value.shape[0] == other.shape[1] and self.value.shape[0] == self.value.shape[1]:
                try:
                    inv_vector = np.linalg.inv(self.value)
                    return TangentVector(other @ inv_vector, self.configuration)
                except np.linalg.LinAlgError:
                    raise ValueError("Matrix inversion failed. The vector matrix may not be invertible.")
            else:
                raise ValueError("Matrix must be square and match vector dimensions for division.")
        else:
            raise TypeError("Right division is only supported for square matrices with dimensions matching the vector.")
    
    def __repr__(self):
        return f"TangentVector(value={self.value.flatten()}, configuration={self.configuration})"
    
def derivative_in_direction(differentiated_function, evaluation_point):
    """
    Computes the derivative of the differentiated_function in the direction at the evaluation_point.
    
    Parameters:
    - differentiated_function: A function that takes a scalar input and an array input.
    - evaluation_point: An array (NumPy array or list) representing the evaluation point.
    
    Returns:
    - TangentVector: A tangent vector whose value is the derivative at the evaluation point.
    """
    # Ensure evaluation_point is a NumPy array
    evaluation_point = np.array(evaluation_point)
    
    # Define a lambda function to reduce the differentiated_function to just the scalar input
    reduced_function = lambda scalar: differentiated_function(scalar, evaluation_point)
    
    # Compute the derivative of the reduced function with respect to the scalar input at 0
    derivative_value = nd.Derivative(reduced_function)(0)
    
    # The value of the tangent vector is the derivative, and it is located at the evaluation point
    return TangentVector(value=derivative_value * np.ones_like(evaluation_point), configuration=evaluation_point)

def part2():
    def rad_rot_func(delta, point):
        """
        Equation 2.49 (page 142) with the addtion of a scalar delta
        """
        # phi = 1
        # theta = 1

        # radial_trans = (1 + phi/np.sqrt(point[0]**2 + point[1]**2))

        # Calculate polar coordinates from the point
        r = phi = np.sqrt(point[0]**2 + point[1]**2)  # Radial distance
        theta = np.arctan2(point[1], point[0])  # Angle in radians
        radial_trans = (1 + phi / r) if r != 0 else 1  # Avoid division by zero

        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])

        return delta * radial_trans * (rotation @ point.reshape(2, 1))
    
    def plot_vector_field(xy, vectors):
        # Plot the arrows using quiver
        plt.figure(figsize=(6,6))
        plt.quiver(xy[:, 0], xy[:, 1], 
                vectors[:, 0], vectors[:, 1], 
                angles='xy', scale_units='xy', scale=3, color='blue')

        # Add grid and axis labels
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        plt.title('Arrows Representing Polar Bases Vectors')
        plt.show()
    
    # Step 1: Generate a grid of points
    x = y = np.linspace(-2, 2, 5)
    X, Y = np.meshgrid(x, y)

    # Step 2: Flatten the grid
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Step 3: Combine into xy array
    configuration = np.vstack((X_flat, Y_flat)).T

    # Step 4: Compute vector values at each configuration
    polar_bases_vals = np.stack([derivative_in_direction(rad_rot_func, config).value for config in configuration])
    plot_vector_field(configuration, polar_bases_vals)

if __name__ == '__main__':
    part2()
