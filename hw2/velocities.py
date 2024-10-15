#!/usr/bin/env python3

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("/home/marcus/classes/rob541/geometric_mechanics/")
from hw1.groups import Group, GroupElement, RepresentationGroup, RepresentationGroupElement


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
    
class GroupTangentVector(TangentVector):
    def __init__(self, configuration):
        # Check if the configuration is a GroupElement
        if not isinstance(configuration, GroupElement):
            raise TypeError("Configuration must be a GroupElement.")
        
        # # Call the parent class constructor
        # super().__init__(value, configuration)
        
        # Copy the Group property from the GroupElement configuration
        self.group = configuration.group

class GroupwiseBasis:
    def __init__(self, group_elements):
        """Constructor that takes in a list of group elements."""
        self.group_elements = group_elements

    def scaled_group_action(self, scalar, g, h):
        """Scaled group action function."""
        # scaled_g = GroupElement(g.configuration * scalar)
        scaled_g = g.value * scalar
        return self.operation(scaled_g, h)

    def derivative_in_direction_of_group_action(self, g, h, direction='left'):
        """Derivative in the direction of a group action."""
        if direction == 'left':
            # Create a lambda function for left action
            func = lambda scalar: self.scaled_group_action(scalar, g, h)
        elif direction == 'right':
            # Implement right action if needed
            func = lambda scalar: self.operation(h, GroupElement(g.configuration * scalar))

        # Use numdifftools to compute the gradient
        gradient = nd.Gradient(func)
        return gradient(0)

    def __call__(self, h, direction='left'):
        """Evaluate the groupwise basis at a given group element."""
        vector_basis = []
        for g in self.group_elements:
            tangent_vector = self.derivative_in_direction_of_group_action(g, h, direction)
            vector_basis.append(tangent_vector)
        return np.array(vector_basis)

def derivative_in_direction(differentiated_function, config):
    """
    Computes the derivative of the differentiated_function in the direction at the config.
    
    Parameters:
    - differentiated_function: A function that takes a scalar input and an array input.
    - config: An array (NumPy array or list) representing the evaluation point.
    
    Returns:
    - TangentVector: A tangent vector whose value is the derivative at the evaluation point.
    """
    # Ensure config is a NumPy array
    config = np.array(config)
    
    # Define a lambda function to reduce the differentiated_function to just the scalar input
    reduced_function = lambda delta: differentiated_function(config, delta).flatten()
    
    # Evaluate the derivative at delta = 0
    # delta is set to zero during the differentiation process to find how the function changes
    # at that specific point (configuration) without any scaling applied yet
    derivative_value = nd.Derivative(reduced_function)(0)

    # Reshape the result to match the configuration dimensions
    derivative_value = derivative_value.reshape(config.shape)
    
    # Return a TangentVector at the config with the computed derivative value
    return TangentVector(value=derivative_value, configuration=config)

def compute_jacobian(func, config):
    """
    Computes the Jacobian of a function `f` at a given point.
    """
    # Wrap the function to only take the configuration input and keep delta fixed
    wrapped_func = lambda delta: func(config, float(delta)).flatten()
    
    # Calculate the Jacobian matrix using numdifftools.Jacobian
    jacobian = nd.Jacobian(wrapped_func)

    # Evaluate the jacobian with a delta fixed at 0
    jacobian_matrix = jacobian(0)
    
    # Construct and return a TangentVector
    return TangentVector(value=jacobian_matrix, configuration=config)

def part2():
    def rect_to_polar(point, delta=0):
        """
        An example function that applies a radial transformation and rotation based on delta.
        When delta = 0, it returns the original configuration.
        """
        # Polar coordinates
        r = np.sqrt(point[0]**2 + point[1]**2)
        
        # Radial transformation factor - make the radial trans one if division by zero
        radial_trans = (1 + delta / r) if r != 0 else 1
        
        # Rotation matrix
        theta = delta  # Small rotation proportional to delta
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        
        # Apply radial transformation and rotation
        transformed_point = radial_trans * (rotation @ point.reshape(2, 1))
        
        return transformed_point.flatten()
    
    def plot_vector_field(xy, vectors, title='Arrows Representing Polar Bases Vectors'):
        # Plot the arrows using quiver
        plt.figure(figsize=(6,6))
        plt.quiver(xy[:, 0], xy[:, 1], 
                vectors[:, 0], vectors[:, 1], 
                angles='xy', scale_units='xy', scale=3, color='blue')

        # Add grid, axis labels, and set equal axes scaling
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        plt.title(f'{title}')
        plt.show()
    
    # Step 1: Generate a grid of points
    x = y = np.linspace(-2, 2, 5)
    X, Y = np.meshgrid(x, y)

    # Step 2: Flatten the grid
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Step 3: Combine into xy array
    configuration = np.vstack((X_flat, Y_flat)).T

    # # Step 4: Compute vector values at each configuration
    # # DELIVERABLE 1
    polar_derivative_pts = np.stack([derivative_in_direction(rect_to_polar, config).value for config in configuration])
    plot_vector_field(configuration, polar_derivative_pts, title='Polar Basis Field - derivative')

    # TODO Step 5: Equivelant Jacobian mapping
    # DELIVERABLE 2
    polar_jacobian_pts = np.array([compute_jacobian(rect_to_polar, config).value for config in configuration])
    plot_vector_field(configuration, polar_jacobian_pts, title='Polar Basis Field - Jacobian')

def part3():
    # DELIVERABLE 1
    # Replicate the groupwise vector basis fields in Figure 2.10 b and c using your direction-derivative function 
    # with the group actions of elements that are δ away from the identity in single components.

    # Page 152
    # TODO: (b) the partial derivative with respect to the left action L_g, evaluated at the group element h
    # TODO: (c) the partial derivative with respect to the riht action R_h, evaluated at the group element g

    def scale_shift(input1, input2):
        return np.array([input1[0] * input2[0], input1[1] + input2[1]])
    
    def rep(x):
        return np.array([[1, 0, x[0]],
                         [0, 1, x[1]],
                         [0, 0, 1]])
    
    def derep(x):
        return np.array([x[0, 1], x[1, 1]])

    g_val = np.array([0, 1])
    h_val = np.array([1, 0])
    identity = np.eye(2)
    
    # Create a representation group
    rep_group = RepresentationGroup(rep, identity=identity, derepresentation_function=derep)
    rep_group.operation = scale_shift

    # Create representation group elements
    g = RepresentationGroupElement(rep_group, g_val)
    h = RepresentationGroupElement(rep_group, h_val)

    # Create a list of group elements for the basis
    basis_elements = GroupTangentVector()

    # Instantiate the GroupwiseBasis class
    groupwise_basis = GroupwiseBasis(basis_elements)

    # Evaluate the groupwise basis at the element h
    vector_basis = groupwise_basis(h)

    print("Groupwise Basis Vectors at h:")
    print(vector_basis)

if __name__ == '__main__':
    part2()
    # part3()
