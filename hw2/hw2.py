#!/usr/bin/env python3

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("/home/marcus/classes/rob541/geometric_mechanics/")
from geo_classes.groups import Group, GroupElement, RepresentationGroup, RepresentationGroupElement
from geo_classes.velocities import TangentVector, GroupTangentVector, GroupwiseBasis


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

def plot_vector_field(xy, vectors, vectors2=None, title='Arrows Representing Polar Bases Vectors'):
    # Plot the arrows using quiver
    plt.figure(figsize=(6,6))
    plt.quiver(xy[:, 0], xy[:, 1], 
            vectors[:, 0], vectors[:, 1], 
            angles='xy', scale_units='xy', scale=3, color='blue')
    
    if vectors2 is not None:
        # Plot vectors2 in red
        plt.quiver(xy[:, 0], xy[:, 1], 
                vectors2[:, 0], vectors2[:, 1], 
                angles='xy', scale_units='xy', scale=3, color='red', label='Y Left Vectors')

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

def part2(configuration):
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

    # Step 4: Compute vector values at each configuration
    # DELIVERABLE 1
    polar_derivative_pts = np.stack([derivative_in_direction(rect_to_polar, config).value for config in configuration])
    plot_vector_field(configuration, polar_derivative_pts, title='Polar Basis Field - derivative')

    # Step 5: Equivelant Jacobian mapping
    # DELIVERABLE 2
    polar_jacobian_pts = np.array([compute_jacobian(rect_to_polar, config).value for config in configuration])
    plot_vector_field(configuration, polar_jacobian_pts, title='Polar Basis Field - Jacobian')

def part3(representation_group, configuration):
    # DELIVERABLE 1
    # Replicate the groupwise vector basis fields in Figure 2.10 b and c using your direction-derivative function 
    # with the group actions of elements that are δ away from the identity in single components.

    # Page 152
    # (b) the partial derivative with respect to the left action L_g, evaluated at the group element h
    # (c) the partial derivative with respect to the riht action R_h, evaluated at the group element g

    def fx_left(h_val, delta):
        g_val = np.array([delta, 0])
        g = RepresentationGroupElement(representation_group, g_val)
        h = RepresentationGroupElement(representation_group, h_val)
        gh = g.left(h)
        gh_val = gh.derepresentation
        return gh_val
    
    def fy_left(h_val, delta):
        g_val = np.array([0, delta])
        g = RepresentationGroupElement(representation_group, g_val)
        h = RepresentationGroupElement(representation_group, h_val)
        gh = g.left(h)
        gh_val = gh.derepresentation
        return gh_val

    def fx_right(h_val, delta):
        g_val = np.array([delta, 0])
        g = RepresentationGroupElement(representation_group, g_val)
        h = RepresentationGroupElement(representation_group, h_val)
        gh = g.right(h)
        gh_val = gh.derepresentation
        return gh_val
    
    def fy_right(h_val, delta):
        g_val = np.array([0, delta])
        g = RepresentationGroupElement(representation_group, g_val)
        h = RepresentationGroupElement(representation_group, h_val)
        gh = g.right(h)
        gh_val = gh.derepresentation
        return gh_val
    
    x_left_pts = np.stack([derivative_in_direction(fx_left, config).value for config in configuration])
    y_left_pts = np.stack([derivative_in_direction(fy_left, config).value for config in configuration])
    x_right_pts = np.stack([derivative_in_direction(fx_right, config).value for config in configuration])
    y_right_pts = np.stack([derivative_in_direction(fy_right, config).value for config in configuration])

    plot_vector_field(configuration, x_left_pts, vectors2=y_left_pts, title='Semi-Direct Product (Left Action)')
    plot_vector_field(configuration, x_right_pts, vectors2=y_right_pts, title='Semi-Direct Product (Right Action)')

def part4(representation_group, configuration): 
    group_elements = [RepresentationGroupElement(representation_group, np.array([config[0], config[1]])) for config in configuration]

    g_val_x = np.array([1, 0])
    g_val_y = np.array([0, 1])

    g_x = RepresentationGroupElement(representation_group, g_val_x)
    g_y = RepresentationGroupElement(representation_group, g_val_y)

    x_left_pts = np.stack([g_x.left_lifted(group_element).derepresentation for group_element in group_elements])
    y_left_pts = np.stack([g_y.left_lifted(group_element).derepresentation for group_element in group_elements])

    x_right_pts = np.stack([g_x.right_lifted(group_element).derepresentation for group_element in group_elements])
    y_right_pts = np.stack([g_y.right_lifted(group_element).derepresentation for group_element in group_elements])

    plot_vector_field(configuration, x_left_pts, y_left_pts)
    plot_vector_field(configuration, x_right_pts, y_right_pts)


if __name__ == '__main__':
    def rep(x):
        return np.array([[x[0], x[1]],
                        [0, 1]])
    
    def derep(x):
        return np.array([x[0, 0], x[0, 1]])

    identity = np.eye(2)

    # Create a representation group
    rep_group = RepresentationGroup(rep, identity=identity, derepresentation_function=derep)

    # Set up a vector field grid
    x = y = np.linspace(-2, 2, 5)
    X, Y = np.meshgrid(x, y)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    configuration = np.vstack((X_flat, Y_flat)).T

    # part2(configuration)
    # part3(rep_group, configuration)
    part4(rep_group, configuration)
