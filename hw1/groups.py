#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class Group:
    def __init__(self, operation, identity: np.array, inverse_operation=None):
        """Class to construct groups 

        Args:
            operation (function): A function that takes in two inputs of the same size and returns an output also of the same size.
            identity (np.array): A numeric value that, when passed as one of the input values to the Operation function, results 
            in the output of Operation being equal to the other input passed to the function
        """
        self.operation = operation
        self.identity = identity
        self.inverse_operation = inverse_operation

    def element(self, value:np.array):
        """Takes in a numeric value and return an instance of a “group element” class

        Args:
            value (np.array): _description_

        Returns:
            GroupElement: a group element
        """
        if isinstance(self, RepresentationGroup):
            return RepresentationGroupElement(self, value)
        else:
            return GroupElement(self, value)

    def identity_element(self):
        """A function that takes no input, and returns the result of the “Element” function with the group's self.Identity as the input.
        """
        self.element(self.identity)


class GroupElement:
    def __init__(self, group: Group, value: np.array):
        """_summary_

        Args:
            group (Group): An instance of the Group class
            value (np.array): _description_
            inverse (function, optional): _description_. Defaults to None.
        """
        self.group = group
        self.value = value

    def left(self, group_element):
        """A function that takes in another instance of this “group element” class, extracts its Value, left-combines self.Value with 
        other.Value using self.Group.Operation, and then constructs a new group element from the result.

        Args:
            group_element (GroupElement): another instance of the GroupElement class

        Returns:
            GroupElement: an instance of the GroupElement class
        """
        value = group_element.value

        output = self.group.operation(self.value, value)

        return self.group.element(output)
    
    def right(self, group_element):
        """A function that takes in another instance of this “group element” class, extracts its Value, right-combines self.Value with
        other.Value using self.Group.Operation, and then constructs a new group element from the result.

        Args:
            group_element (GroupElement): another instance of the GroupElement class

        Returns:
            GroupElement: an instance of the GroupElement class
        """
        value = group_element.value

        output = self.group.operation(value, self.value)

        return self.group.element(output)
    
    def inverse_element(self):
        """A method that passes self.value to self.inverse_function, and then generates and returns a new group element constructed from 
        the inverted value

        Returns:
            GroupElement: an instance of the GroupElement class
        """
        return self.group.element(self.group.inverse_operation(self.value))
    
    def AD(self, element):
        """Compute the adjoint action of another group element."""
        return self.group.element(self.value @ element.value @ self.inverse_element().value)

    def AD_inv(self, element):
        """Compute the adjoint-inverse action of another group element."""
        return self.group.element(self.inverse_element().value @ element.value @ self.value)
    
class RepresentationGroup(Group):
    def __init__(self, representation_function, derepresentation_function, identity) -> None:
        self.representation_function = representation_function
        self.derepresentation_function = derepresentation_function
        self.identity = identity

        # If identity is passed as a list of parameters, compute the identity matrix
        if isinstance(identity, list):
            self.identity = representation_function(*identity)
        # Otherwise, assume identity is a matrix and set it directly
        elif isinstance(identity, np.ndarray):
            self.identity = identity
        else:
            raise ValueError("Identity must be a list of parameters or an identity matrix.")

        # Group operation is matrix multiplication
        super().__init__(np.matmul, self.identity)

    def representation(self, parameters):
        # Map the input parameters to the representation matrix
        # print(*parameters)
        return self.representation_function(parameters)
    
    def derepresentation(self, matrix):
        """Derepresent the matrix"""
        return self.derepresentation_function(matrix)

class RepresentationGroupElement(GroupElement):
    def __init__(self, group, value):
        self.group = group
        self.group.inverse_operation = np.linalg.inv
        
        # If value is already a matrix, set it as the value
        if isinstance(value, np.ndarray):
            if value.ndim == 2 and value.shape[0] == value.shape[1]:  # Check if it's a square matrix
                self.value = value
            elif value.ndim == 1:  # If it's a 1D array, assume it's a set of parameters
                self.value = group.representation(value)
            else:
                raise ValueError("The provided array must be a 1D array (parameters) or a square matrix.")
        # If value is a list of parameters, generate the representation matrix using the group's function
        elif isinstance(value, list):
            self.value = group.representation(value)
        else:
            raise ValueError("Value must be either a matrix, a list of parameters, or a 1D array.")

        # Ensure the matrix is square and invertible
        if self.value.shape[0] != self.value.shape[1]:
            raise ValueError("The matrix must be square.")
        if np.linalg.det(self.value) == 0:
            raise ValueError("The matrix must be invertible.")
    
    @property
    def derepresentation(self):
        """Return the derepresentation of the group element."""
        return self.group.derepresentation(self.value)

def product_scale_shift_test():
    def inv_func(input):
        return 1 / input

    def scale_shift(input1, input2):
        return np.array([input1[0] * input2[0], input1[1] + input2[1]])

    def scale_shift_inverse(input1, input2):
        return np.array([input1[0] / input2[0], input1[1] - input2[1]])
        
    g1 = np.array([3, -1])
    g2 = np.array([1/2, 3/2])
    identity = np.array([1, 0])

    group = Group(scale_shift, identity, inverse_operation=inv_func)

    g3 = group.element(g1).left(group.element(g2))
    g3_inv = group.element(g1).inverse_element().left(group.element(g1))

    print(g3.value)
    print(g3_inv.value)

def se2_plot(data, labels, xlim=(0, 3), ylim=(0, 3), title=None):
    # Define primary colors (RGB)
    primary_colors = ['red', 'green', 'blue', 'magenta', 'black', 'orange', 'yellow', 'cyan']

    # Ensure there are enough colors for the number of arrows
    num_arrows = data.shape[0]
    colors = primary_colors[:num_arrows]  # Select the first `num_arrows` primary colors

    # Create the plot
    plt.figure()

    # Plot the quiver with the primary colors for each arrow
    quiver = plt.quiver(data[:, 0], data[:, 1], 
                        np.cos(data[:, 2]), np.sin(data[:, 2]), 
                        color=colors, scale=20, headwidth=3, headlength=5, headaxislength=4)

    # Create legend entries by using dummy points with the same color as each arrow
    for i, label in enumerate(labels):
        plt.scatter([], [], color=colors[i], label=label)

    # Add legend
    plt.legend()

    # Labels, grid, and limits
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)

    # Show the plot
    plt.show()

def part1(printing=True):
    def trans_rot_se2_representation(group_element):
        return np.array([[np.cos(group_element[2]), -np.sin(group_element[2]), group_element[0]],
                        [np.sin(group_element[2]), np.cos(group_element[2]), group_element[1]],
                        [0, 0, 1]])

    def se2_derepresentation(trans_matrix):
        return np.array([trans_matrix[0, 2], trans_matrix[1, 2], np.arctan2(trans_matrix[1, 0], trans_matrix[0, 0])])  # Using atan2 for angle 

    def transform(input1, input2):
        top_left = input1[:2, :2] @ input2[:2, :2]
        top_right = input1[:2, :2] @ input2[:2, 2] + input1[:2, 2]

        output = np.eye(3)
        output[:2, :2] = top_left
        output[:2, 2] = top_right
        return output

    g_val = np.array([0, 1, -np.pi/4])
    h_val = np.array([1, 2, -np.pi/2])

    g = trans_rot_se2_representation(g_val)
    h = trans_rot_se2_representation(h_val)

    identity = np.eye(3)

    group = Group(transform, identity, np.linalg.inv)

    gh = group.element(g).left(group.element(h))
    gh_val = se2_derepresentation(gh.value)
    hg = group.element(g).right(group.element(h))
    hg_val = se2_derepresentation(hg.value)

    # PART 3 - Illustrating the relative positions
    # Compute g relative to h: h_inv * g
    h_inv = group.element(h).inverse_element()
    g_relative_to_h = h_inv.left(group.element(g))
    g_relative_to_h_val = se2_derepresentation(g_relative_to_h.value)

    # Compute h relative to g: g_inv * h
    g_inv = group.element(g).inverse_element()
    h_relative_to_g = g_inv.left(group.element(h))
    h_relative_to_g_val = se2_derepresentation(h_relative_to_g.value)

    if printing:
        print('PART 1 OUTPUT:')
        print(f'g = {g_val}')
        print(f'h = {h_val}')
        print(f'gh = {gh_val}')
        print(f'hg = {hg_val}')
        print(f'g relative to h = {g_relative_to_h_val}')
        print(f'h relative to g = {h_relative_to_g_val}')
        print('\n')

    # Plot the relative positions
    se2_plot(np.stack([g_val, h_val, gh_val, hg_val, g_relative_to_h_val, h_relative_to_g_val]), 
             labels=['g', 'h', 'gh', 'hg', 'g relative to h', 'h relative to g'], 
             xlim=(-2, 3), ylim=(-2, 3), 
             title="Relative Positions of g and h")
    return np.array([g_val, h_val, gh_val, hg_val, g_relative_to_h_val, h_relative_to_g_val])

def part2(printing=True):
    def transform(input1, input2):
        top_left = input1[:2, :2] @ input2[:2, :2]
        top_right = input1[:2, :2] @ input2[:2, 2] + input1[:2, 2]

        output = np.eye(3)
        output[:2, :2] = top_left
        output[:2, 2] = top_right
        return output
    
    def rep(x):
        # Returns a 3x3 transformation matrix based on the input parameters
        return np.array([[np.cos(x[2]), -np.sin(x[2]), x[0]],
                         [np.sin(x[2]),  np.cos(x[2]), x[1]],
                         [0,             0,              1]])

    def derep(x):
        return np.array([x[0, 2], x[1, 2], np.arctan2(x[1, 0], x[0, 0])])  # Using atan2 for angle 

    g_val = np.array([0, 1, -np.pi/4])
    h_val = np.array([1, 2, -np.pi/2])
    identity = np.eye(3)

    # Create a representation group
    rep_group = RepresentationGroup(rep, identity=identity, derepresentation_function=derep)
    rep_group.operation = transform

    # Create representation group elements
    g = RepresentationGroupElement(rep_group, g_val)
    h = RepresentationGroupElement(rep_group, h_val)

    gh = g.left(h)
    gh_val = gh.derepresentation
    hg = g.right(h)
    hg_val = hg.derepresentation

    # PART 3 - Illustrating the relative positions
    # Compute g relative to h: h_inv * g
    h_inv = h.inverse_element()
    g_relative_to_h = h_inv.left(g)
    g_relative_to_h_val = g_relative_to_h.derepresentation

    # Compute h relative to g: g_inv * h
    g_inv = g.inverse_element()
    h_relative_to_g = g_inv.left(h)
    h_relative_to_g_val = h_relative_to_g.derepresentation

    if printing:
        print('PART 2 OUTPUT:')
        print(f'g = {g_val}')
        print(f'h = {h_val}')
        print(f'gh = {gh_val}')
        print(f'hg = {hg_val}')
        print(f'g relative to h = {g_relative_to_h_val}')
        print(f'h relative to g = {h_relative_to_g_val}')
        print('\n')

    # Plot the relative positions
    se2_plot(np.stack([g_val, h_val, gh_val, hg_val, g_relative_to_h_val, h_relative_to_g_val]), 
             labels=['g', 'h', 'gh', 'hg', 'g relative to h', 'h relative to g'], 
             xlim=(-2, 3), ylim=(-2, 3), 
             title="Relative Positions of g and h")

    return np.array([g_val, h_val, gh_val, hg_val, g_relative_to_h_val, h_relative_to_g_val])

def part3(printing=True):  
    def rep(x):
        # Returns a 3x3 transformation matrix based on the input parameters
        return np.array([[np.cos(x[2]), -np.sin(x[2]), x[0]],
                         [np.sin(x[2]),  np.cos(x[2]), x[1]],
                         [0,             0,              1]])
    def derep(x):
        return np.array([x[0, 2], x[1, 2], np.arctan2(x[1, 0], x[0, 0])])  # Using atan2 for angle

    # Create a representation group
    rep_group = RepresentationGroup(rep, identity=np.eye(3), derepresentation_function=derep)

    # PART 3 - DELIVERABLE 1
    g1_val = np.array([0, 1, -np.pi / 4]) 
    g2_val = np.array([1, 2, -np.pi / 2]) 

    # Create representation group elements
    g1 = RepresentationGroupElement(rep_group, g1_val)
    g2 = RepresentationGroupElement(rep_group, g2_val)

    # Compute the relative position h21 of g2 with respect to g1
    h21 = g1.inverse_element().left(g2)

    # Compute the adjoint at g1 of this relative position
    h12 = g1.AD(h21)

    # Demonstrate that the left action of the adjointed relative position on g1 brings it to g2
    rslt = g1.right(h12)

    # PART 3 - DELIVERABLE 2
    h1_val = np.array([-1, 0, np.pi/2])
    h1 = RepresentationGroupElement(rep_group, h1_val)

    # Move g1 by h1
    g1_prime = g1.left(h1)

    # Move g2 by h2 = ADh_inv(h12)h1
    h2 = h21.AD_inv(h1)
    g2_prime = g2.left(h2)

    # Illustrate that g1_prime and g2_prime preserves the relative displacement between the two elements 
    rel_pos1 = g1.inverse_element().left(g2)
    rel_pos2 = g1_prime.inverse_element().left(g2_prime)

    try:
        if not np.linalg.norm(g2_val - rslt.derepresentation) <= 0.0000001:
            raise ValueError("g2 does not equal h_12*g1 !")
        print("Part 3.1 test passed! g2 equals h_12*g1 !")
    except ValueError as e:
        print(e)

    try:
        if not np.linalg.norm(rel_pos1.derepresentation - rel_pos2.derepresentation) <= 0.0000001:
            raise ValueError("g1 * g2 does not equal g1' * g2'!")
        print("Part 3.2 test passed! g1 * g2 equals g1' * g2'!")
    except ValueError as e:
        print(e)

    if printing:
        print('PART 3 OUTPUT:')
        print('These values should be equal:')
        print(f'g2 = {g2_val}')
        print(f'h_12 * g1 = {rslt.derepresentation}')
        print('These values should also be equal:')
        print(f'g1 * g2 = {np.round(rel_pos1.derepresentation, 5)}')
        print(f'g1` * g2` = {np.round(rel_pos2.derepresentation, 5)}')


if __name__ == '__main__':
    pt1_output = part1(printing=False)
    pt2_output = part2(printing=False)

    try:
        if not np.array_equal(pt1_output, pt2_output):
            raise ValueError("Output from part 1 does not match part 2 output")
        print("Part 1-2 test passed! Part 1 outputs match part 2 outputs!")
    except ValueError as e:
        print(e)

    part3(printing=False)