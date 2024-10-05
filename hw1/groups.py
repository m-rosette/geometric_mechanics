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
    
    def adjoint_action(self, value):
        """Compute the adjoint action of another group element."""
        return self.value @ value @ self.inverse_element()

    def adjoint_inverse_action(self, value):
        """Compute the adjoint-inverse action of another group element."""
        return self.inverse_element() @ value @ self.value    
    
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

def inverse_test():
    """Testing inverse function against the examples from Figure 1.19
    """
    g1 = 3
    g2 = 6
    identity = 1

    def inv_func(input):
        return 1 / input

    group = Group(np.multiply, identity, inverse_operation=inv_func)

    g3_inv = group.element(g2).left(group.element(g1).inverse_element())

    print(g3_inv.value)

def inv_func(input):
    return 1 / input

def product_scale_shift():
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

def rot(theta):
     return np.array([[np.cos(theta[2]), -np.sin(theta[2])],
                     [np.sin(theta[2]), np.cos(theta[2])]])

def trans_rot_se2(group_element):
    return np.array([[np.cos(group_element[2]), -np.sin(group_element[2]), group_element[0]],
                    [np.sin(group_element[2]), np.cos(group_element[2]), group_element[1]],
                    [0, 0, 1]])

def transform(input1, input2):
    top_left = input1[:2, :2] @ input2[:2, :2]
    top_right = input1[:2, :2] @ input2[:2, 2] + input1[:2, 2]

    output = np.eye(3)
    output[:2, :2] = top_left
    output[:2, 2] = top_right
    return output

def extract_se2(trans_matrix):
    theta = np.arccos(trans_matrix[0, 0])
    x = trans_matrix[0, 2]
    y = trans_matrix[1, 2]
    return np.array([x, y, theta])

def se2_plot(data, labels, xlim=(0, 3), ylim=(0, 3), title=None):
    # Define primary colors (RGB)
    primary_colors = ['red', 'green', 'blue', 'magenta', 'black', 'orange', 'yellow', 'cyan']

    # Ensure there are enough colors for the number of arrows
    num_arrows = data.shape[0]
    colors = primary_colors[:num_arrows] # Select the first `num_arrows` primary colors

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

def se2_example():

    g = np.array([0, 1, -np.pi/4])
    h = np.array([1, 2, -np.pi/2])

    g_trans_rot = trans_rot_se2(g)
    h_trans_rot = trans_rot_se2(h)

    identity = np.eye(3)

    group = Group(transform, identity)

    gh_trans_rot = group.element(g_trans_rot).left(group.element(h_trans_rot))
    gh = extract_se2(gh_trans_rot.value)
    hg_trans_rot = group.element(g_trans_rot).right(group.element(h_trans_rot))
    hg = extract_se2(hg_trans_rot.value)

    se2_plot(np.stack([g, h, gh, hg]), labels=['g', 'h', 'gh', 'hg'], xlim=(-1, 3), ylim=(-1, 3))

def represenatation_test():
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta),  np.cos(theta)]])
    
    def scalar_to_square_mat(scalar):
        return np.array([[1, scalar],
                         [0, 1]])
    
    def coord_to_mat(coord):
        return np.array([[1, 0, coord[0]],
                         [0, 1, coord[1]],
                         [0, 0, 1]])
    
    def mat_to_coord(matrix):
        return [matrix[0, 2], matrix[1, 2]]
    
    def coord_add(input1, input2):
        return np.array([[1, 0, input1[0, 2] + input2[0, 2]],
                         [0, 1, input1[1, 2] + input2[1, 2]],
                         [0, 0, 1]])
    
    def rep(x):
        # Returns a 3x3 transformation matrix based on the input parameters
        return np.array([[np.cos(x[2]), -np.sin(x[2]), x[0]],
                         [np.sin(x[2]),  np.cos(x[2]), x[1]],
                         [0,             0,              1]])
    def derep(x):
        return np.array([x[0, 2], x[1, 2], np.arctan2(x[1, 0], x[0, 0])])  # Using atan2 for angle

    # Create a representation group for 2D rotations
    rep_group = RepresentationGroup(rep, identity=np.eye(3), derepresentation_function=derep)  # identity = rotation by 0 (theta = 0)

    g1_val = np.array([0, 1, -np.pi / 4]) 
    g2_val = np.array([1, 2, -np.pi / 2]) 

    # Example 4: Pass a single scalar value to generate the matrix
    g1 = RepresentationGroupElement(rep_group, g1_val)
    g2 = RepresentationGroupElement(rep_group, g2_val)

    # Combine g1 and g2
    h21 = g1.inverse_element().left(g2)

    # TODO fix adjoint operation - there is an issue with the matrix multiplication

    h12 = g1.adjoint_action(h21)


    rslt = g1.left(h12)



    print('These values should be equal:')
    print(f'g2 = {g2_val}')
    print(f'h_12 * g1 = {np.round(rslt.derepresentation, 3)}')

if __name__ == '__main__':
    # product_scale_shift()
    # se2_example()

    represenatation_test()