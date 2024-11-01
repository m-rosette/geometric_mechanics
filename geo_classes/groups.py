#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from geo_classes.velocities import TangentVector


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

    def left_lifted(self, other):
        """
        Left lifted action: Th Lg
        Computes the tangent vector from the left action of 'self' on 'other' 
        by numerically differentiating with respect to a small perturbation.
        """
        def left_numeric(delta):
            # Perturb 'other' by a small delta value
            perturbed_value = other.value + delta
            perturbed_element = self.group.element(perturbed_value)
            return self.left(perturbed_element).value   

        # Use nd.Jacobian to differentiate with respect to perturbation delta
        jacobian = nd.Jacobian(left_numeric)

        # Compute the Jacobian evaluated at zero perturbation (delta = 0)
        # delta = np.zeros_like(other.value)
        jacobian_at_zero = jacobian(self.value)

        value = jacobian_at_zero @ other.value
        
        return TangentVector(value=value, configuration=other.value)

    def right_lifted(self, other):
        """
        Right lifted action: Th Rg
        Computes the tangent vector from the right action of 'self' on 'other' 
        by numerically differentiating with respect to a small perturbation.
        """
        def right_numeric(delta):
            # Perturb 'other' by a small delta value
            perturbed_value = other.value + delta
            perturbed_element = self.group.element(perturbed_value)
            return self.right(perturbed_element).value

        # Use nd.Jacobian to differentiate with respect to perturbation delta
        jacobian = nd.Jacobian(right_numeric)

        # Compute the Jacobian evaluated at zero perturbation (delta = 0)
        delta = np.zeros_like(other.value)
        jacobian_at_zero = jacobian(delta)

        value = jacobian_at_zero @ other.value
        
        return TangentVector(value=value, configuration=other.value)
    
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

        # # Ensure the matrix is square and invertible
        # if self.value.shape[0] != self.value.shape[1]:
        #     raise ValueError("The matrix must be square.")
        # if np.linalg.det(self.value) == 0:
        #     raise ValueError("The matrix must be invertible.")
    
    @property
    def derepresentation(self):
        """Return the derepresentation of the group element."""
        return self.group.derepresentation(self.value)