#!/usr/bin/env python3

import numpy as np
import numdifftools as nd

from geo_classes.groups import Group, GroupElement, RepresentationGroup, RepresentationGroupElement

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
    def __init__(self, value, configuration):
        # Check if the configuration is a GroupElement
        if not isinstance(configuration, GroupElement):
            raise TypeError("Configuration must be a GroupElement.")
        
        # Call the parent class constructor
        super().__init__(value, configuration)
        
        # Copy the Group property from the GroupElement configuration
        self.group = configuration.group

class LieGroupElement(GroupElement):
    # (T_hL_g): g_dot -> J_Lg(h)g_dot
    # (T_hR_g): g_dot -> J_Rg(h)g_dot

    pass

class RepLieGroupElement:
    # Lifted actions
    # (T_hL_g): g_dot -> self.rep @ g_dot.rep
    # (T_hR_g): g_dot -> g_dot.rep @ self.rep

    pass

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