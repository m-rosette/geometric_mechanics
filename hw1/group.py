#!/usr/bin/env python3

import numpy as np


class Group:
    def __init__(self, operation, identity: np.array):
        """Class to construct groups 

        Args:
            operation (function): A function that takes in two inputs of the same size and returns an output also of the same size.
            identity (np.array): A numeric value that, when passed as one of the input values to the Operation function, results 
            in the output of Operation being equal to the other input passed to the function
        """
        self.operation = operation
        self.identity = identity

    def element(self, value:np.array):
        """Takes in a numeric value and return an instance of a “group element” class

        Args:
            value (np.array): _description_

        Returns:
            GroupElement: a group element
        """
        return GroupElement(self, value)

    def identity_element(self):
        """A function that takes no input, and returns the result of the “Element” function with the group's self.Identity as the input.
        """
        self.element(self.identity)


class GroupElement:
    def __init__(self, group: Group, value: np.array, inverse_function=None):
        """_summary_

        Args:
            group (Group): An instance of the Group class
            value (np.array): _description_
            inverse (function, optional): _description_. Defaults to None.
        """
        self.group = group
        self.value = value
        self.inverse_function = inverse_function

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

        return GroupElement(self.group, output)
    
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

        return GroupElement(self.group, output)
    
    def inverse_element(self):
        """A method that passes self.value to self.inverse_function, and then generates and returns a new group element constructed from 
        the inverted value

        Returns:
            GroupElement: an instance of the GroupElement class
        """
        return GroupElement(self.group, self.inverse_function(self.value))


def test():
    g1 = 2
    g2 = 3
    # g1 = np.array(([1, 2], [3, 4]))
    # g2 = np.array(([5, 6], [7, 8]))

    # print(np.matmul(g2, g1))

    group = Group(np.add, g1)

    g3 = group.element(g1).right(group.element(g2))

    print(g3.value)


if __name__ == '__main__':
    test()
