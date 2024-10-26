#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/marcus/classes/rob541/geometric_mechanics/")
from geo_classes.groups import Group, GroupElement, RepresentationGroup, RepresentationGroupElement

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
             title="Part 1: Relative Positions of g and h")
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
             title="Part 2: Relative Positions of g and h")

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

    # Plot the positions
    se2_plot(np.stack([g1_val, g2_val, rel_pos1.derepresentation, rel_pos2.derepresentation]), 
             labels=['g1', 'g2', "g1's relative pos", "g2's relative pos"], 
             xlim=(-2, 3), ylim=(-2, 3), 
             title="Part 3: Relative Displacements of g1's and g2's relative positions")

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
    pt1_output = part1(printing=True)
    # pt2_output = part2(printing=False)

    # try:
    #     if not np.array_equal(pt1_output, pt2_output):
    #         raise ValueError("Output from part 1 does not match part 2 output")
    #     print("Part 1-2 test passed! Part 1 outputs match part 2 outputs!")
    # except ValueError as e:
    #     print(e)

    # part3(printing=True)