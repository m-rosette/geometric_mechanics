#! /usr/bin/python3
import sys
sys.path.append('../')
from geomotion import (
    utilityfunctions as ut,
    rigidbody as rb)
from hw3 import simplekinematicchain as kc
import numpy as np
from matplotlib import pyplot as plt

# Set the group as SE2 from rigidbody
G = rb.SE2


class DiffKinematicChain(kc.KinematicChain):

    def __init__(self,
                 links,
                 joint_axes):

        """Initialize a kinematic chain augmented with Jacobian methods"""

        # Call the constructor for the base KinematicChain class

        # Create placeholders for the last-calculated Jacobian value and the index it was requested for


    def Jacobian_Ad_inv(self,
                        link_index,  # Link number (with 1 as the first link)
                        output_frame='body'):  # options are world, body, spatial

        """Calculate the Jacobian by using the Adjoint_inverse to transfer velocities from the joints to the links"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints

        ########
        # Populate the Jacobian matrix by finding the transform from each joint before the chosen link to the
        # end of the link, and using its Adjoint-inverse to transform the joint axis to the body frame of
        # the selected link, and then transform this velocity to the world, or spatial coordinates if requested

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):

            # create a transform g_rel that describes the position of the selected link relative the jth joint (which
            # is at the (j-1)th location in link_positions_with_base

            # use the Adjoint-inverse of this relative transformation to map the jth joint axis ( (j-1)th entry)
            # out to the end of the selected link in the link's body frame

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates

            # If the output_frame input is 'spatial', use the adjoint of the link position to map the axis back to
            # the identity

            # Insert the value of J_joint into the (j-1)th index of J

            # Store J and the last requested index

        return J

    def Jacobian_Ad(self,
                    link_index,  # Link number (with 1 as the first link)
                    output_frame='body'):  # options are world, body, spatial

        """Calculate the Jacobian by using the Adjoint to transfer velocities from the joints to the origin"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints

        ########
        # Populate the Jacobian matrix by finding the position of each joint in the world (which is the same as the
        # position of the previous link), and using its Adjoint to send the axis into spatial coordinates

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):

            # use the Adjoint of the position of this joint to map its joint axis ( (j-1)th entry)
            # back to the identity of the group

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates

            # If the output_frame input is 'body', use the adjoint-inverse of the link position to map the axis back to
            # the identity

            # Insert the value of J_joint into the (j-1)th index of J

            # Store J and the last requested index


        return J

    def draw_Jacobian(self,
                      ax):

        """ Draw the components of the last-requested Jacobian"""

        # Get the location of the last-requested link, and use ut.column to make into a numpy column array

        # Use np.tile to make a matrix in which each column is the coordinates of the link end

        # Use ax.quiver to plot a set of arrows at the selected link end, (remembering to use only the xy components
        # and not the theta component)


if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc = DiffKinematicChain(links, joint_axes)

    # Set the joint angles to pi/4, -pi/2 and 3*pi/4
    kc.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi])

    # Create a plotting axis
    ax = plt.subplot(1, 1, 1)

    J_Ad_inv = kc.Jacobian_Ad_inv(3, 'world')
    print(J_Ad_inv)

    J_Ad = kc.Jacobian_Ad(3, 'world')
    print(J_Ad_inv)

    # Draw the chain
    kc.draw(ax)

    kc.draw_Jacobian(ax)

    # Tell pyplot to draw
    plt.show()
