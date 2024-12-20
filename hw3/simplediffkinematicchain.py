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
        super().__init__(links, joint_axes) 

        # Create placeholders for the last-calculated Jacobian value and the index it was requested for
        self.last_jacobian = None
        self.jacobian_idx = -1 # starting with an invalid index as this is just a placeholder

    def Jacobian_Ad_inv(self,
                        link_index,  # Link number (with 1 as the first link)
                        output_frame='body'):  # options are world, body, spatial

        """Calculate the Jacobian by using the Adjoint_inverse to transfer velocities from the joints to the links"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints
        num_rows = G.element_shape[0]
        num_columns = len(self.joint_angles)
        J = np.zeros((num_rows, num_columns))

        ########
        # Populate the Jacobian matrix by finding the transform from each joint before the chosen link to the
        # end of the link, and using its Adjoint-inverse to transform the joint axis to the body frame of
        # the selected link, and then transform this velocity to the world, or spatial coordinates if requested

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry
        link_positions_with_base = [G.identity_element()] + self.link_positions

        # Making an instance of the desired link
        selected_link = self.link_positions[link_index - 1]

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):
            # Extract the current link position and joint axis
            link_position = link_positions_with_base[j - 1]
            joint_axis = self.joint_axes[j - 1]

            # create a transform g_rel that describes the position of the selected link relative the jth joint (which
            # is at the (j-1)th location in link_positions_with_base
            g_rel = link_position.inverse * selected_link

            # use the Adjoint-inverse of this relative transformation to map the jth joint axis ( (j-1)th entry)
            # out to the end of the selected link in the link's body frame
            J_joint = g_rel.Ad_inv(joint_axis)

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates
            if output_frame == 'world':
                J_joint = selected_link.TL(J_joint)

            elif output_frame == 'body':
                # Return the initial adjoint inverse calculation
                pass

            # If the output_frame input is 'spatial', use the adjoint of the link position to map the axis back to
            # the identity
            elif output_frame == 'spatial':
                # Use the adjoint of the link position to map the axis back to the identity
                J_joint = selected_link.Ad(J_joint)
            
            else:
                raise ValueError(f"output_frame supports, 'body', 'spatial' or 'world'. Your input: {output_frame}")

            # Insert the value of J_joint into the (j-1)th index of J
            J[:, j - 1] = J_joint.value[:num_rows]

            # Store J and the last requested index
            self.last_jacobian = J.copy()
            self.jacobian_idx = link_index

        return J

    def Jacobian_Ad(self,
                    link_index,  # Link number (with 1 as the first link)
                    output_frame='body'):  # options are world, body, spatial

        """Calculate the Jacobian by using the Adjoint to transfer velocities from the joints to the origin"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints
        num_rows = G.element_shape[0]
        num_columns = len(self.joint_angles)
        J = np.zeros((num_rows, num_columns))

        ########
        # Populate the Jacobian matrix by finding the position of each joint in the world (which is the same as the
        # position of the previous link), and using its Adjoint to send the axis into spatial coordinates

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry
        link_positions_with_base = [G.identity_element()] + self.link_positions

        # Making an instance of the desired link
        selected_link = self.link_positions[link_index - 1]

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):
            # Extract the current link position and joint axis
            link_position = link_positions_with_base[j - 1]
            joint_axis = self.joint_axes[j - 1]

            # use the Adjoint of the position of this joint to map its joint axis ( (j-1)th entry)
            # back to the identity of the group
            J_joint = link_position.Ad(joint_axis)

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates
            if output_frame == 'world':
                J_joint = selected_link.TR(J_joint)

            # If the output_frame input is 'body', use the adjoint-inverse of the link position to map the axis back to
            # the identity
            elif output_frame == 'body':
                J_joint = selected_link.Ad_inv(J_joint)

            elif output_frame == 'spatial':
                # Return the initial adjoint calculation
                pass

            else:
                raise ValueError(f"output_frame supports, 'body', 'spatial' or 'world'. Your input: {output_frame}")

            # Insert the value of J_joint into the (j-1)th index of J
            J[:, j - 1] = J_joint.value[:num_rows]

            # Store J and the last requested index
            self.last_jacobian = J.copy()
            self.jacobian_idx = link_index

        return J

    def draw_Jacobian(self, ax, arrow_scale=2):
        """ Draw the kinematic chain links with Jacobian vectors at the end of each link"""

        # Ensure that there is a last calculated Jacobian to draw
        if self.last_jacobian is None or self.jacobian_idx < 1:
            print("No Jacobian calculated. Please calculate the Jacobian first.")
            return

        # Plot Jacobian vectors at the end of each link
        for i, link_position in enumerate(self.link_positions):
            # Get the current link's position in xy (only the translational part)
            link_end = link_position.value[:2]
            
            # Extract the Jacobian components corresponding to this link (only the x and y components)
            jacobian_components = self.last_jacobian[:2, i]

            # Use ax.quiver to plot the Jacobian vector at the end of the link
            ax.quiver(link_end[0], link_end[1],  # Starting point (x, y)
                    jacobian_components[0], jacobian_components[1],  # Direction (vx, vy)
                    angles='xy', scale_units='xy', scale=arrow_scale, color='r')

        # Set limits and labels for better visualization
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Kinematic Chain with Jacobian Vectors')

        # Set the plot aspect ratio to 'equal'
        ax.set_aspect('equal')
        ax.grid()

def single_arm_test(links, joint_axes):
    # Create a kinematic chain
    kc = DiffKinematicChain(links, joint_axes)

    # Set the joint angles to pi/4, -pi/2 and 3*pi/4
    kc.set_configuration([.25 * np.pi, -.5 * np.pi, .75 * np.pi])

    # Create a plotting axis
    ax = plt.subplot(1, 1, 1)

    J_Ad_inv = kc.Jacobian_Ad_inv(3, 'world')
    print(J_Ad_inv)

    J_Ad = kc.Jacobian_Ad(3, 'world')
    print(J_Ad)

    # Draw the chain
    kc.draw(ax)

    kc.draw_Jacobian(ax)

    # Tell pyplot to draw
    plt.show()

def hw3_deliverables(num_plots, joint_angles, links, joint_axes):
    # Create a figure and an array of axes for the subplots
    fig, ax = plt.subplots(1, num_plots, figsize=(15, 5))  # Use plt.subplots here

    for i in range(num_plots):
        kc = DiffKinematicChain(links, joint_axes)
        kc.set_configuration(joint_angles[i])
        
        J_Ad_inv = kc.Jacobian_Ad_inv(3, 'world')
        # print(J_Ad_inv)

        J_Ad = kc.Jacobian_Ad(3, 'world')
        # # print(J_Ad)

        kc.draw(ax[i])          # Draw the kinematic chain on the ith axis
        kc.draw_Jacobian(ax[i]) # Draw the Jacobian on the ith axis

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes = [G.Lie_alg_vector([0, 0, 1])] * 3

    # single_arm_test(links, joint_axes)

    # Stage the triple subplots
    joint_angles = np.array([[.25 * np.pi, -.5 * np.pi, .75 * np.pi],
                             [-.5 * np.pi, -.75 * np.pi, .25 * np.pi],
                             [.75 * np.pi, .25 * np.pi, -.75 * np.pi]])
    
    hw3_deliverables(3, joint_angles, links, joint_axes)


