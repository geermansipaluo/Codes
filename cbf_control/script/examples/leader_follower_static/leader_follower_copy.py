'''
TODO: UPDATE DESCRIPTION

 Sean Wilson
 10/2019
'''
import math

# !/usr/bin/env python3
# Import Robotarium Utilities
import src.cbf_control.script.robotarium as robotarium
from src.cbf_control.script.utilities.graph import *
from src.cbf_control.script.utilities.barrier_certificates import *
from src.cbf_control.script.utilities.misc import *
from src.cbf_control.script.utilities.controllers import *
import src.cbf_control.script.examples.leader_follower_static.LeaderCBF as LCBF
import src.cbf_control.script.examples.leader_follower_static.FollowerCBF as FCBF
from src.cbf_control.script.utilities.transformations import *
import src.cbf_control.src.pub_vel as pub
import src.cbf_control.src.pub1_vel as pub1
import src.cbf_control.src.pub2_vel as pub2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as R

# Other Imports
import numpy as np
from numpy.linalg import inv
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

# Experiment Constants
iterations = 5000  # Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
N = 3  # Number of robots to use, this must stay 4 unless the Laplacian is changed.

close_enough = 0.1;  # How close the leader must get to the waypoint to move to the next one.

# Create the desired Laplacian
# followers = -completeGL(N-1)
# L = np.zeros((N,N))
# L[1:N,1:N] = followers
# L[1,1] = L[1,1] + 1
# L[1,0] = -1
L = np.array([[0, 0, 0], [-1, 2, -1], [-1, -1, 2]])

# Find connections
# [rows,cols] = np.where(L==1)

# For computational/memory reasons, initialize the velocity vector
dxi = np.zeros((2, N))

# Initialize leader state
state = 0

# Limit maximum linear speed of any robot
magnitude_limit = 1

# Create gains for our formation control algorithm
formation_control_gain = 10
desired_distance = 0.14

# Initial Conditions to Avoid Barrier Use in the Beginning.
initial_conditions = np.array([[0.2, 0.2, 0.2], [0.3, 0.17, 0.06], [1.571, 1.571, 1.571]])

# Instantiate the Robotarium object with these parameters
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,
                          sim_in_real_time=True)

# Grab Robotarium tools to do simgle-integrator to unicycle conversions and collision avoidance
# Single-integrator -> unicycle dynamics mapping
si_to_uni_dyn = create_si_to_uni_mapping()
# Single-integrator barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate()

# Get the waypoints
waypoints = np.zeros((2, len(r.d)))
for i in range(len(r.d)):
    waypoints[:, i] = r.d[i]

# Get the safe regions
C = np.zeros((len(r.d), 2, 2))
for i in range(len(r.d)):
    C[i, :, :] = r.C[i]

# state for leader
state = 0

flag = 0

# set the state limit
state_limit = len(waypoints[1]) - 1

pubv = rospy.Publisher('/epuck_robot_0/mobile_base/cmd_vel', Twist, queue_size=10)
pubv1 = rospy.Publisher('/epuck_robot_1/mobile_base/cmd_vel', Twist, queue_size=10)
pubv2 = rospy.Publisher('/epuck_robot_2/mobile_base/cmd_vel', Twist, queue_size=10)


def callback(msg1, msg2, msg3):
    global state
    global dxi
    # Get the most recent pose information from the Robotarium. The time delay is
    # approximately 0.033s
    # x = r.get_poses()
    x = np.zeros((3, N))
    x[:, 0] = [
        msg1.pose.pose.position.x,
        msg1.pose.pose.position.y,

            -R.from_quat([msg1.pose.pose.orientation.x, msg1.pose.pose.orientation.y, msg1.pose.pose.orientation.z,
                         msg1.pose.pose.orientation.w]).as_euler('xyz')[2]
    ]
    x[:, 1] = [msg2.pose.pose.position.x,
               msg2.pose.pose.position.y,
               -R.from_quat(
                   [msg2.pose.pose.orientation.x, msg2.pose.pose.orientation.y, msg2.pose.pose.orientation.z,
                    msg2.pose.pose.orientation.w]).as_euler('xyz')[2]
               ]

    x[:, 2] = [msg3.pose.pose.position.x,
               msg3.pose.pose.position.y,
               -R.from_quat(
                   [msg3.pose.pose.orientation.x, msg3.pose.pose.orientation.y, msg3.pose.pose.orientation.z,
                    msg3.pose.pose.orientation.w]).as_euler('xyz')[2]
               ]
    print("x1,x2,x3", x[:, 0], x[:, 1], x[:, 2])
    print("-----------")


    # store all agents positions and CBF values in a matrix
    # x_all = np.zeros((2, iterations, N))
    # h_all = np.zeros((iterations, N))
    # if state != 6:
    #	for i in range(N):
    #		x_all[:, t, i] = x[:2,i]
    #		num1 = np.dot((x[:2,i] - waypoints[:,state]).T, inv(C[state, :, :]).T)
    #		num2 = np.dot(inv(C[state, :, :]), (x[:2,i] - waypoints[:,state]))
    #		h_all[t, i] = 1 - np.dot(num1, num2)
    # else:
    #	break

    # get state waypoint
    waypoint = waypoints[:, state].reshape((2, 1))

    # Controller for system formation control
    # Leader
    dxi[:, [0]] = LCBF.LeaderCBF(x[:2, 0].reshape((2, 1)), waypoint, state, C)

    # Followers
    for i in range(1, N):
        # Zero velocities and get the topological neighbors of agent i
        dxi[:, [i]] = np.zeros((2, 1))
        neighbors = topological_neighbors(L, i)

        for j in neighbors:
            dxi[:, [i]] += formation_control_gain * (
                    np.power(np.linalg.norm(x[:2, [j]] - x[:2, [i]]), 2) - np.power(desired_distance, 2)) * (
                                   x[:2, [j]] - x[:2, [i]])

        dxi[:, [i]] = FCBF.FollowerCBF(x[:2, :], dxi[:, [i]], waypoint, state, C, i)

    if np.linalg.norm(x[:2, [0]] - waypoint) <= close_enough:
        state = (state + 1)
    print("waypoint", waypoint)
    # Keep single integrator control vectors under specified magnitude
    # Threshold control inputs
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]

    # Use barriers and convert single-integrator sto unicycle commands
    dxi = si_barrier_cert(dxi, x[:2, :])

    dxu = si_to_uni_dyn(dxi, x)
    # print("-------------------------------")
    # print("dxu: ", dxu)

    # Set the velocities of agents 1,...,N to dxu
    # r.set_velocities(np.arange(N), dxu)

    # Threshold linear velocities
    max_linear_velocity = r.max_wheel_velocity
    max_angular_velocity = r.max_angular_velocity
    idxs_to_threshold = np.where(np.abs(dxu[0, :]) > max_linear_velocity)
    dxu[0, idxs_to_threshold] = np.sign(dxu[0, idxs_to_threshold]) * max_linear_velocity

    # Threshold angular velocities
    idxs_to_threshold = np.where(np.abs(dxu[1, :]) > max_angular_velocity)
    dxu[1, idxs_to_threshold] = np.sign(dxu[1, idxs_to_threshold]) * max_angular_velocity

    # publish velocity
    pub.pub_vel(pubv, dxu[:, 0])
    pub1.pub_vel(pubv1, dxu[:, 1])
    pub2.pub_vel(pubv2, dxu[:, 2])

    time.sleep(0.033)

if __name__ == '__main__':
    rospy.init_node('pub_vel', anonymous=True)

    x1_sub = Subscriber('/vicon1/epuck1/odom', Odometry, queue_size=10)
    x2_sub = Subscriber('/vicon2/epuck2/odom', Odometry, queue_size=10)
    x3_sub = Subscriber('/vicon3/epuck3/odom', Odometry, queue_size=10)
    sync = ApproximateTimeSynchronizer([x1_sub, x2_sub, x3_sub], queue_size=10, slop=0.1)
    sync.registerCallback(callback)
    rospy.spin()

# Iterate the simulation
# r.step()

# Call at end of script to print debug information and for your script to run on the Robotarium server properly
# r.call_at_scripts_end()
