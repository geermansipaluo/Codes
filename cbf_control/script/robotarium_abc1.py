import time
import math
import src.cbf_control.script.utilities.polyhedron as polyhedron
import src.cbf_control.script.utilities.inflate_region as inflate_region
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import src.cbf_control.script.utilities.misc as misc

# RobotariumABC: This is an interface for the Robotarium class that
# ensures the simulator and the robots match up properly.  

# THIS FILE SHOULD NEVER BE MODIFIED OR SUBMITTED!

class RobotariumABC(ABC):

    def __init__(self, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([])):

        #Check user input types
        assert isinstance(number_of_robots,int), "The number of robots used argument (number_of_robots) provided to create the Robotarium object must be an integer type. Recieved type %r." % type(number_of_robots).__name__
        assert isinstance(initial_conditions,np.ndarray), "The initial conditions array argument (initial_conditions) provided to create the Robotarium object must be a numpy ndarray. Recieved type %r." % type(initial_conditions).__name__
        assert isinstance(show_figure,bool), "The display figure window argument (show_figure) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(show_figure).__name__
        assert isinstance(sim_in_real_time,bool), "The simulation running at 0.033s per loop (sim_real_time) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(show_figure).__name__
        
        #Check user input ranges/sizes
        assert (number_of_robots >= 0 and number_of_robots <= 50), "Requested %r robots to be used when creating the Robotarium object. The deployed number of robots must be between 0 and 50." % number_of_robots 
        if (initial_conditions.size > 0):
            assert initial_conditions.shape == (3, number_of_robots), "Initial conditions provided when creating the Robotarium object must of size 3xN, where N is the number of robots used. Expected a 3 x %r array but recieved a %r x %r array." % (number_of_robots, initial_conditions.shape[0], initial_conditions.shape[1])


        self.number_of_robots = number_of_robots
        self.show_figure = show_figure
        self.initial_conditions = initial_conditions

        # Boundary stuff -> lower left point / width / height
        self.boundaries = [0, 0, 2.05, 2.05]

        self.file_path = None
        self.current_file_size = 0

        # Constants
        self.time_step = 0.033
        self.robot_diameter = 0.07
        self.wheel_radius = 0.02
        self.base_length = 0.105

        # Safe regions
        self.A = []
        self.b = []
        self.C = []
        self.d = []
        self.safe_regions = []
        self.safe_region_patches = []

        # Velocity limits
        self.max_linear_velocity = 0.2
        self.max_angular_velocity = 2*(self.wheel_radius/self.robot_diameter)*(self.max_linear_velocity/self.wheel_radius)
        self.max_wheel_velocity = self.max_linear_velocity/self.wheel_radius

        self.robot_radius = self.robot_diameter/2

        # Robot state variables
        self.velocities = np.zeros((2, number_of_robots))
        self.poses = self.initial_conditions

        if self.initial_conditions.size == 0:
            self.poses = misc.generate_initial_conditions(self.number_of_robots, spacing=0.2, width=2.5, height=1.5)
        
        self.left_led_commands = []
        self.right_led_commands = []

        # Obstacle stuff(boundaries and obstacles)
        self.bounds = polyhedron.from_bounds([[0.05], [0.05]], [[2.05], [2.05]])
        obstacles = np.zeros((7, 2, 5))

        # Generate obstacles
        self.obstacles = np.zeros((7, 2, 5))
        self.obstacles_patches = []
        obs = np.array([[0.4, 0.5, 0.5, 0.4, 0.4], [0, 0, 0.3, 0.3, 0.3]])
        self.obstacles[0] = obs
        obs = np.array([[0, 0.5, 1.1, 0, 0], [0.7, 0.9, 0.6, 2.05, 2.05]])
        self.obstacles[1] = obs
        obs = np.array([[0, 0.75, 0.9, 1.2, 1.2], [2.05, 1.6, 1.8, 2.05, 2.05]])
        self.obstacles[2] = obs
        obs = np.array([[2.05, 1.6, 1.8, 2.05, 2.05], [1.6, 1.6, 1.45, 1.45, 1.45]])
        self.obstacles[3] = obs
        obs = np.array([[1.6, 1.8, 1.7, 1.2, 1.2], [1.6, 1.45, 1.2, 1.5, 1.6]])
        self.obstacles[4] = obs
        obs = np.array([[1.25, 1.5, 1.6, 2.05, 2.05], [0, 0.3, 0.5, 0, 0]])
        self.obstacles[5] = obs
        obs = np.array([[2.05, 1.6, 1.6, 2.05, 2.05], [0, 0.5, 0.7, 0.75, 0.75]])
        self.obstacles[6] = obs

        # IRIS stuff
        self.seeds = np.array([[0.1, 0.5, 1.2, 1.2, 1.1, 1.8], [0.1, 0.4, 0.4, 0.7, 1.7, 1.8]])
        self.seeds_patch = []

        # Visualization
        self.figure = []
        self.axes = []
        self.left_led_patches = []
        self.right_led_patches = []
        self.chassis_patches = []
        self.right_wheel_patches = []
        self.left_wheel_patches = []

        if(self.show_figure):
            self.figure, self.axes = plt.subplots()
            self.axes.set_axis_off()
            for i in range(number_of_robots):
                # Draw robots
                p = patches.RegularPolygon(self.poses[:2, i], 4, radius=math.sqrt(2)*self.robot_radius, orientation=self.poses[2,i]+math.pi/4, facecolor='#FFD700', edgecolor = 'k')
                rled = patches.Circle(self.poses[:2, i]+0.75*self.robot_radius*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i]))+\
                                        0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))),\
                                       self.robot_radius/5, fill=False)
                lled = patches.Circle(self.poses[:2, i]+0.75*self.robot_radius*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i]))+\
                                        0.015*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))),\
                                       self.robot_radius/5, fill=False)
                rw = patches.Circle(self.poses[:2, i]+self.robot_radius*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')
                lw = patches.Circle(self.poses[:2, i]+self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')
                #lw = patches.RegularPolygon(self.poses[:2, i]+self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                #                                0.035*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                #                                4, math.sqrt(2)*0.02, self.poses[2,i]+math.pi/4, facecolor='k')

                self.chassis_patches.append(p)
                self.left_led_patches.append(lled)
                self.right_led_patches.append(rled)
                self.right_wheel_patches.append(rw)
                self.left_wheel_patches.append(lw)
                
                self.axes.add_patch(rw)
                self.axes.add_patch(lw)
                self.axes.add_patch(p)
                self.axes.add_patch(lled)
                self.axes.add_patch(rled)

            # Draw arena
            self.boundary_patch = self.axes.add_patch(patches.Rectangle(self.boundaries[:2], self.boundaries[2], self.boundaries[3], fill=False))

            # Draw obstacles
            for i in range(obstacles.shape[0]):
                self.obstacles_patches = self.axes.add_patch(patches.Polygon(self.obstacles[i].T, fill=True, edgecolor='k'))

            # Draw seeds
            # for i in range(self.seeds.shape[1]):
            #     self.seeds_patch = self.axes.add_patch(plt.Circle(self.seeds[:, i], 0.01, fill=True, color='r'))

            self.axes.set_xlim(self.boundaries[0]-0.1, self.boundaries[0]+self.boundaries[2]+0.1)
            self.axes.set_ylim(self.boundaries[1]-0.1, self.boundaries[1]+self.boundaries[3]+0.1)
                            
            plt.ion()
            plt.show()

            plt.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=1.03, wspace=0, hspace=0)

            # Draw safe regions
            # first create a safe region and through IRIS to get the safe regions
            temp_A =[]
            temp_b = []
            temp_C = []
            temp_d = []
            for i in range(self.seeds.shape[1]):
                seed = self.seeds[:, i]
                temp_A, temp_b, temp_C, temp_d = inflate_region.inflate_region_feedback(self.obstacles, seed, self.bounds)
                self.A.append(temp_A)
                self.b.append(temp_b)
                self.C.append(temp_C)
                self.d.append(temp_d)
                self.safe_regions.append([temp_A, temp_b, temp_C, temp_d])

            # Second Draw safe regions
            th = np.linspace(0, 2*np.pi, 100)
            y = np.vstack((np.cos(th), np.sin(th)))
            for i in range(len(self.safe_regions)):
                d = self.safe_regions[i][3]
                x = np.dot(self.safe_regions[i][2], y)
                for j in range(x.shape[1]):
                    x[:, j] = x[:, j] + d
                plt.plot(x[0,:], x[1,:], color='r', linewidth=2)



    def set_velocities(self, ids, velocities):

        # Threshold linear velocities
        idxs = np.where(np.abs(velocities[0, :]) > self.max_linear_velocity)
        velocities[0, idxs] = self.max_linear_velocity*np.sign(velocities[0, idxs])

        # Threshold angular velocities
        idxs = np.where(np.abs(velocities[1, :]) > self.max_angular_velocity)
        velocities[1, idxs] = self.max_angular_velocity*np.sign(velocities[1, idxs])
        self.velocities = velocities

    @abstractmethod
    def get_poses(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    #Protected Functions
    def _threshold(self, dxu):
        dxdd = self._uni_to_diff(dxu)

        to_thresh = np.absolute(dxdd) > self.max_wheel_velocity
        dxdd[to_thresh] = self.max_wheel_velocity*np.sign(dxdd[to_thresh])

        dxu = self._diff_to_uni(dxdd)

    def _uni_to_diff(self, dxu):
        r = self.wheel_radius
        l = self.base_length
        dxdd = np.vstack((1/(2*r)*(2*dxu[0,:]-l*dxu[1,:]),1/(2*r)*(2*dxu[0,:]+l*dxu[1,:])))

        return dxdd

    def _diff_to_uni(self, dxdd):
        r = self.wheel_radius
        l = self.base_length
        dxu = np.vstack((r/(2)*(dxdd[0,:]+dxdd[1,:]),r/l*(dxdd[1,:]-dxdd[0,:])))

        return dxu

    def _validate(self, errors = {}):
        # This is meant to be called on every iteration of step.
        # Checks to make sure robots are operating within the bounds of reality.

        p = self.poses
        b = self.boundaries
        N = self.number_of_robots


        for i in range(N):
            x = p[0,i]
            y = p[1,i]

            if(x < b[0] or x > (b[0] + b[2]) or y < b[1] or y > (b[1] + b[3])):
                    if "boundary" in errors:
                        errors["boundary"] += 1
                    else:
                        errors["boundary"] = 1
                        errors["boundary_string"] = "iteration(s) robots were outside the boundaries."

        for j in range(N-1):
            for k in range(j+1,N):
                if(np.linalg.norm(p[:2,j]-p[:2,k]) <= self.robot_diameter):
                    if "collision" in errors:
                        errors["collision"] += 1
                    else:
                        errors["collision"] = 1
                        errors["collision_string"] = "iteration(s) where robots collided."

        dxdd = self._uni_to_diff(self.velocities)
        exceeding = np.absolute(dxdd) > self.max_wheel_velocity
        if(np.any(exceeding)):
            if "actuator" in errors:
                errors["actuator"] += 1
            else:
                errors["actuator"] = 1
                errors["actuator_string"] = "iteration(s) where the actuator limits were exceeded."

        return errors

