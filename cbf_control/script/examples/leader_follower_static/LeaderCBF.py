from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import quadprog as solver2

import itertools
import numpy as np
from scipy.special import comb
import math
from numpy.linalg import inv

from src.cbf_control.script.utilities.transformations import *

def LeaderCBF(x, waypoint, state, C):
    """
    This function implements the CBF controller for a leader vehicle.

    :param x: current position of the leader vehicle
    :param dx: current velocity of the leader vehicle
    :param waypoint: desired position of the follower vehicle
    :param state: current state of the  vehicle
    :param C: Safe region matrix
    :param d: Safe region center

    :return: control input for the follower vehicle
    """
    # Define the constant
    close_radius = 0.05
    gamma = 1

    H = np.eye(2)
    f = np.zeros((2, 1))

    num_robots = 1
    num_constraints = 1

    A = np.zeros((num_constraints, 2))
    b = np.zeros((num_constraints, 1))

    num1 = np.dot((x - waypoint).T, inv(C[state, :, :]).T)
    num2 = np.dot(inv(C[state, :, :]), (x - waypoint))
    h_fcbf = 1 - np.dot(num1, num2)
    h_fcbf_dot = -2*np.dot(inv(C[state, :, :]).T, np.dot(inv(C[state, :, :]), (x - waypoint)))
    if h_fcbf.all() < 0:
        num1 = np.dot((x - waypoint).T, inv(C[state - 1, :, :]).T)
        num2 = np.dot(inv(C[state - 1, :, :]), (x - waypoint))
        h_fcbf1 = 0.8 - np.dot(num1, num2)
        h_fcbf_dot1 = -2*np.dot(inv(C[state - 1, :, :]).T, np.dot(inv(C[state - 1, :, :]), (x - waypoint)))
        A[0, :] = (-h_fcbf_dot-h_fcbf_dot1).T
        b[0, 0] = gamma*np.sign(min(h_fcbf[0], h_fcbf1[0]))
    else:
        diff = np.dot((x - waypoint).T, (x - waypoint))
        h_fcbf2 = np.power(close_radius, 2) - diff
        h_fcbf_dot2 = -2*(x - waypoint)
        A[0, :] = (-h_fcbf_dot-h_fcbf_dot2).T
        b[0, 0] = gamma*np.sign(min(h_fcbf[0], h_fcbf2[0]))

    # Solve the QP problem
    result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
    dx0 = np.reshape(result, (2, 1), order='F')
    return dx0




