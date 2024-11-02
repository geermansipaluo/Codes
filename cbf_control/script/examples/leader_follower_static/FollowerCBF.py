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

def FollowerCBF(x, dx, waypoint, state, C, k):
    """
    This function implements the CBF controller for a follower vehicle.

    :param x: current position of the leader vehicle
    :param dx: current velocity of the leader vehicle
    :param waypoint: desired position of the follower vehicle
    :param state: current state of the  vehicle
    :param C: Safe region matrix
    :param d: Safe region center

    :return: control input for the follower vehicle
    """
    # Define the constant
    safety_radius = 0.15
    gamma1 = 1
    gamma2 = 100 / 2
    rho = 0.5
    # k's robot state
    xk = x[:, k].reshape(2, 1)
    # number of constraints
    con_flag = 0

    H = np.eye(2)
    f = -2*dx

    num_robots = 3
    num_constraints = 1 + num_robots - 1

    A = np.zeros((num_constraints, 2))
    b = np.zeros((num_constraints, 1))

    num1 = np.dot((xk - waypoint).T, inv(C[state, :, :]).T)
    num2 = np.dot(inv(C[state, :, :]), (xk - waypoint))
    h_fcbf = 1 - np.dot(num1, num2)
    h_fcbf_dot = -2*np.dot(inv(C[state, :, :]).T, np.dot(inv(C[state, :, :]), (xk - waypoint)))
    if h_fcbf.all() < 0:
        num1 = np.dot((xk - waypoint).T, inv(C[state - 1, :, :]).T)
        num2 = np.dot(inv(C[state - 1, :, :]), (xk - waypoint))
        h_fcbf1 = 1 - np.dot(num1, num2)
        h_fcbf_dot1 = -2*np.dot(inv(C[state - 1, :, :]).T, np.dot(inv(C[state - 1, :, :]), (xk - waypoint)))
        A[con_flag, :] = (-h_fcbf_dot-h_fcbf_dot1).T
        b[con_flag, 0] = gamma1*np.sign(min(h_fcbf1, h_fcbf))
        con_flag += 1
    else:
        A[con_flag, :] = -h_fcbf_dot.T
        b[con_flag, 0] = gamma1*np.sign(h_fcbf)*np.power(abs(h_fcbf), rho)
        con_flag += 1

    # for i in range(0, num_robots):
    #     if i != k:
    #         error = xk - x[:, i].reshape(2, 1)
    #         h_zcbf = (error[0] * error[0] + error[1] * error[1]) - np.power(safety_radius, 2)
    #         A[con_flag, :] = -2 * error.T
    #         b[con_flag, 0] = gamma2 * np.power(h_zcbf, 3)
    #         con_flag += 1

    # Solve the QP problem
    result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
    dx0 = np.reshape(result, (2, 1), order='F')
    return dx0






