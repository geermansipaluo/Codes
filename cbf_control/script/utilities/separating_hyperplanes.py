import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import mosek
import src.cbf_control.script.utilities.mosek_ldp as mldp

def separating_hyperplanes(obstacles, C, d):
    """
    This function takes in a list of obstacles and a constraint set C and d and returns True if the constraint set is separable, and False otherwise.
    """

    dim = C.shape[0]
    infeas_start = False
    n_obs = obstacles.shape[0]
    pts_per_obs = obstacles.shape[2]
    Cinv = inv(C)
    Cinv2 = Cinv.dot(Cinv)
    if n_obs == 0 or isempty(C):
        A = np.zeros((0, dim))
        b = np.zeros((0, 1))
        infeas_start = False
        return  A, b, infeas_start

    uncover_obstacles = np.ones((n_obs, 1))
    planes_to_use = np.zeros((n_obs, 1))

    # find the closest point to the ellipse
    # d_tile = np.tile(d, (7, 1, pts_per_obs)).reshape(7,2,pts_per_obs)
    image_pts = np.zeros((n_obs, 2, pts_per_obs))
    for i in range(n_obs):
        for j in range(pts_per_obs):
            image_pts[i, :, j] = Cinv.dot(obstacles[i, :, j] - d)

    # translate obstacle's dimensions from n_obs*2*pts_per_obs to 2,n_obs*pts_per_obs
    # temp = obstacles[0, :, :]
    # for i in range(1, n_obs):
    #     temp = np.concatenate((temp, obstacles[i, :, :]), axis=1)
    # image_pts = (Cinv.dot(temp - d_tile)).reshape(2, n_obs, pts_per_obs)
    image_dists = np.sum(image_pts**2, axis=1).T
    obs_image_dists = np.min(image_dists, axis=0)
    obs_sort_idx = np.argsort(obs_image_dists)

    flat_obs_pts = obstacles[0, :, :]
    for i in range(1, n_obs):
        flat_obs_pts = np.concatenate((flat_obs_pts, obstacles[i, :, :]), axis=1)
    A = np.zeros((n_obs, dim))
    b = np.zeros((n_obs, 1))
    #finding the tangent plane
    for i in obs_sort_idx:
        if uncover_obstacles[i] == 1:
            obs = obstacles[i, :, :]
            ys = image_pts[i, :, :]
            dists = image_dists[:, i]
            idx = np.argmin(dists)
            xi = obs[:, idx]
            nhat = 2*Cinv2.dot(xi-d)
            nhat = nhat/np.linalg.norm(nhat)
            b0 = nhat.dot(xi)
            if all(nhat.dot(obs) - b0 >= 0):
                A[i, :] = nhat.T
                b[i] = b0
            else:
                ystar_temp = mldp.mosek_ldp(ys)
                ystar = np.zeros((2, 1))
                ystar[0] = ystar_temp[0]
                ystar[1] = ystar_temp[1]
                if np.linalg.norm(ystar) < 1e-3:
                   infeas_start = True
                   A[i, :] = -nhat.T
                   b[i] = -nhat.T.dot(xi)
                else:
                    dt = np.zeros((2, 1))
                    dt[0] = d[0]
                    dt[1] = d[1]
                    xstar = C.dot(ystar) + dt
                    nhat = 2*Cinv2.dot(xstar-dt)
                    nhat = nhat/np.linalg.norm(nhat)
                    b[i] = nhat.T.dot(xstar)
                    A[i, :] = nhat.T

            check = A[i, :].dot(flat_obs_pts) >= b[i]
            check =check.reshape(n_obs, pts_per_obs)
            check = check.T
            excluded = check.all(axis=0)
            uncover_obstacles[excluded] = False

            planes_to_use[i] = True
            uncover_obstacles[i] = False

            if ~uncover_obstacles.any():
                break

    for i in range(planes_to_use.shape[0]-1,-1,-1):
        if planes_to_use[i] == 0:
            A = np.delete(A,i,0)
            b = np.delete(b,i,0)

    return A, b, infeas_start


def isempty(A):
    """
    This function takes in a matrix A and returns True if A is empty, and False otherwise.
    """
    return (A.shape[0] == 0 or A.shape[1] == 0)