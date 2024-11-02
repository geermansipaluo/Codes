import numpy as np
import src.cbf_control.script.utilities.inflation_results as ir
import src.cbf_control.script.utilities.separating_hyperplanes as sh
import src.cbf_control.script.utilities.maximal_ellipse as me

def inflate_region_feedback(obstacle, seed, bounds):

    results = ir.InflationResults()
    results.start = seed
    results.obstacles = obstacle
    results.bounds = bounds

    iter_limit = 100
    iter = 0

    dim = bounds.A_bounds.shape[1]
    d = results.start
    C = 1e-4*np.eye(dim)
    best_vol = -np.inf
    while True:
        [A, b, infeas_start] = sh.separating_hyperplanes(results.obstacles, C, d)
        A_bounds = bounds.A_bounds
        b_bounds = bounds.b_bounds
        A = np.vstack((A, A_bounds))
        b = np.vstack((b, b_bounds))

        [C, d, cvx_optval] = me.maximal_ellipse(A, b)

        if abs(cvx_optval - best_vol)/best_vol < 2e-2 or iter > iter_limit:
            break
        best_vol = cvx_optval
        iter += 1
    print("Inflation completed in", iter, "iterations")
    return A, b, C, d
