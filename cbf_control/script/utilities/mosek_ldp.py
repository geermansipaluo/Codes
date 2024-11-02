import mosek
import numpy as np
import sys
import os
import math

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


# Define a function to solve the LDP problem using MOSEK
def mosek_ldp(ys):
    dim = ys.shape[0]
    nw = ys.shape[1]
    nvar = 1 + dim + nw
    # Since the value of infinity is ignored, we define it solely
    # for symbolic purposes
    inf = 0.0

    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)
        # Create a task
        with env.Task() as task:
            # Attach a log stream printer to the task
            task.set_Stream(mosek.streamtype.log, streamprinter)
            # Bound keys for constraints
            bkc = [mosek.boundkey.fx]*(dim+1)

            # bound values for constraints
            blc = np.append(np.zeros(dim), 1)
            buc = np.append(np.zeros(dim), 1)

            # Bound keys for variables
            bkx = [mosek.boundkey.fr,
                   mosek.boundkey.fr,
                   mosek.boundkey.lo,
                   mosek.boundkey.lo,
                   mosek.boundkey.lo,
                   mosek.boundkey.lo,
                   mosek.boundkey.lo,
                   mosek.boundkey.lo]

            #bound values for variables
            blx = np.append(-inf*np.ones(dim), np.zeros(nw+1))
            bux = inf*np.ones(nvar)

            # Below is the sparse representation of the A
            # matrix stored by column.
            a1 = -np.eye(dim)
            a = np.append(a1, ys, axis=1)
            a = np.append(a, np.zeros((dim, 1)), axis=1)
            a2 = np.zeros((1, dim))
            a2 = np.append(a2, np.ones((1, nw)), axis=1)
            a2 = np.append(a2, np.zeros((1, 1)), axis=1)
            a = np.append(a, a2, axis=0)

            asub = []
            aval = []
            idxi = 0
            idxj = 0
            temp_asub = []
            temp_aval = []
            for i in range(a.shape[1]):
                for j in range(a.shape[0]):
                    if a[j,i] != 0:
                        temp_asub.append(j)
                        temp_aval.append(a[j,i])
                        idxj += 1
                asub.append(temp_asub)
                aval.append(temp_aval)
                temp_asub = []
                temp_aval = []
                idxi += 1
                idxj = 0

            numcon = len(bkc)
            numvar = len(bkx)

            # Append the objective coefficients
            c = np.append(np.zeros(dim+nw), 1)

            # Append 'numcon' empty constraints.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            task.appendvars(numvar)

            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])
                # Set the bounds on variable j
                task.putvarbound(j, bkx[j], blx[j], bux[j])

            for j in range(len(aval)):
                # Input column j of A
                task.putacol(j, asub[j], aval[j])

            for i in range(numcon):
                # Set the bounds on constraint i
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # Set up and input quadratic objective
            # qsubi = [0, 1, 2, 3, 4, 5, 6, 7]
            # qsubj = [0, 1, 2, 3, 4, 5, 6, 7]
            # qval = [2, 2, 2, 2, 2, 2, 2, 2]
            # task.putqobj(qsubi, qsubj, qval)

            # Input the affine conic constraints
            task.appendafes(3)
            task.putafefentrylist(range(3), [7, 0, 1], [1]*3)
            # Quadratic conic constraints
            quadcone = task.appendquadraticconedomain(3)
            task.appendacc(quadcone, [0, 1, 2], None)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Optimize the problem
            task.optimize()

            # Get the solution values
            xx = task.getxx(mosek.soltype.itr)
            sol = xx[0:dim]
    return sol
