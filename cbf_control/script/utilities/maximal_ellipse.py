import mosek
import numpy as np
import scipy.linalg
from math import ceil, log2
from numpy.linalg import det
import sys
from scipy import sparse
import scipy.sparse.linalg
import time

# Since the value of infinity is ignored, we define it solely
# for symbolic purposes
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

class number_data:
    def __init__(self):
        self.t = []
        self.d = []
        self.s = []
        self.sprime = self.s
        self.z = []
        self.f = []
        self.g = []
    def set_params(self, i, x):
        if i == 0:
            self.t = x
        elif i == 1:
            self.d = x
        elif i == 2:
            self.s = x
        elif i == 3:
            self.sprime = x
        elif i == 4:
            self.z = x
        elif i == 5:
            self.f = x
        elif i == 6:
            self.g = x

def maximal_ellipse(A, b):
    [Ad, ia] = np.unique(A,axis=0,return_index=True)
    A = Ad
    b = b[ia]

    [C, d] = mosek_nofusion(A, b)
    volume = det(C)
    return C, d, volume

def mosek_nofusion(A, b):
    m = A.shape[0]
    n = A.shape[1]
    l = ceil(log2(n))
    num = [1, n, 2**l-1, 2**l-1, 2**l, m*n, m]

    nvar = 0
    ndx = number_data()
    inf = 0

    for i in range(7):
        ndx.set_params(i, nvar + np.arange(1,num[i]+1))
        nvar += num[i]
    ndx.f = np.reshape(ndx.f, (n,m))
    ndx.f = ndx.f.T

    ncon = int(n*m+m+n+n+(2**l-n)+1+(n*(n-1)/2)+(2**l-1))

    nabar = int(n*m*n+n+n+(n*(n-1)/2))
    abar_ptr = 0

    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)
        # Create a task
        with env.Task() as task:
            # Append the objective coefficients
            c = np.zeros(nvar)
            c[ndx.t-1] = 1

            BARVARDIM = [2 * n]

            # Bound keys for constraints
            bkc = [mosek.boundkey.fx] * ncon

            # Bound values for constraints
            blc = -inf * np.ones(ncon)
            buc = inf * np.ones(ncon)

            # initial values for variables
            a = np.zeros((ncon, nvar))

            barai = []
            baraj = []
            barak = []
            baraval = []
            con_ndx = 0
            for i in range(m):
                for j in range(n):
                    subk = j + np.zeros(n)
                    subl = np.arange(n)
                    swap_mask = subk < subl
                    swap = subk[swap_mask]
                    subk[swap_mask] = subl[swap_mask]
                    subl[swap_mask] = swap

                    barai.append(list(subk))
                    baraj.append(list(subl))
                    barak.append(con_ndx)
                    baraval.append(list(A[i, :]))
                    abar_ptr += n

                    a[con_ndx , ndx.f[i, j] - 1] = -1
                    blc[con_ndx ] = 0
                    buc[con_ndx ] = 0
                    con_ndx += 1
                a[con_ndx , ndx.d - 1] = A[i, :]
                a[con_ndx , ndx.g[i] - 1] = 1
                blc[con_ndx ] = b[i]
                buc[con_ndx ] = b[i]
                con_ndx += 1

            for j in range(n):
                barai.append([j + n])
                baraj.append([j])
                barak.append(con_ndx)
                baraval.append(list([1]))
                abar_ptr += 1

                a[con_ndx, ndx.z[j] - 1] = -1
                blc[con_ndx] = 0
                buc[con_ndx] = 0
                con_ndx += 1

            for j in range(n):
                barai.append([j + n])
                baraj.append([j + n])
                barak.append(con_ndx)
                baraval.append(list([1]))
                abar_ptr += 1

                a[con_ndx, ndx.z[j] - 1] = -1
                blc[con_ndx] = 0
                buc[con_ndx] = 0
                con_ndx += 1

            for j in range(n, 2 ** l):
                a[con_ndx, ndx.z[j] - 1] = 1
                a[con_ndx, ndx.t - 1] = -1
                blc[con_ndx] = 0
                buc[con_ndx] = 0
                con_ndx += 1

            for k in range(n, 2 * n - 1):
                for j in range(k + 1, 2 * n):
                    barai.append([j])
                    baraj.append([k])
                    baraval.append(list([1]))
                    barak.append(con_ndx)
                    abar_ptr += 1
                    blc[con_ndx] = 0
                    buc[con_ndx] = 0
                    con_ndx += 1

            a[con_ndx, ndx.t - 1] = 2 ** (l / 2)
            a[con_ndx, ndx.s[-1] - 1] = -1
            blc[con_ndx] = 0
            buc[con_ndx] = 0
            con_ndx += 1

            for j in range(2 ** l - 1):
                a[con_ndx, ndx.s[j] - 1] = 1
                a[con_ndx, ndx.sprime[j] - 1] = -1
                blc[con_ndx] = 0
                buc[con_ndx] = 0
                con_ndx += 1


            # transform float list barai and baraj to int list
            for i in range(len(barai)):
               barai[i] = [int(x) for x in barai[i]]
               baraj[i] = [int(x) for x in baraj[i]]

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            idxi = 0
            idxj = 0
            temp_asub = []
            temp_aval = []
            for i in range(a.shape[1]):
                for j in range(a.shape[0]):
                    if a[j, i] != 0:
                        temp_asub.append(j)
                        temp_aval.append(a[j, i])
                        idxj += 1
                asub.append(temp_asub)
                aval.append(temp_aval)
                temp_asub = []
                temp_aval = []
                idxi += 1
                idxj = 0

            # Append the variables to the task
            task.appendvars(nvar)
            task.appendbarvars(BARVARDIM)

            # Append the constraints to the task
            task.appendcons(ncon)

            for j in range(nvar):
                # set the linear term c_j in the objective
                task.putcj(j, c[j])
                # set the bounds on the variables x_j
                task.putvarbound(j, mosek.boundkey.fr, -inf, inf)

            for j in range(ncon):
                # set the bounds on the constraints
                task.putconbound(j, bkc[j], blc[j], buc[j])
                # set the coefficients of the constraints
                task.putacol(j, asub[j], aval[j])

            # Add the quadratic cone constraints
            cone_ptr = 0
            lhs = np.concatenate((ndx.z - 1, ndx.sprime - 1))
            lhs_ptr = 0
            qua_sub = []
            rqua_sub = []
            for j in range(2 ** l - 1):
                rqua_sub.extend(lhs[lhs_ptr:lhs_ptr + 2])
                rqua_sub.extend([ndx.s[j] - 1])
                lhs_ptr += 2
                cone_ptr += 3
            for i in range(m):
                qua_sub.extend([ndx.g[i] - 1])
                qua_sub.extend(ndx.f[i, :] - 1)
                cone_ptr = cone_ptr + n + 1

            # Rows of the quadratic cone
            len_rquadcone = np.arange(len(rqua_sub))
            # Rows of the rotated quadratic cone
            len_quadcone = len(rqua_sub) + np.arange(len(qua_sub))
            # Concatenate the two sets of rows
            sub = np.append(rqua_sub, qua_sub)

            task.appendafes(cone_ptr)
            task.putafefentrylist(range(cone_ptr), sub, [1]*cone_ptr)

            # Rotated quadratic cone
            rcycle_len = int(len(rqua_sub) / 3.0)
            for i in range(rcycle_len):
                rquadcone = task.appendrquadraticconedomain(3)
                task.appendacc(rquadcone, len_rquadcone[i*3:i*3+3], None)

            # Quadratic cone
            cycle_len = int(len(qua_sub) / 3)
            for i in range(cycle_len):
                quadcone = task.appendquadraticconedomain(3)
                task.appendacc(quadcone, len_quadcone[i*3:i*3+3], None)



            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)

            # Divide all off-diagonal entries of Abar by 2
            for i in range(len(barai)):
                for j in range(len(barai[i])):
                    if barai[i][j] != baraj[i][j]:
                        temp = baraval[i][j] / 2
                        baraval[i][j] = temp


            # Diagonal F matrix
            for i in range(len(barai)):
                syma = task.appendsparsesymmat(BARVARDIM[0], barai[i], baraj[i], baraval[i])
                task.putbaraij(barak[i], 0, [syma], [1.0])

            # Optimize the task
            task.optimize()

            # Get the solution
            barx = task.getbarxj(mosek.soltype.itr, 0)
            xx = task.getxx(mosek.soltype.itr)

            # Extract the ellipsoid
            Y = np.zeros((2 * n, 2 * n))
            flat_ndx = 0
            d = []
            for i in range(2*n):
                for j in range(i, 2*n):
                    Y[j, i] = barx[flat_ndx]
                    flat_ndx += 1
            Y = Y + np.tril(Y, -1).T
            C = Y[:n, :n]
            for i in range(len(ndx.d)):
                d.append(xx[ndx.d[i] - 1])
            d = np.array(d)
            return C, d

            # Add variables







