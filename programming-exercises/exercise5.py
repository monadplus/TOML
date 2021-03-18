#!/usr/bin/env python

import numpy as np
from cvxopt import matrix, solvers, spdiag

solvers.options['maxiters'] = 1000

m = 2 # inequalities
n = 2 # variables

def F(x=None, z=None):
    if x is None:
        return m, matrix([2.0, 1.0], (n, 1))
    f_0 = x[0]**2 + x[1]**2
    f_1 = (x[0] - 1)**2 + (x[1] - 1)**2 - 1
    f_2 = (x[0] - 1)**2 + (x[1] + 1)**2 - 1
    f = matrix([f_0, f_1, f_2], (m+1, 1))
    Df_0 = [2*x[0], 2*x[1]]
    Df_1 = [2*(x[0] - 1), 2*(x[1] - 1)]
    Df_2 = [2*(x[0] - 1), 2*(x[1] + 1)]
    Df = matrix(np.array([Df_0, Df_1, Df_2]))
    if z is None:
        return f, Df
    else:
        # H = spdiag([H_0, H_1, H_2])
        H_0 = z[0] * matrix([2, 0, 0, 2], (n, n))
        H_1 = z[1] * matrix([2, 0, 0, 2], (n, n))
        H_2 = z[2] * matrix([2, 0, 0, 2], (n, n))
        H = H_0 + H_1 + H_2
        return f, Df, H

res = solvers.cp(F)
print(res['status'])
print(res['x'])
