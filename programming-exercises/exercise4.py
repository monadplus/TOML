#!/usr/bin/env python

import numpy as np
from cvxopt import matrix, solvers, spdiag

# matrices in cvxopt are weirdly indexed.

# >>> m = matrix([1,2,3,4], (2,2)))
# >>> print(m)
# [1 3]
# [2 4]

# m[0] = 1
# m[1] = 2
# m[2] = 3
# print(m[0,:]) = [1 3]

# >>> print(matrix(np.array([[1,2],[3,4]])))
# [1 2]
# [3 4]


m = 1 # #inequalities
n = 1 # #variables

# return None if x is not in the domain of f_0
def F(x=None, z=None):
    if x is None:
        return m, matrix(3.0, (n, 1))
    # x is always in the domain
    f_0 = x[0]**2 + 1
    f_1 = (x[0] - 2)*(x[0] - 4)
    f = matrix([f_0, f_1], (m+1, 1))
    Df_0 = 2*x[0]
    Df_1 = 2*(x[0] - 3)
    Df = matrix([Df_0, Df_1]) # (m+1, 1)
    if z is None:
        return f, Df
    else:
        H = spdiag([z[0]*2 + z[1]*2])
        return f, Df, H

res = solvers.cp(F)
print(res['status'])
print(res['x'])
