#!/usr/bin/env python

import math
from numpy import array
import cvxpy as cp

x = cp.Variable(3, name="x")
R = cp.Variable(3, name="R") # R_12, R_23, R_32
objective_fn = cp.log(x[0]) + cp.log(x[1]) + cp.log(x[2])
constraints = [ x[0] + x[1] <= R[0]
              , x[0] <= R[1]
              , x[2] <= R[2]
              , R[0] + R[1] + R[2] <= 1
              ]
assert objective_fn.is_dcp()
assert all(constraint.is_dcp() for constraint in constraints)
problem = cp.Problem(cp.Maximize(objective_fn), constraints)
problem.solve()
print("status: ", problem.status)
print("p*: ",  problem.value)
print(f'x*: {x.value}')
print(f'R*: {R.value}')
print("Dual values: ", [*map(lambda x: x.dual_value, constraints)])
