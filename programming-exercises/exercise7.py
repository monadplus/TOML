#!/usr/bin/env python

import math
from numpy import array
import cvxpy as cp

x = cp.Variable(3, name="x")
objective_fn = - cp.log(x[0]) - cp.log(x[1]) - cp.log(x[2])
constraints = [ x[0] + x[2] <= 1
              , x[0] + x[1] <= 2
              , x[2] <= 1
              , x[0] >= 0
              , x[1] >= 0
              , x[2] >= 0
              ]
assert objective_fn.is_dcp()
assert all(constraint.is_dcp() for constraint in constraints)
problem = cp.Problem(cp.Minimize(objective_fn), constraints)
problem.solve()
print("status: ", problem.status)
print(f'x*: {x.value}')
print("p*: ",  problem.value)
print("Dual values: ", [*map(lambda x: x.dual_value, constraints)])
