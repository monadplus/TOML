#!/usr/bin/env python

import math
import numpy as np
import cvxpy as cp

x = cp.Variable(2)
P = np.array(np.mat('1. 0.; 0. 1.'))
P1 = np.array(np.mat('1. 0.; 0. 1.'))
P2 = np.array(np.mat('9. 0.; 0. 1.'))
P3 = np.array(np.mat('1. 0.; 0. 0.'))
P4 = np.array(np.mat('0. 0.; 0. 1.'))

constraints = [ x[0] >= 0.5
              , x[0] + x[1] + 1 >= 1
              , - cp.quad_form(x, P1) - 1 <= 0.
              # , cp.quad_form(x, P2) >= 9.
              # , cp.quad_form(x, P3) - x[1] >= 0
              # , cp.quad_form(x, P4) - x[0] >= 0
              ]
prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)), constraints)
prob.solve()
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, " x2 = ", x[1].value)
