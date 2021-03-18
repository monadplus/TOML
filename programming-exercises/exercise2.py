#!/usr/bin/env python

import math
import numpy as np
from scipy.optimize import minimize
from timeit import timeit

def obj_fun(x):
    return x[0]**2 + x[1]**2

def fun_jac(x):
    dx = 2*x[0]
    dy = 2*x[1]
    return np.array([dx, dy])

ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([x[0] - 0.5
                                       , x[0] + x[1] - 1
                                       , x[0]**2 + x[1]**2 - 1
                                       , 9*x[0]**2 + x[1]**2 - 9
                                       , x[0]**2 - x[1]
                                       , x[1]**2 - x[0]]
                                       )
            }

x0Feasible = np.array([3, 3])
print(f'Feasible start point')
res = timeit(minimize, obj_fun, x0Feasible, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
print(f'[x1*, x2*] = {res.x}\n')

# violates constrant 0.5 <= x_1
x0NotFeasible = np.array([-0.5, 1])
print(f'Not feasible start point')
res = timeit(minimize, obj_fun, x0NotFeasible, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
print(f'[x1*, x2*] = {res.x}\n')

print(f'With Jacobian')
res = timeit(minimize, obj_fun, x0Feasible, method='SLSQP', jac=fun_jac, constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
print(f'[x1*, x2*] = {res.x}\n')
