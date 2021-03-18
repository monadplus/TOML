#!/usr/bin/env python

import math
import numpy as np
from scipy.optimize import minimize
from timeit import timeit
from numdifftools import Jacobian, Hessian

def obj_fun(x):
    return math.exp(x[0])*(4*x[0]**2 + 2*x[1]**2 + 4*x[0]*x[1] + 2*x[1] + 1)

def fun_jac(x):
    dx = math.exp(x[0])*(4*x[0]**2 + 4*x[0]*(x[1] + 2) + 2*x[1]**2 + 6*x[1] + 1)
    dy = math.exp(x[0])*(4*x[0] + 4*x[1] + 2)
    return np.array([dx, dy])

# Slow, it seems it is computed each time..
fun_jac_true = Jacobian(obj_fun)

def fun_hess(x, a):
    dxx = math.exp(x[0])*(4*x[0]**2 + 4*x[0]*(x[1] + 2)*2*x[1]**2 + 10*x[1] + 9)
    dxy = math.exp(x[0])*(4*x[0] + 4*x[1] + 6)
    dyx = 2*math.exp(x[0])*(2*x[0] + 2*x[1] + 3)
    dyy = 4*math.exp(x[0])
    return np.array([[dxx, dxy],[dyx, dyy]])

ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array( -x[0]*x[1]+x[0]+x[1]-1.5
                                        , x[0]*x[1]+10)
            }

x0s = np.array([[.0, .0],[10.0,20.0],[-10.0, 1.0],[-30.0,-30.0]])

for x0 in x0s:
    print('=== Regular ===')
    res = timeit(minimize, obj_fun, x0, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
    print(f'[x1*, x2*] = {res.x}\n')

    print('=== Optimized ===')
    res = timeit(minimize, obj_fun, x0, method='SLSQP', jac=fun_jac, constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
    print(f'[x1*, x2*] = {res.x}\n')
