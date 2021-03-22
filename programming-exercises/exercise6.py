#!/usr/bin/env python

import math
import numpy as np
from scipy.optimize import minimize
from timeit import timeit
from numdifftools import Jacobian, Hessian

def solveUsingScipy(objFun, x0):
    res = minimize(objFun, x0, method='SLSQP', constraints=[], options={'ftol': 1e-4, 'disp': True})
    print(f'x1* = {res.x}\n')

def gdm(x0, objFun, jacobian=None, accuracy=1e-4):
    """Gradient Descent Method"""
    if jacobian is None:
        jacobian = Jacobian(objFun)
    steps = 0
    x = x0
    d = -jacobian(x)
    while(math.fabs(d) > accuracy):
        def backtrackingLineSearch():
            # Backtracking Line Search
            alpha = 0.25
            beta = 0.85
            t = 1
            steps = 0
            while(objFun(x + t*d) >= objFun(x) + alpha*t*jacobian(x)*d):
                t *= beta
                steps += 1
            return t
        t = backtrackingLineSearch()
        x = x + t*d
        d = -jacobian(x)
        steps += 1
    return (x, objFun(x), steps, math.fabs(d))

problems = [ (np.array([3.0]), lambda x:  2*x[0]**2 - 0.5, "2*x[0]**2 - 0.5")
           , (np.array([-2.0]), lambda x:  2*x[0]**4 - 4*x[0]**2 + x - 0.5, "2*x[0]**4 - 4*x[0]**2 + x - 0.5")
           , (np.array([-0.5]), lambda x:  2*x[0]**4 - 4*x[0]**2 + x - 0.5, "2*x[0]**4 - 4*x[0]**2 + x - 0.5")
           , (np.array([0.5]), lambda x:  2*x[0]**4 - 4*x[0]**2 + x - 0.5, "2*x[0]**4 - 4*x[0]**2 + x - 0.5")
           ]
for (x0, objFun, title) in problems:
    print(f'=== Solving {title} ===')
    (x, p, steps, finalAccuracy) = gdm(x0, objFun)
    print(f'x* = {x}')
    print(f'p* = {p}')
    print(f'steps = {steps}')
    print(f'eta = {finalAccuracy}')
    solveUsingScipy(objFun, x0)
    print("=============================================")
