#!/usr/bin/env python

import math
from numpy import array
from scipy.optimize import minimize
from timeit import timeit
from gdm import gdm
from newtons import newtons

def solveUsingScipy(objFun, x0, jac, hess):
    print(f'SLSQP:')
    res = minimize(objFun, x0, method='SLSQP', constraints=[], options={'ftol': 1e-4, 'disp': True})
    print(f'\tx1* = {res.x}\n')

problems = [ (array([3.0]), lambda x:  2*x[0]**2 - 0.5, lambda x: array([ 4*x[0] ]), None, "2*x[0]**2 - 0.5")
           , (array([-2.0]), lambda x:  2*x[0]**4 - 4*x[0]**2 + x - 0.5, None, None, "2*x[0]**4 - 4*x[0]**2 + x - 0.5")
           , (array([-0.5]), lambda x:  2*x[0]**4 - 4*x[0]**2 + x - 0.5, None, None, "2*x[0]**4 - 4*x[0]**2 + x - 0.5")
           , (array([0.5]), lambda x:  2*x[0]**4 - 4*x[0]**2 + x - 0.5, None, None, "2*x[0]**4 - 4*x[0]**2 + x - 0.5")
           , (array([2.0]), lambda x:  2*x[0]**4 - 4*x[0]**2 + x - 0.5, None, None, "2*x[0]**4 - 4*x[0]**2 + x - 0.5")
           ]

def solveUsingGDM(objFun, x0, jac, hess):
    print(f'Gradient Descent Method:')
    (x, p, steps, finalAccuracy) = timeit(gdm, objFun, x0, jacobian=jac)
    print(f'\tx* = {x[0]}')
    print(f'\tp* = {p}')
    print(f'\tsteps = {steps}')
    print(f'\teta = {finalAccuracy}')

def solveUsingNM(objFun, x0, jac, hess):
    print(f'Newton\'s Method:')
    (x, p, steps, finalAccuracy) = timeit(newtons, objFun, x0, jacobian=jac, hessian=hess)
    print(f'\tx* = {x}')
    print(f'\tp* = {p}')
    print(f'\tsteps = {steps}')
    print(f'\teta = {finalAccuracy}')

for (x0, objFun, jac, hess, title) in problems:
    print(f"============ Solving {title} =================")
    solveUsingScipy(objFun, x0, jac, hess)
    solveUsingGDM(objFun, x0, jac, hess)
    solveUsingNM(objFun, x0, jac, hess)
