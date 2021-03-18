#!/usr/bin/env python

import math
import numpy as np
from scipy.optimize import minimize
from timeit import timeit

def obj_fun(x):
    return x[0]**2 + x[1]**2

ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([ -x[0]**2 - x[0]*x[1] - x[1]**2 + 3
                                        , 3*x[0] + 2*x[1] - 3])}

x0 = np.array([0, 0])
res = timeit(minimize, obj_fun, x0, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': True})
print(f'[x1*, x2*] = {res.x}\n')
