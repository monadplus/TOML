import math
import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian
from backtracking import backtrackingLineSearch
from numpy.linalg import inv

def newtons(objFun, x0, jacobian=None, hessian=None, accuracy=1e-4):
    """Newton's Method"""
    # For some reason, Jacobian returns the jacobian in matrix form i.e. [[]]
    if jacobian is None: jacobian = Jacobian(objFun)
    if hessian is None: hessian = Hessian(objFun)
    def getLambdaSquare(x):
        j = jacobian(x).reshape(-1)
        hinv = inv(hessian(x))
        return j.T @ hinv @ j
    def getStep(x):
        j = jacobian(x).reshape(-1)
        hinv = inv(hessian(x))
        return - hinv @ j
    steps = 0
    x = x0
    d = getStep(x)
    l = getLambdaSquare(x)
    # TODO l may be negative...
    while(l/2 > accuracy):
        t = backtrackingLineSearch(objFun, x, jacobian=jacobian)
        x = x + t*d
        d = getStep(x)
        l = getLambdaSquare(x)
        steps += 1
    return (x, objFun(x), steps, l)
