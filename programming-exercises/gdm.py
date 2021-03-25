import math
import numpy as np
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian
from backtracking import backtrackingLineSearch
import numpy.linalg as LA

def gdm(objFun, x0, jacobian=None, accuracy=1e-4):
    """Gradient Descent Method.
    """
    if jacobian is None:
        jacobian = Jacobian(objFun)
    def getStep(x): return -jacobian(x)
    def getEta(d): return LA.norm(d, ord=2)
    steps = 0 # number of iterations
    x = x0
    d = getStep(x)
    eta = getEta(d)
    while(eta > accuracy):
        t = backtrackingLineSearch(objFun, x, jacobian=jacobian)
        x = x + t*d
        d = getStep(x)
        eta = getEta(d)
        steps += 1
    return (x, objFun(x), steps, eta)
