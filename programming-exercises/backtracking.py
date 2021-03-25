from numdifftools import Jacobian

def backtrackingLineSearch(objFun, x, alpha=0.25, beta=0.85, jacobian=None):
    """Backtracking Line Search"""
    t = 1
    if jacobian is None: jacobian = Jacobian(objFun)
    j = jacobian(x)
    # TODO do you also use jacobian for newton's ?
    while(objFun(x + t*(-j)) >= objFun(x) + alpha*t*j.T*(-j)):
        t *= beta
    return t
