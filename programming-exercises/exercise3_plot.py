#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def objFun(x1,x2):
    return x1**2 + x2**2

def ineq1(x1,x2):
    return x1**2 + x1*x2 + x2**2 - 3

def ineq2(x1,x2):
    return 3*x1 + 2*x2 - 3

x1 = np.arange(-5, 5, 0.5)
x2 = np.arange(-5, 5, 0.5)
x1, x2 = np.meshgrid(x1, x2)
z = objFun(x1,x2)
in1 = ineq1(x1,x2)
in2 = ineq2(x1,x2)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=cm.viridis)
ax.plot_surface(x1, x2, in1, rstride=1, cstride=1, cmap=cm.inferno)
ax.plot_surface(x1, x2, in2, rstride=1, cstride=1, cmap=cm.plasma)
# plt.show()
plt.savefig('./report/figs/exercise3_plot.png')
