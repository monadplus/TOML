#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def objFun(x1,x2):
    return np.exp(x1)*(4*x1**2 + 2*x2**2 + 4*x1*x2 + 2*x2 + 1)

x1 = np.arange(-5, 5, 0.5)
x2 = np.arange(-5, 5, 0.5)
x1, x2 = np.meshgrid(x1, x2)
z = objFun(x1,x2)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=cm.viridis)
plt.savefig('./report/figs/exercise1_plot.png')
