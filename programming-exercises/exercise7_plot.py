#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def objFun(x1,x2):
    return np.log(x1) + np.log(x2)

x1 = np.arange(0.1, 5, 0.5)
x2 = np.arange(0.1, 5, 0.5)
x1, x2 = np.meshgrid(x1, x2)
z = objFun(x1,x2)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=cm.viridis)
# plt.show()
plt.savefig('./report/figs/exercise7_plot.png')
