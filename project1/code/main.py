#!/usr/bin/env python3
# coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import gpkit
from gpkit import Variable, VectorVariable, Model
from gpkit.nomials import Monomial, Posynomial, PosynomialInequality

plotsDir = 'plots/'

P     = 32.            # Payload [byte]
R     = 31.25          # CC2420 Radio Rate [kbyte/s = Byte/ms]
D     = 8              # number of levels
C     = 5              # neighbors size (connectivity)
N     = C*D**2         # number of nodes

# NOTE times are in milliseconds (ms)
Lmax  = 5000.          # Maximal allowed Delay (ms)
Emax  = 1.             # Maximal Energy Budjet (J)

L_pbl = 4.             # preamble length [byte]
L_hdr = 9. + L_pbl     # header length [byte]
L_ack = 9. + L_pbl     # ACK length [byte]
L_ps  = 5. + L_pbl     # preamble strobe length [byte]

Tal  = 0.95            # ack listen period [ms]
Thdr = L_hdr/R         # header transmission duration [ms]
Tack = L_ack/R         # ACK transmission duration [ms]
Tps  = L_ps/R          # preamble strobe transmission duration [ms]
Tcw  = 15*0.62         # Contention window size [ms]
Tcs  = 2.60            # Time [ms] to turn the radio into TX and probe the channel (carrier sense)
Tdata = Thdr + P/R + Tack # data packet transmission duration [ms]

Fs   = 1.0/(60*30*1000) # 1 packet/half_hour = 1/(60*30*1000) pk/ms

Tw_max  = 500.       # Maximum Duration of Tw in ms
Tw_min  = 100.       # Minimum Duration of Tw in ms

def computeEnergy(Tw, Fs):
    d = 1 # where energy is maximized  (worst-case scenario)
    I = (2*d+1)/(2*d-1)
    Fi = Fs*((D**2 - d**2)/(2*d - 1))
    Fout = Fi + Fs
    Fb = (C - I)*Fout
    alpha1 = Tcs + Tal + (3/2)*Tps*((Tps + Tal)/2 + Tack + Tdata)*Fb
    alpha2 = Fout/2
    alpha3 = ((Tps+Tal)/2 + Tcs + Tal + Tack + Tdata)*Fout + ((3/2)*Tps + Tack + Tdata)*Fi + (3/4)*Tps*Fb
    return alpha1/Tw + alpha2*Tw + alpha3

def computeDelay(Tw, Fs):
    d = D # where delay is maximized (worst-case scenario)
    # TODO are the formulas ok?
    beta1 = 0.5*d
    beta2 = (Tcw/2 + Tdata)*d
    return beta1*Tw + beta2

def exercise1():
    Fss = list(map(lambda x: 1.0/(x*60*1000), [1, 5, 10, 15, 20, 25, 30]))
    Tws = list(np.linspace(Tw_min, Tw_max, num=20))
    arr = np.zeros((len(Fss),(len(Tws)), 2), dtype=float)
    for i, Fs in enumerate(Fss):
        for j, Tw in enumerate(Tws):
            arr[i,j, 0] = computeEnergy(Tw, Fs)
            arr[i,j, 1] = computeDelay(Tw, Fs)

    for i, subArr in enumerate(arr):
        fig, axs = plt.subplots(1, 3, figsize=(9, 9))
        axs[0].plot(Tws, subArr[:, 0], color='yellow')
        axs[0].set_xlabel('Tw')
        axs[0].set_title('Energy ~ Tw')

        axs[1].plot(Tws, subArr[:, 1], color='red')
        axs[1].set_xlabel('Tw')
        axs[1].set_title('Delay ~ Tw')

        axs[2].plot(Tws, subArr[:, 0], color='yellow', label='energy')
        axs[2].plot(Tws, subArr[:, 1], color='red', label='delay')
        axs[2].set_xlabel('Tw')
        axs[2].legend()
        axs[2].set_title('Energy/Delay')

        fig.savefig(plotsDir + 'exercise_1_{}.png'.format(str(i)))

if __name__=="__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')
    exercise1()
