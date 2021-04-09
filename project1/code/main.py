#!/usr/bin/env python3
# coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from gpkit import Variable, Model
from scipy.optimize import minimize

plotsDir = 'plots/'

P     = 32.            # Payload [byte]
R     = 31.25          # CC2420 Radio Rate [kbyte/s = Byte/ms]
D     = 8              # number of levels
C     = 5              # neighbors size (connectivity)
N     = C*D**2         # number of nodes

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

Tw_max  = 500.       # Maximum Duration of Tw in ms
Tw_min  = 100.       # Minimum Duration of Tw in ms

def getAlphas(Fs) -> (float, float, float):
    d = 1 # where energy is maximized  (worst-case scenario)
    I = (2*d+1)/(2*d-1)
    Fi = Fs*((D**2 - d**2)/(2*d - 1))
    Fout = Fi + Fs
    Fb = (C - I)*Fout
    alpha1 = Tcs + Tal + (3/2)*Tps*((Tps + Tal)/2 + Tack + Tdata)*Fb
    alpha2 = Fout/2
    alpha3 = ((Tps+Tal)/2 + Tcs + Tal + Tack + Tdata)*Fout + ((3/2)*Tps + Tack + Tdata)*Fi + (3/4)*Tps*Fb
    return (alpha1, alpha2, alpha3)

def computeEnergy(Fs):
    # Trick to compute alphas once per call.
    alpha1, alpha2, alpha3 = getAlphas(Fs)
    # print(f'alpha1 {alpha1}'); print(f'alpha2 {alpha2}'); print(f'alpha3 {alpha3}')
    def go(Tw):
        return alpha1/Tw + alpha2*Tw + alpha3
    return go

def computeDelay(Tw, Fs):
    d = D # where delay is maximized (worst-case scenario)
    beta1 = 0.5*d
    beta2 = (Tcw/2 + Tdata)*d
    # print(f'beta1 {beta1}'); print(f'beta2 {beta2}')
    return beta1*Tw + beta2

def exercise1():
    # Compute energy and delay
    Fss = list(map(lambda x: 1.0/(x*60*1000), [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]))
    Tws = list(np.linspace(Tw_min, Tw_max, num=20))
    arr = np.zeros((len(Fss),(len(Tws)), 2), dtype=float)
    for i, Fs in enumerate(Fss):
        for j, Tw in enumerate(Tws):
            arr[i,j, 0] = computeEnergy(Fs)(Tw)
            arr[i,j, 1] = computeDelay(Tw, Fs)

    for i, subArr in enumerate(arr):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))

        # Fig 1
        axs[0].plot(Tws, subArr[:, 0], color='blue')
        axs[0].set_xlabel('$T_w$(ms)')
        axs[0].set_ylabel('Energy(J)')
        axs[0].set_title('Energy ~ $T_w$')

        # Fig 2
        axs[1].plot(Tws, subArr[:, 1], color='red')
        axs[1].set_xlabel('$T_w$(ms)')
        axs[1].set_ylabel('Delay(ms)')
        axs[1].set_title('Delay ~ $T_w$')

        # Fig 3
        axs[2].set_xlabel('$T_w$(ms)')
        axs[2].set_ylabel('Energy(J)')
        l1, = axs[2].plot(Tws, subArr[:, 0], color='blue', label='Energy')
        axcopy = axs[2].twinx()
        axcopy.set_ylabel('Delay(ms)')
        l2, = axcopy.plot(Tws, subArr[:, 1], color='red', label='Delay')
        plt.legend((l1, l2), (l1.get_label(), l2.get_label()), loc='upper left')

        # Fig 4
        axs[3].plot(subArr[:, 0], subArr[:, 1], color='black')
        axs[3].set_xlabel('Energy(J)')
        axs[3].set_ylabel('Delay(ms)')
        axs[3].set_title('Energy ~ Delay')

        # Whole plot
        fig.suptitle('XMAC: energy vs. delay', fontsize=12)
        fig.tight_layout()
        fp = plotsDir + 'exercise_1_{}.png'.format(str(i))
        fig.savefig(fp)
        print(f'(Exercise 1) {fp} written succesfully.')

def bottleneckConstraint(Fs, Tw: Variable):
    # Since the difference between ceiling and not ceiling is minimal, wlog we remove the ceiling.
    Ttx = (Tw/2)+Tack+Tdata
    I = C # If d=0 then I_d = C
    Fout1 = Fs*(D**2) # Fs*((D**2 - d**2 + 2*d - 1)/(2*d - 1)) where d = 1
    Etx1 = (Tcs + Tal + Ttx)*Fout1
    return I*Etx1 <= 1/4


# The output should be linearly incrementing until Lmax >= L(Tw of Emax).
# Then it should be constant.
def p1(Fs, Lmax) -> float:
    """
    minimize E

    subject to:
      L <= L_{max}
      T_w >= T_w^{min}
      |I^0|*E_{tx}^1 <= 1/4

    var. Tw
    """
    Tw = Variable('Tw')
    objective = computeEnergy(Fs)(Tw)
    constraints = [ computeDelay(Tw, Fs) <= Lmax
                  , Tw >= Tw_min
                  , bottleneckConstraint(Fs, Tw)
                  ]
    m = Model(objective, constraints)
    # m.debug() # Some problems are not feasible
    try:
        sol = m.solve(verbosity=0)
        return round(sol['variables'][Tw], 1)
    except Exception:
        return None

def p2(Fs, Ebudget) -> float:
    """
    minimize L

    subject to:
      E <= E_{budget}
      T_w >= T_w^{min}
      |I^0|*E_{tx}^1 <= 1/4

    var. Tw
    """
    Tw = Variable('Tw')
    objective = computeDelay(Tw, Fs)
    constraints = [ computeEnergy(Fs)(Tw) <= Ebudget
                  , Tw >= Tw_min
                  , bottleneckConstraint(Fs, Tw)
                  ]
    m = Model(objective, constraints)
    sol = m.solve(verbosity=0)
    # m.debug() # Some problems are not feasible
    try:
        sol = m.solve(verbosity=0)
        # Rouding to avoid numerical problems
        return round(sol['variables'][Tw], 1)
    except Exception:
        return None

def exercise2():
    Fs = 1.0/(30.0*60.0*1000.0) # arbitrary
    Lmaxs = np.linspace(500.0, 3000.0, num=20)
    Ebudgets = np.linspace(0.1, 3.0, num=20)
    Tws1 = np.fromiter(map(lambda Lmax: p1(Fs, Lmax), Lmaxs), dtype=float)
    Tws2 = np.fromiter(map(lambda Ebudget: p2(Fs, Ebudget), Ebudgets), dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Fig 1
    ax1.plot(Lmaxs, Tws1, color='blue')
    ax1.set_xlabel('$L_{max}$(ms)')
    ax1.set_ylabel('$T_w$(ms)')
    ax1.set_title('$T_w$ optimized (w.r.t. energy)')

    # Fig 2
    ax2.plot(Ebudgets, Tws2, color='blue')
    ax2.set_xlabel('$E_{budget}$(J)')
    ax2.set_ylabel('$T_w$(ms)')
    ax2.set_title('$T_w$ optimized (w.r.t. delay)')

    fig.suptitle('XMAC: optimization', fontsize=12)
    fig.tight_layout()
    fp = plotsDir + 'exercise_2.png'
    fig.savefig(fp)
    print(f'(Exercise 2) {fp} written succesfully.')

def exercise3():
    Fs = 1.0/(30.0*60.0*1000.0) # arbitrary

    # Constants
    Eworst = computeEnergy(Fs)(Tw_min)
    Lworst = computeDelay(Tw_max, Fs)

    # Variables:
    # x[0] = Tw
    # x[1] = E_1
    # x[2] = L_1

    def objective(x):
        E_1 = x[1]
        L_1 = x[2]
        return - np.log(Eworst - E_1) - np.log(Lworst - L_1)

    def constraints(x):
        Tw  = x[0]
        E_1 = x[1]
        L_1 = x[2]
        E = computeEnergy(Fs)(Tw)
        L = computeDelay(Tw, Fs)
        return [ Eworst - E
               , E_1 - E
               , Lworst - L
               , L_1 - L
               , Tw - Tw_min
               , bottleneckConstraint(Fs, Tw)
               ]

    ineq_cons = { 'type' : 'ineq',
                  'fun': constraints}

    x0 = np.array([300.0, 0.02, 1000.0])
    res = minimize(objective, x0, method='SLSQP', constraints=[ineq_cons], options={'ftol': 1e-9, 'disp': False})
    p = round(res['x'][0], 2)
    print(f'(Exercise 3) Tw* = {p} milliseconds w.r.t. energy/delay')

if __name__=="__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')
    exercise1()
    exercise2()
    exercise3()
