#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg, integrate
from matplotlib import pyplot as plt
from matplotlib import rc
import minitn.bases.dvr as dvr
import minitn
rc('font', family='Times New Roman')
rc('text', usetex=True)

DTYPE = np.complex128

q0 = 25.
dim = 999
hbar = 1.0
omega = 1.0
m = 1.0

def vfunc(x):
    return 0.5 * omega**2 * m**2 * x**2



def main1():
    solver = dvr.FastSineDVR(-q0, q0, dim, hbar=hbar, m_e=m)
    solver.set_v_func(vfunc)
    solver.solve()
    xspace = np.linspace(-q0, q0, 1000, endpoint=True)
    for i in [0, 1, 10, 100]:
        phi = solver.dvr2cont(solver.eigenstates[i])
        plt.plot(xspace, phi(xspace), '-', label='$n = {}$'.format(i))
    plt.xlim(-q0, q0)
    plt.xlabel(r'''$q$ (a.\,u.)''')
    plt.ylabel(r'''$\psi(q)$''')
    plt.legend()
    plt.show()

def main2():
    dim = 100
    solver = dvr.SineDVR(-q0, q0, dim, hbar=hbar, m_e=m)
    solver.set_v_func(vfunc)
    e, _ = solver.solve()
    n = np.arange(0, dim-1)
    plt.plot(hbar*omega*(n + 0.5), hbar*omega*(n + 0.5), '--', label='Reference')
    plt.plot(hbar*omega*(n + 0.5), e, '.', label='DVR')
    plt.xlim(0, dim)
    plt.xlabel(r'''Excact Energy''')
    plt.ylabel(r'''Grid Energy''')
    plt.legend()
    plt.show()


def main3():
    try:
        qlist, elist = np.loadtxt('q6b-data.txt')
    except:
        dimspace = np.arange(34, 1001,)
        elist = []
        for dim in dimspace:
            solver = dvr.FastSineDVR(-q0, q0, dim, hbar=hbar, m_e=m)
            solver.set_v_func(vfunc)
            e, _ = solver.solve()
            elist.append(e[32] - hbar * omega * (32 + 0.5))
        qlist = 2.0 * q0 / (dimspace - 1)
        np.savetxt('q6b-data.txt', np.array([qlist, elist]))
    plt.plot(qlist, elist, '.', label='')
    plt.semilogx(qlist, elist, '.', label='')
    plt.xlabel(r'''$\Delta q$ (a.\,u.)''')
    plt.ylabel(r'''Energy Error''')
    plt.xlim(0, 0.5)
    plt.ylim(-1, 1)
    plt.show()


def main3l():
    try:
        qlist, elist = np.loadtxt('q6b-data.txt')
    except:
        dimspace = np.arange(34, 1001,)
        elist = []
        for dim in dimspace:
            solver = dvr.FastSineDVR(-q0, q0, dim, hbar=hbar, m_e=m)
            solver.set_v_func(vfunc)
            e, _ = solver.solve()
            elist.append(e[32] - hbar * omega * (32 + 0.5))
        qlist = 2.0 * q0 / (dimspace - 1)
        np.savetxt('q6b-data.txt', np.array([qlist, elist]))
    plt.semilogx(qlist, elist, '.', label='')
    plt.xlabel(r'''$\Delta q$ (a.\,u.)''')
    plt.ylabel(r'''Energy Error''')
    plt.xlim(0, 1)
    plt.ylim(-10, 10)
    plt.show()

if __name__ == '__main__':
    main3l()
