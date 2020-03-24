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
dim = 101
hbar = 1.0
omega = 1.0
m = 1.0



def vfunc(x):
    return 0.5 * omega**2 * m**2 * x**2


def eigenpairs(q0, dim):
    xspace = np.linspace(-q0, q0, dim, endpoint=True)
    dx = 2.0 * q0 / (dim - 1)
    T = hbar ** 2 / (2.0 * m * dx ** 2) * (
        np.diag([2.0]*dim) + np.diag([-1.0]*(dim - 1), k=1) +
        np.diag([-1.0]*(dim - 1), k=-1)
    )
    V = np.diag(vfunc(xspace))
    H = T + V
    return linalg.eigh(H)


def ploter1():
    dim = 1001
    q0 = 25.
    xspace = np.linspace(-q0, q0, dim, endpoint=True)
    _, v = eigenpairs(q0, dim)
    for n in [0, 1, 10, 100]:
        plt.plot(xspace, v[:, n], '-', label='$n = {}$'.format(n))
    plt.xlim(-q0, q0)
    plt.xlabel(r'''$q$ (a.\,u.)''')
    plt.ylabel(r'''$\psi(q)$''')
    plt.legend()
    plt.show()


def ploter2():
    e, _ = eigenpairs(q0, dim)
    n = np.array(list(range(0, dim)))
    plt.plot(hbar*omega*(n + 0.5), hbar*omega*(n + 0.5), '-', label='Reference')
    plt.plot(hbar*omega*(n + 0.5), e, '.', label='Grid')
    plt.xlim(0, dim)
    plt.xlabel(r'''Excact Energy''')
    plt.ylabel(r'''Grid Energy''')
    plt.legend()
    plt.show()

def ploter3():
    try:
        q, e = np.loadtxt('q6a-data.txt')
    except:
        dimspace = np.arange(33, 1001)
        e = []
        for dim in dimspace:
            e.append(eigenpairs(q0, dim)[0][32] - hbar * omega * (32 + 0.5))
        q = 2.0 * q0 / (dimspace - 1)
        np.savetxt('q6a-data.txt', np.array([q, e]))
    plt.plot(q, e, '.', label='')
    plt.xlabel(r'''$\Delta q$ (a.\,u.)''')
    plt.ylabel(r'''Energy Error''')
    plt.xlim(0, 0.5)
    plt.ylim(-10, 10)
    plt.show()

def ploter3l():
    try:
        q, e = np.loadtxt('q6a-data.txt')
    except:
        dimspace = np.arange(33, 1001)
        e = []
        for dim in dimspace:
            e.append(eigenpairs(q0, dim)[0][32] - hbar * omega * (32 + 0.5))
        q = 2.0 * q0 / (dimspace - 1)
        np.savetxt('q6a-data.txt', np.array([q, e]))
    plt.semilogx(q, e, '.', label='')
    plt.xlabel(r'''$\Delta q$ (a.\,u.)''')
    plt.ylabel(r'''Energy Error''')
    plt.xlim(-0, 1)
    plt.ylim(-10, 10)
    plt.show()

def main():
    ploter3l()


if __name__ == '__main__':
    main()
