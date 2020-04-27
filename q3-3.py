#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg, integrate
from matplotlib import pyplot as plt
from matplotlib import rc

rc('font', family='Times New Roman')
rc('text', usetex=True)

DTYPE = np.complex128

# set hbar omega = 1
omega_0 = np.concatenate((
    np.linspace(0, 2.5, 100, endpoint=False),
    np.linspace(2.5, 3, 1000, endpoint=False),
    np.linspace(3, 4.85, 100, endpoint=False),
    np.linspace(4.85, 4.95, 1000, endpoint=False),
    np.linspace(4.95, 6.9349, 100, endpoint=False),
    np.linspace(6.9349, 6.9351, 1000, endpoint=False),
    np.linspace(6.9351, 8, 100)
))
b = 1.0 / 3.0


def main():
    # for max_n in range(1, 5):
    #     quasienergy(max_n)
    # quasienergy(10)
    # plt.legend(loc=3)
    transit(100)
    plt.show()


def transit(max_n):
    transit = []
    for w0 in omega_0:
        dim = 2 * (max_n * 2 + 1)
        ham = np.zeros((dim, dim), dtype=DTYPE)
        for i in range(-max_n, max_n + 1):
            head = 2 * (max_n + i)
            ham[head, head] = -0.5 * w0 + i
            ham[head + 1, head + 1] = 0.5 * w0 + i
            if i != -max_n:
                ham[head, head - 1] = b
                ham[head + 1, head - 2] = b
            if i != max_n:
                ham[head, head + 3] = b
                ham[head + 1, head + 2] = b
        _, v = np.linalg.eigh(ham)
        origin = max_n * 2
        ans = 0.0
        alpha0 = v[origin]
        for n, row in enumerate(v):
            if n % 2 == 1:
                tmp = np.dot(np.abs(row)**2, np.abs(alpha0)**2)
                ans += tmp
        transit.append(ans)
        # fst_fbz = ham[origin - 1:origin + 1, origin - 1:origin + 1]
        # snd_fbz = ham[origin - 3:origin + 3, origin - 3:origin + 3]
        # fst_w, _ = np.linalg.eigh(fst_fbz)
        # snd_w, _ = np.linalg.eigh(snd_fbz)
        # fst_fbz_e.append(fst_w[0:2])
        # snd_fbz_e.append(snd_w[2:4])
    plt.plot(omega_0, transit, '-', )
    plt.xlabel(r'''$w_0/w$''')
    plt.ylabel(r'''Transition Probability''')
    
    return


def quasienergy(max_n):
    floquet_e = []
    for w0 in omega_0:
        dim = 2 * (max_n * 2 + 1)
        ham = np.zeros((dim, dim))
        for i in range(-max_n, max_n + 1):
            head = 2 * (max_n + i)
            ham[head, head] = -0.5 * w0 + i
            ham[head + 1, head + 1] = 0.5 * w0 + i
            if i != -max_n:
                ham[head, head- 1] = b
                ham[head + 1, head - 2] = b
            if i != max_n:
                ham[head, head + 3] = b
                ham[head + 1, head + 2] = b
        w, _ = np.linalg.eigh(ham)
        origin = max_n * 2
        floquet_e.append(w[origin:origin + 2])
        # fst_fbz = ham[origin - 1:origin + 1, origin - 1:origin + 1]
        # snd_fbz = ham[origin - 3:origin + 3, origin - 3:origin + 3]
        # fst_w, _ = np.linalg.eigh(fst_fbz)
        # snd_w, _ = np.linalg.eigh(snd_fbz)
        # fst_fbz_e.append(fst_w[0:2])
        # snd_fbz_e.append(snd_w[2:4])
    for e in zip(*floquet_e):
        plt.plot(omega_0, e, '-', label='$k = {}$'.format(max_n))
    plt.xlabel(r'''$w_0/w$''')
    plt.ylabel(r'''Quasienergy / $(\hbar\omega)$''')
    
    return


if __name__ == "__main__":
    main()
