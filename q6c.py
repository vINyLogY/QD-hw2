#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg, integrate
from scipy.sparse.linalg import LinearOperator, eigsh
from matplotlib import pyplot as plt
from matplotlib import rc
import minitn.bases.dvr as dvr
import minitn
rc('font', family='Times New Roman')
rc('text', usetex=True)

DTYPE = np.complex128

q0 = 25.
dim = 99
hbar = 1.0
omega = 1.0
m = 1.0

def vfunc(x):
    return 0.5 * omega**2 * m**2 * x**2



def main():
    solver = dvr.PO_DVR([(-q0, q0, dim)] * 2, hbar=hbar, m_e=m, fast=False)
    solver.set_v_func([vfunc, vfunc])
    
    
    e, v = solver.solve(n_state=10)
    plotter = solver.plot_wf(v[0], str(0))
    plotter.send(None)
    for i in range(1, 10):
        plotter.send((v[i], str(i)))
    np.savetxt('e.txt', e)
    np.savetxt('v.txt', v)
    

if __name__ == '__main__':
    main()
