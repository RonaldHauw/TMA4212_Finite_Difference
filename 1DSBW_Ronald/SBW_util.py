import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from tqdm import tqdm
import sys
from matplotlib.animation import FuncAnimation


def gershgorin_bound(A):
    '''
        return geshgorin bound on spectral radius of A
    '''
    N,M = A.shape
    assert(N==M, 'matrix not square')
    specrad = 0
    for i in range(0, N):
        diag_elem = A[i,i]
        sum_non_diag = 0.0
        for j in range(0, M):
            if j != i:
                sum_non_diag += abs(A[i,j])
        cur_specrad = diag_elem + abs(sum_non_diag)
        if cur_specrad>= specrad:
            specrad = cur_specrad
    return specrad

def specrad(A):
    eigs = la.eigvals(A)
    return np.max(eigs)
    

def track_specrad(A, tstep, title):
    dat = specrad(A)
    if tstep==1:
        open("{0}.txt".format(title), 'w').close()
    f = open("{0}.txt".format(title), "a")
    f.write('{0}: {1}\n'.format(tstep, dat))
    f.close()
    return dat

def color_grads(steps):
    blue_max = np.array([0.2, 0.2, 0.99])
    red_max = np.array([0.99, 0.2, 0.2])
    green_max = np.array([0.2, 0.99, 0.2])
    blue_min = np.array([0.09, 0.09, 0.4])
    red_min = np.array([0.3, 0.09, 0.09])
    green_min = np.array([0.09, 0.3, 0.09])
    B = []
    R = []
    G = []
    for n in np.linspace(0.0, 1.0, steps):
        B += [blue_max*n+(1-n)*blue_min]
        G += [green_max*n+(1-n)*green_min]
        R += [red_max*n+(1-n)*red_min]
    
    return R, G, B
        
    
    
    

