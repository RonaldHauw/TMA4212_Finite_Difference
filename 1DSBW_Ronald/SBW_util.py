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
    if tstep==1:
        open("{0}.txt".format(title), 'w').close()
    f = open("{0}.txt".format(title), "a")
    f.write('\n {0}: {1}'.format(tstep, specrad(A)))
    f.close()
