#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 05:40:30 2019

@author: frode
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


#==============================================================================
#   Integration method: Runge-Kutta 4
#==============================================================================

def RK4(f, y, t, h):
    k1 = h*f(t,y)
    k2 = h*f(t+0.5*h, y+0.5*k1)
    k3 = h*f(t+0.5*h, y+0.5*k2)
    k4 = h*f(t+h, y+k3)
    return y + (1.0/6)*(k1 + 2*k2 + 2*k3 + k4)



#==============================================================================
#   Define parameters and derivative function
#==============================================================================

# Scaling factors
L = 1.0
D = 1.0e-10
tau = L**2 / D

# Discretization
N = 10
M = N*N
h = L / (N-1)

# Order of function values
n0 = 1.0
f0 = 1.0
m0 = 1.0

# System details
Dn = 1.0e-10
Dm = 1.0e-10
chi = 1.0
delta = 1.0
mu = 1.0
lam = 1.0

# Dimensionless coeffs
dn = 1.0e-3 # Dn / D
dm = 1.0e-3 # Dm / D
gamma = dn #chi*f0 / D
eta = 10.0 #tau*m0*delta
alpha = 0.1 #tau*mu*n0/m0
beta = 0.1 #tau*lam

# Matrices  A, B1 and BN
# A
Adiagonals = [[-4]*M,
              ([2]+[1]*(N-2)+[0])*(N-1) + [2] + [1]*(N-2),  
              ([1]*(N-2)+[2]+[0])*(N-1) + [1]*(N-2) + [2],
              [2]*N + [1]*(N*(N-2)),
              [1]*(N*(N-2)) + [2]*N]
Aoffsets = [0,1,-1,N,-N]

# B1
B1diagonals = [([0]+[1]*(N-2)+[0])*(N-1) + [0] + [1]*(N-2),  
               ([1]*(N-2)+[0]+[0])*(N-1) + [1]*(N-2) + [0]]
B1offsets = [1,-1]

# BN
BNdiagonals = [[0]*N + [1]*(N*(N-2)),
               [1]*(N*(N-2)) + [0]*N]
BNoffsets = [N,-N]

# Build them all
A = diags(Adiagonals, Aoffsets)
B1 = 0.5*diags(B1diagonals, B1offsets)
BN = 0.5*diags(BNdiagonals, BNoffsets)

#==============================================================================
# Derivative function
#==============================================================================
def F(t, z):
    M = len(z)/3
    n = z[:M]
    f = z[M:2*M]
    m = z[2*M:]
    dndt = (1./h**2)*( dn * A.dot(n)
                    - gamma * diags(B1.dot(n),0).dot(B1.dot(f))
                    - gamma * diags(BN.dot(n),0).dot(BN.dot(f))
                    - gamma * diags(n,0).dot(A.dot(f)) )
    dfdt = - delta * diags(m,0).dot(f)
    dmdt = (dm / h**2) * A.dot(m) + alpha*n - beta*m
    return np.concatenate((dndt, dfdt, dmdt), axis=0)


def tumour_solver(z_init, k, T):
    time_steps = int(T/k)
    Z = np.zeros((3*M, time_steps + 1))
    Z[:, 0] = z_init
    
    for t in range(time_steps):
        Z[:, t+1] = RK4(F, Z[:,t], t, k)
    
    return Z

n_init = np.zeros(M)
f_init = np.zeros(M)
m_init = np.zeros(M)
for i in range(N - N/2 + 1):
    n_init[N*(i + N/4) + N/4:N*(i + 1 + N/4) - N/4 ] = 1.0

for i in range(M):
    m_init[i] = 1.0
    if i%2 == 0:
        f_init[i] = 1.0
    else:
        f_init[i] = 0.5

z_init = np.concatenate((n_init,f_init,m_init), axis = 0)

k = 0.1 * h**2 / (4*dn + 4*gamma*f0)
print k
T = 50*k
Z = tumour_solver(z_init, k, T)

for i in [0, int(T/k)-1]:
    plt.imshow(Z[:M,i].reshape((N,N)))
    plt.show()
    plt.imshow(Z[M:2*M,i].reshape((N,N)))
    plt.show()
    
    
print Z
















