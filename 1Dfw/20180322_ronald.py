#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:25:52 2019

Methods:
1) Ronald: fw in time, 1D
2) Frode: fw in time, 2D
3) ? Frode: bw in time,  solve lin syst every timestep

Per method:
1) stability and convergence
2) convergence plots 
3) test case

Realistic setting solved



@author:
"""
import numpy as np
import matplotlib.pyplot as plt

def fw_solver_1D(N,M, t_max):
    eps_u = 0.001
    eps_v = 0.001
    gamma_u = 0.001
    alpha_v = 0.001
    beta_v = 0.001
    eta_w = 0.001

    h = 1.0/float(N)
    k = 1.0/float(M)

    U = np.ndarray((M*t_max,N))
    W = np.ndarray((M*t_max,N))
    V = np.ndarray((M*t_max,N))



    # initial conditions
    U[0,:] = np.zeros((N))
    W[0,:] = np.zeros((N))
    V[0,:] = np.zeros((N))

    U[0,45:55] = 1.0
    W[0,0:100] = 0
    V[0, 0:100]= 0

    # ERROR IN k and h!!

    for t in range(0,M*t_max-1):
        for x in range(1, N-1):
            U[t+1, x] = eps_u*1/h*(U[t, x+1]- 2*U[t,x]+U[t, x-1])\
                        -gamma_u*1/h*(U[t,x]-U[t,x-1])*(W[t,x]-W[t, x-1])\
                        -gamma_u*1/h*U[t,x]*(W[t,x+1]-2*W[t,x]+W[t, x-1]) + U[t,x]
            V[t+1, x] = eps_v*1/h*(V[t, x+1]-2*V[t,x]+V[t,x-1]) + h*alpha_v*U[t,x]\
                        - h*beta_v*W[t,x]

        # setting the boundary
        U[t+1, 0] = U[t+1,1]
        U[t+1, -1] = U[t+1, -2]
        W[t+1, 0] = W[t+1, 1]
        W[t+1, -1] = W[t+1, -2]

        # updating W
        for x in range(0, N-1):
            W[t+1, x] = -eta_w*h*U[t, x]*W[t,x]+W[t,x]

    return U, W, V

U, W, V = fw_solver(100,1000, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
h = 1.0/100.0
x_range = np.arange(0.0, 1.0, h)
sol1, = ax.plot(x_range, U[0,:], 'b-')
sol1, = ax.plot(x_range, U[20,:], 'b-')
sol1, = ax.plot(x_range, U[99,:], 'b-')
sol1, = ax.plot(x_range, W[0,:], 'r-')
sol1, = ax.plot(x_range, W[20,:], 'r-')
sol1, = ax.plot(x_range, W[99,:], 'r-')
sol1, = ax.plot(x_range, V[0,:], 'g-')
sol1, = ax.plot(x_range, V[20,:], 'g-')
sol1, = ax.plot(x_range, V[99,:], 'g-')
plt.show()
