"""
@author: frode

This file contains functions for solving the tumour problem in both 1D and 2D.

The main-function calls to smaller trial-functions, which call the solver
functions using some initial conditions, and then plot the outputs.

What remains to be done: Alter the 
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


#==============================================================================
#   Numerical integration
#==============================================================================

def RK4(f, y, t, h):
    """
    One step of the numerical solution to the DE  (dy/dt = f).
    
    :param f:  Time-derivative of  y
    :param y:  Previous value of y, used in finding the next
    :param t:  Time
    :param h:  Time-step length
    
    :return:  Value of  y  at time  t+h
    """
    k1 = h*f(t,y)
    k2 = h*f(t+0.5*h, y+0.5*k1)
    k3 = h*f(t+0.5*h, y+0.5*k2)
    k4 = h*f(t+h, y+k3)
    return y + (1.0/6)*(k1 + 2*k2 + 2*k3 + k4)



#==============================================================================
#   1D
#==============================================================================

def tumour_solver_1D(coeffs, z_init, L, T, k_given, method):
    """
    Solves the 1D set of differential equations modelling cancer cells.
    
    :param coeffs:  Coefficients of the equations
    :param z_init:  Initial conditions for the functions   n, f, m
    :param L:  Length of the 1D system
    :param T:  End time of the simulation
    :param method:  String in  {"explicit", "implicit"}, determines solver
    
    :return:  Array  Z  of solutions at different times up to  T.
    """
    dn, dm, gamma, eta, alpha, beta = coeffs
    
    # Discretization
    N = len(z_init)/3
    h = L / (N-1)
    k = 0.25*h**2 / max(dn,dm)
    if method=="implicit":
        k = k_given*k
    time_steps = int(T/k)
    
    # Matrices  I, A, B1,  used in the schemes
    I = diags([[1]*N],[0])
    Adiagonals = [[-2]*N, [2]+[1]*(N-2), [1]*(N-2)+[2]]
    Aoffsets = [0,1,-1]
    A = diags(Adiagonals, Aoffsets)
    B1diagonals = [[0]+[1]*(N-2), [-1]*(N-2)+[0]]
    B1offsets = [1,-1]
    B1 = 0.5*diags(B1diagonals, B1offsets)
    
    def dzdt(t, z):
        """Computes the time derivative of the function everywhere, by RK4."""
        n = z[:N]
        f = z[N:2*N]
        m = z[2*N:]
        dndt = (1./h**2)*(dn * A.dot(n)
                        - gamma * (B1.dot(f))*(B1.dot(n))
                        - gamma * ( A.dot(f))*(n))
        dfdt = - eta*m*f
        dmdt = (dm / h**2) * A.dot(m) + alpha*n - beta*m
        return np.concatenate((dndt, dfdt, dmdt), axis=0)
    
    def explicit_next(t, z):
        """Computes the next step  by an explicit scheme, namely RK4."""
        z_next = RK4(dzdt, z, t, k)
        return z_next
    
    def implicit_next(z):
        """Computes the next step from  z  by a semi-implicit scheme."""
        M = len(z)/3
        n = z[:M]
        f = z[M:2*M]
        m = z[2*M:]
        f_next = f/(1+eta*k*m)      # First value of  f
        n_next = spsolve((I - (dn*k/h**2)*A 
                  + (gamma*k/h**2)*diags([B1.dot(f_next)],[0]).dot(B1) 
                  + (gamma*k/h**2)*diags([A.dot(f_next)],[0])),   n)
        m_next = spsolve(((1+ beta*k)*I - (dm*k/h**2)*A),  m + alpha*k*n)
        f_next = f/(1+eta*k*m_next) # Improved value of  f.
        return np.concatenate((n_next, f_next, m_next), axis=0)

    Z = np.zeros((len(z_init), time_steps + 1))
    Z[:, 0] = z_init
    
    if method == "explicit":
        for t in range(time_steps):
            Z[:, t+1] = explicit_next(t*k, Z[:,t])
    elif method == "implicit":
        for t in range(time_steps):
            Z[:, t+1] = implicit_next(Z[:,t])
    return Z



#==============================================================================
#   2D
#==============================================================================

def tumour_solver_2D(coeffs, z_init, L, T, k_given, method):
    """
    Solves the 2D set of differential equations modelling cancer cells.
    
    :param coeffs:  Coefficients of the equations
    :param z_init:  Initial conditions for the functions   n, f, m
    :param L:  Length of the 1D system
    :param T:  End time of the simulation
    
    :return:  Array  Z  of solutions at different times up to  T.
    """
    dn, dm, gamma, eta, alpha, beta = coeffs
    
    # Discretization
    N = int(np.sqrt(len(z_init)/3))
    M = N*N
    h = L / (N-1)
    k = 0.125*h**2 / max(dn,dm)
    if method=="implicit":
        k = k_given*k
    time_steps = int(T/k)
    
    # Matrices  A, B1 and BN, used in the scheme
    I = diags([[1]*M],[0])
    Adiagonals = [[-4]*M,
                  ([2]+[1]*(N-2)+[0])*(N-1) + [2] + [1]*(N-2),  
                  ([1]*(N-2)+[2]+[0])*(N-1) + [1]*(N-2) + [2],
                  [2]*N + [1]*(N*(N-2)),
                  [1]*(N*(N-2)) + [2]*N]
    Aoffsets = [0,1,-1,N,-N]
    A = diags(Adiagonals, Aoffsets)
    B1diagonals = [([0]+[1]*(N-2)+[0])*(N-1) + [0] + [1]*(N-2),  
                   ([-1]*(N-2)+[0]+[0])*(N-1) + [-1]*(N-2) + [0]]
    B1offsets = [1,-1]
    B1 = 0.5*diags(B1diagonals, B1offsets)
    BNdiagonals = [[0]*N + [1]*(N*(N-2)),
                   [-1]*(N*(N-2)) + [0]*N]
    BNoffsets = [N,-N]
    BN = 0.5*diags(BNdiagonals, BNoffsets)
    
    def dzdt(t, z):
        """Computes the time derivative everywhere, for all functions."""
        n = z[:M]
        f = z[M:2*M]
        m = z[2*M:]
        dndt = (1./h**2)*(dn * A.dot(n)
                        - gamma * (B1.dot(f))*(B1.dot(n))
                        - gamma * (BN.dot(f))*(BN.dot(n))
                        - gamma * ( A.dot(f))*(n))
        dfdt = - eta*m*f
        dmdt = (dm / h**2) * A.dot(m) + alpha*n - beta*m
        return np.concatenate((dndt, dfdt, dmdt), axis=0)
    
    def explicit_next(t, z):
        """Computes the next step explicitly, by RK4."""
        z_next = RK4(dzdt, z, t, k)
        return z_next
    
    def implicit_next(t, z):
        """Computes the next step semi-implicitly (backward)."""
        M = len(z)/3
        n = z[:M]
        f = z[M:2*M]
        m = z[2*M:]
        f_next = f/(1+eta*k*m)      # First value of  f.
        n_next = spsolve((I - (dn*k/h**2)*A 
                  + (gamma*k/h**2)*diags([B1.dot(f_next)],[0]).dot(B1) 
                  + (gamma*k/h**2)*diags([BN.dot(f_next)],[0]).dot(BN) 
                  + (gamma*k/h**2)*diags([A.dot(f_next)],[0])),   n)
        m_next = spsolve(((1+ beta*k)*I - (dm*k/h**2)*A),  m + alpha*k*n)
        f_next = f/(1+eta*k*m_next) # Improved value of  f.
        return np.concatenate((n_next, f_next, m_next), axis=0)

    Z = np.zeros((3*M, time_steps + 1))
    Z[:, 0] = z_init
    
    if method == "explicit":
        for t in range(time_steps):
            Z[:, t+1] = explicit_next(t*k, Z[:,t])
    elif method == "implicit":
        for t in range(time_steps):
            Z[:, t+1] = implicit_next(t*k, Z[:,t])
    return Z



#==============================================================================
#   Trials
#==============================================================================
  
def trial_1D(N, N_T, k_given, method="explicit"):
    """Just a simple test of the 1D solver"""
    
    # Equation coefficients
    dn = 0.001
    dm = 0.001
    gamma = 0.005
    eta = 10.0
    alpha = 0.1
    beta = 0.0
    
    coeffs = (dn, dm, gamma, eta, alpha, beta)
    
    # Dimensions
    L = 1.0
    h = L / (N-1)
    
    # Time steps and end
    k = 0.25*h**2 / max(dn,dm)
    if method=="implicit":
        k = k_given*k
    jump = N_T/10
    
    # Initial conditions
    epsilon = 0.01
    n0 = [np.exp(-((1.0*x)/N)**2/epsilon) for x in range(N)]
    f0 = [(1.0 - 0.25*np.exp(-((1.0*x)/N)**2/epsilon))*2.0 for x in range(N)]
    m0 = [0.5*np.exp(-((1.0*x)/N)**2/epsilon) for x in range(N)]
    z0 = np.concatenate((n0, f0, m0), axis=0)
    
    # Solve system
    Z_all = tumour_solver_1D(coeffs, z0, L, N_T*k, k_given, method)
    
    # Make plots
    for i in range(N_T)[::jump]:
        plt.plot(range(N), Z_all[:N,i])
    plt.show()
    
    for i in range(N_T)[0::jump]:
        plt.plot(range(N), Z_all[N:2*N,i])
    plt.show()
    
    for i in range(N_T)[::jump]:
        plt.plot(range(N), Z_all[2*N:,i])
    plt.show()
    

def trial_2D(N, N_T, k_given=1.0, method="explicit"):
    """Just a simple test of the 2D solver"""
    
    # Equation coeffs
    dn = 1.0e-3 # Dn / D
    dm = 1.0e-3 # Dm / D
    gamma = 0.005 #chi*f0 / D
    eta = 10.0 #tau*m0*delta
    alpha = 0.1 #tau*mu*n0/m0
    beta = 0.#1 #tau*lam
    
    coeffs = (dn, dm, gamma, eta, alpha, beta)
    
    # Dimensions
    M = N*N
    L = 1.0
    h = L / (N-1)
    
    # Discretisations
    k = 0.125*h**2 / max(dn,dm)
    if method=="implicit":
        k = k_given*k
    
    # Build initial conditions
    n_init = np.zeros(M)
    f_init = np.zeros(M)
    m_init = np.zeros(M)
    for i in range(N):
        for j in range(N):
            f_init[i*N + j] = 1.0
    for i in range(N/10):
        for j in range(N/10):
            n_init[i*N + j] = 0.5
            f_init[i*N + j] = 0.5
    z_init = np.concatenate((n_init,f_init,m_init), axis = 0)
    
    # Solve system
    Z = tumour_solver_2D(coeffs, z_init, L, N_T*k, k_given, method)
    
    for i in range(N_T+1)[::N_T/5]:
        print "\n\nStage ", i
        print "Left to right:"
        print "n - tumour cells"
        print "f - ECM (extracellular matrix)"
        print "m - MDE (matrix degrading enzymes)"
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.imshow(Z[:M,i].reshape((N,N)))
        plt.subplot(132)
        plt.imshow(Z[M:2*M,i].reshape((N,N)))
        plt.subplot(133)
        plt.imshow(Z[2*M:,i].reshape((N,N)))
        plt.show()



#==============================================================================
#       M A I N
#==============================================================================

if __name__=="__main__":
    trial_1D(100, 1000, 1.0, "implicit")
    trial_2D(60, 200, 5.0, "implicit")



