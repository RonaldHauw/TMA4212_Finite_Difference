import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import sys
import SBW_util as util
from matplotlib.animation import FuncAnimation


eps_u = 0.001 # 0.01
eps_v = 0.001 # 0.001
gamma_u = 0.005# 0.05
zeta = 0.0
alpha_v = 0.1
beta_v = 0.1
eta_w = 10.0

def constr_lineqU(U, W, V, N, M, T):
    '''
        N: nb of x grid points (int)
        T: current timestep (int)
        U: discrete solution of u (np.array)
        W: discrete solution of w (np.array)
        V: discrete solution of v (np.array)
        M: nb of time steps (int)
    '''

    h = 1.0/float(N)
    k = 1.0/float(M)

    #assert(U.shape == W.shape and W.shape == V.shape, 'Dim error')
    #assert(U.shape[1] ==N and U.shape[0] == M, 'Dim error')
    DT = 0

    X_length = N
    A2Ut = np.zeros((X_length, X_length))
    A1Ut = np.zeros((X_length, X_length))

    fU = np.zeros((X_length, ))

    # BOUNDARY CONDITIONS
    A2Ut[0,0], A2Ut[0, 1] = -1, 1 # left boundary
    A2Ut[-1, -2], A2Ut[-1,-1] = -1, 1 # right boundary
    
    A1Ut[0,0], A1Ut[0, 1] = -1, 1 # left boundary
    A1Ut[-1, -2], A1Ut[-1,-1] = -1, 1 # right boundary

    # A1 UM+1 = f - A2 UM
    for i in range(1, X_length-1): # for each x in space do
        A2Ut[i, i] = -1 - zeta*(-2*eps_u*k/(h**2)+ gamma_u*k/(h**2)*(W[T-DT, i+1]+W[T-DT, i-1]-2*W[T-DT, i])) # contribution of UN
        A2Ut[i, i+1] = - zeta*(eps_u*k/(h**2)+gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1])) # contribution of UN-1
        A2Ut[i, i-1] = - zeta*(eps_u*k/(h**2)-gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1]))

        A1Ut[i,i] = 1 - (1-zeta)*(-2*eps_u*k/(h**2)+ gamma_u*k/(h**2)*(W[T-DT, i+1]+W[T-DT, i-1]-2*W[T-DT, i]))
        A1Ut[i, i+1] = - (1-zeta)*(eps_u*k/(h**2)+gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1]))
        A1Ut[i, i-1] = -(1-zeta)*(eps_u*k/(h**2)-gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1]))
    
    dummy = A2Ut@U[T-1,:]
    fU = fU - dummy
    return A1Ut, fU, A2Ut



def constr_lineqV(U, W, V, N, M, T):
    '''
        N: nb of x grid points (int)
        T: current timestep (int)
        U: discrete solution of u (np.array)
        W: discrete solution of w (np.array)
        V: discrete solution of v (np.array)
        M: nb of time steps (int)
    '''

    k = 1.0/float(M)
    h = 1.0/float(N)
    #k = 0.25*h**2*1.0/eps_v

    #assert(U.shape == W.shape and W.shape == V.shape, 'Dim error')
    #assert(U.shape[1]==N and U.shape[0] == M, 'Dim error')

    X_length = N
    A2Vt = np.zeros((X_length, X_length))
    A1Vt = np.zeros((X_length, X_length))

    fV = np.zeros((X_length, ))

    # BOUNDARY CONDITIONS
    A2Vt[0,0], A2Vt[0, 1] = -1.0, 1.0 # left boundary
    A2Vt[-1, -2], A2Vt[-1,-1] = -1.0, 1.0 # right boundary

    A1Vt[0,0], A1Vt[0, 1] = -1.0, 1.0 # left boundary
    A1Vt[-1, -2], A1Vt[-1,-1] = -1.0, 1.0 # right boundary
    # A1 VM+1 = f - A2 VM
    for i in range(1, X_length-1): # for each x in space do
        A1Vt[i, i]   = 1 + (1-zeta)*2*eps_v*k/(h**2) + beta_v*(1-zeta)
        A1Vt[i, i-1] = -(1-zeta)*eps_v*k/(h**2)
        A1Vt[i, i+1] = -(1-zeta)*eps_v*k/(h**2)
        
        A2Vt[i, i]   = -1 + zeta*2*eps_v*k/(h**2) + beta_v*zeta
        A2Vt[i, i-1] = -zeta*eps_v*k/(h**2)
        A2Vt[i, i+1] = -zeta*eps_v*k/(h**2)

        fV[i] = alpha_v*U[T-1, i]
    
    dummy = A2Vt@V[T-1,:]
    fV = fV - dummy

    return A1Vt, fV, A2Vt



def constr_lineqW(U, W, V, N, M, T):
    '''
        N: nb of x grid points (int)
        T: current timestep (int)
        U: discrete solution of u (np.array)
        W: discrete solution of w (np.array)
        V: discrete solution of v (np.array)
        M: nb of time steps (int)
    '''

    #k = 1.0/float(M)
    h = 1.0/float(N)
    k = 1.0/float(M)

    #k = 0.25*h**2*1.0/eps_v

    #assert(U.shape == W.shape and W.shape == V.shape, 'Dim error')
    #assert(U.shape[1]==N and U.shape[0] == M, 'Dim error')

    X_length = N
    A2Wt = np.zeros((X_length, X_length))
    A1Wt = np.zeros((X_length, X_length))

    fW = np.zeros((X_length, ))

    for i in range(0, X_length): # for each x in space do
        A1Wt[i,i] = 1.0 + k*(1-zeta)*eta_w*V[T,i]
        
        A2Wt[i,i] = -1.0 + k*zeta*eta_w*V[T,i]
    
    dummy = A2Wt@W[T-1,:]
    fW = fW - dummy

    return A1Wt, fW, A2Wt




def SB_solver_1D(N, M, nb_sec):
    print(M)
    h = 1.0/float(N)
    k = 1.0/float(M)
    

    U = np.zeros((M*nb_sec,N))
    W = np.zeros((M*nb_sec,N))
    V = np.zeros((M*nb_sec,N))
    
    epsilon = 0.01

    # J = 0: SET INITIAL conditions
    n0 = [np.exp(-((1.0*x)/N)**2/epsilon)*1.0 for x in range(N)]
    f0 = [(1.0 - 0.25*np.exp(-((1.0*x)/N)**2/epsilon))*1.0 for x in range(N)]
    m0 = [0.5*np.exp(-((1.0*x)/N)**2/epsilon) for x in range(N)]
    U[0,:] = np.array(n0)
    U[0,0], U[0,-1] = U[0,1], U[0,-2]
    #V[0,:] = np.array(m0)
    #V[0,0], V[0,-1] = V[0,1], V[0,-2]
    W[0,:] = np.array(f0)
    
    rhoU, rhoV, rhoW = [], [], []
    
    # J = 1..M
    for j in range(1, M*nb_sec): # for each time do
        #print('_', end='')
        AV, fV, BV = constr_lineqV(U, W, V, N, M, j) # use newer values of U already?
        V_solve = la.solve(AV, fV)#, V_guess
        V[j,:] = V_solve
        
        AW, fW , BW = constr_lineqW(U, W, V, N, M, j)
        W_solve = la.solve(AW, fW)
        W[j,:] = W_solve
        
        AU, fU, BU = constr_lineqU(U, W, V, N, M, j)
        U_solve = la.solve(AU, fU)#, U_guess
        U[j,:] = U_solve
    
    return U, V, W, AU, BU, AV, BV, rhoU, rhoV, rhoW



compsol = True

N_tsol = 1024
nb_sec = 128

hsol = 1.0/float(N_tsol)
#ksol = 0.25*h**2 / max(eps_u,eps_v)
#M_tsol = int(1.0/k)
M_tsol=1024
ksol=1.0/float(M_tsol)


if compsol :
    Usol, Vsol, Wsol, AUx, BUx, AVx, BVx, rhoUx, rhoVx, rhoWx = SB_solver_1D(N_tsol,M_tsol, nb_sec)
    
e=np.zeros(20)

for i in range (5,  9):
    N_t=2**i
    nb_sec = 128
    M_t=2**i
    k=1.0/float(M_tsol)
    Tsol= M_tsol*nb_sec -1
    T= M_t*nb_sec -1
    K=M_tsol-1
   # print(M_t,N_t)
    Usolfit=np.zeros((M_t*nb_sec,N_t))
    U, V, W, AUx, BUx, AVx, BVx, rhoUx, rhoVx, rhoWx = SB_solver_1D(N_t,M_t, nb_sec)
    #print(np.size(Usolfit),np.size(U))
    
    #for j in range(10) :
        #Usolfit[j,:] =U[2**(M_t-i)*j,2**(6-i):] 
        
    

    e[i-1]= la.norm(U[T,:] - Usol[Tsol,::2**(K-i)])
   # print(e)
    
    
    
    
xx = np.array(range(0,20))
fig = plt.figure(1)
plt.plot(xx, e)
   

    
