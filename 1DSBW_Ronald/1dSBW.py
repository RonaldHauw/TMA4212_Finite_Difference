import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from tqdm import tqdm
import sys
import SBW_util as util
from matplotlib.animation import FuncAnimation


# init cond
# plots 1D 
# report (todo's, structure, division of work )
# 2D scheme correspondence
# stability 
# convergence plots
# test case w oxygen 


eps_u = 0.001 # 0.01
eps_v = 0.001 # 0.001
gamma_u = 0.0041# 0.05
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

    #k = 1.0/float(M)
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
    A2Ut[0,0], A2Ut[0, 1] = -1.0/h, 1.0/h # left boundary
    A2Ut[-1, -2], A2Ut[-1,-1] = -1.0/h, 1.0/h # right boundary
    
    A1Ut[0,0], A1Ut[0, 1] = -1.0/h, 1.0/h # left boundary
    A1Ut[-1, -2], A1Ut[-1,-1] = -1.0/h, 1.0/h # right boundary

    # A1 UM+1 = f - A2 UM
    for i in range(1, X_length-1): # for each x in space do
        A2Ut[i, i] = -1 - zeta*(-2*eps_u*k/(h**2)+ gamma_u*k/(h**2)*(W[T-DT, i+1]+W[T-DT, i-1]-2*W[T-DT, i])) # contribution of UN
        A2Ut[i, i+1] = - zeta*(eps_u*k/(h**2)-gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1])) # contribution of UN-1
        A2Ut[i, i-1] = - zeta*(eps_u*k/(h**2)+gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1]))

        A1Ut[i,i] = 1 - (1-zeta)*(-2*eps_u*k/(h**2)+ gamma_u*k/(h**2)*(W[T-DT, i+1]+W[T-DT, i-1]-2*W[T-DT, i]))
        A1Ut[i, i+1] = - (1-zeta)*(eps_u*k/(h**2)-gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1]))
        A1Ut[i, i-1] = -(1-zeta)*(eps_u*k/(h**2)+gamma_u*k/(4*h**2)*(W[T-DT, i+1]-W[T-DT,i-1]))
    
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

    k = 1.0/float(M)
    h = 1.0/float(N)
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

    k = 1.0/float(M)
    h = 1.0/float(N)

    U = np.zeros((M*nb_sec,N))
    W = np.zeros((M*nb_sec,N))
    V = np.zeros((M*nb_sec,N))
    
    epsilon = 0.01

    # J = 0: SET INITIAL conditions
    n0 = [np.exp(-((1.0*x)/N)**2/epsilon) for x in range(N)]
    f0 = [(1.0 - np.exp(-((1.0*x)/N)**2/epsilon))*0.7 for x in range(N)]
    m0 = [0.5*np.exp(-((1.0*x)/N)**2/epsilon) for x in range(N)]
    U[0,:] = np.array(n0)
    #V[0,:] = np.array(m0)
    W[0,:] = np.array(f0)
    
    rhoU, rhoV, rhoW = [], [], []
    
    # J = 1..M
    for j in tqdm(range(1, M*nb_sec)): # for each time do
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
        if spec_log or rho_plot: 
            rhoU += [util.track_specrad(-la.inv(AU)@BU, j, '1dSWB_specrad_CU_z{0}'.format(zeta))]
            rhoV += [util.track_specrad(-la.inv(AV)@BV, j, '1dSWB_specrad_CV_z{0}'.format(zeta))]
            rhoW += [util.track_specrad(-la.inv(AW)@BW, j, '1dSWB_specrad_CW_z{0}'.format(zeta))]
    
    return U, V, W, AU, BU, AV, BV, rhoU, rhoV, rhoW





N_t = 100
M_t = 100
nb_sec = 20

print('Simulating with N = ', N_t, '  M = ', M_t, ' for ', nb_sec, ' seconds. ')
print('Zeta = ', zeta)
sys.stdout.flush()

run = False
anim_plot = False
rho_plot = False
Dthree_plot = False
spec_log = False
Dtwo_plot = True


if run:
    U, V, W, AUx, BUx, AVx, BVx, rhoUx, rhoVx, rhoWx = SB_solver_1D(N_t,M_t, nb_sec)

plt.clf()
plt.cla()
plt.close()

xx = np.arange(0,N_t)
yy = np.arange(0,M_t*nb_sec)
X, Y = np.meshgrid(xx,yy)

if Dthree_plot:
    fig3 = plt.figure(1)
    ax = Axes3D(fig3)
    ax.plot_surface(X,Y,V[0:])
    ax.plot_surface(X,Y,U)

if anim_plot:
    
    figa = plt.figure(2)
    figa, ax = plt.subplots()
    ln, = plt.plot(xx, U[0,:], 'b')
    ln1, = plt.plot(xx, V[0,:], 'r')
    ln2, = plt.plot(xx, W[0,:], 'green')
    
    def init():
        ax.set_xlim(0, N_t)
        ax.set_ylim(0, 1)
        return ln, ln1, ln2,
    
    def update(frame):
        ln.set_data(xx, U[frame,:])
        ln1.set_data(xx, V[frame,:])
        ln2.set_data(xx, W[frame,:])
    
        return ln, ln1, ln2,
    
    ani = FuncAnimation(figa, update, frames=np.array(range(0, M_t*nb_sec)),
                        init_func=init, blit=True)
    plt.show()

if Dtwo_plot:
    steps = 10
    fig = plt.figure(4)
    R, G, B = util.color_grads(steps)
    xx = np.array(range(0,N_t))
    for i in range(0, steps-1):
        cur_t = int(i*M_t*nb_sec/steps)
        plt.plot(xx, U[cur_t,:], color=B[i])
        plt.plot(xx, V[cur_t,:], color=R[i])
        plt.plot(xx, W[cur_t,:], color=G[i])
    plt.show()

    

if rho_plot:
    fig = plt.figure(3)
    xx = np.array(range(0, M_t*nb_sec-1))
    l1 = plt.plot(xx, rhoUx, color='blue')
    l2 = plt.plot(xx, rhoVx, color='red')
    l3 = plt.plot(xx, rhoWx, color='green')
    plt.legend(('rho(C_U)', 'rho(C_V)', 'rho(C_W)'))
    plt.show()