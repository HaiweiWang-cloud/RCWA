import numpy as np
import rcwa

def getInitialSMatrix(N):
    '''
    Inputs:
        N - number of modes in the problem
    '''
    S11 = np.zeros(N)
    S12 = np.identity(N)
    S22 = np.zeros(N)
    S21 = np.identity(N)

    S = [S11, S12, S21, S22]

    return S

def build_layer_S_matrix(K_x, K_y, W, V, prop_constants, k_0, L):
    '''
    Compute the scattering matrix using the eigenmodes of the layer. 
    Assuming the layer is sandwiched between two zero-thickness homogeneous layers of air.

    Inputs: 
        K_x, K_y - wavevector components of the plane wave expansion, rasterised.
        W, V - Electric and magnetic eigenvector matrices.
        prop_constants - array of eigenvalues
        k_0 - free space wavevector
        L - layer thickness

    Outputs:
        Ordered tuple of S-matrix components (S11, S12, S21, S22), here by symmetry S12 = S21, S11 = S22.
    '''
    I = np.eye(W.shape)
    V0 = solve_eigenproblem_uniform(K_x, K_y, 1)[0] 

    X = np.diag(np.exp(-prop_constants*k_0*L))
    W_inv = np.linalg.inv(W)
    V_inv = np.linalg.inv(V)
    A = W_inv + V_inv @ V_0
    A_inv = np.linalg.inv(A) 
    B = W_inv - V_inv @ V_0
    F = np.linalg.inv(A - X @ B @ A_inv @ X @ B) 
    S11 = F @ (X @ B @ A_inv @ X @ A - B)
    S12 = F @ X @ (A - B @ A_inv @ B)
   
    return (S11, S12, S12, S11)

def starProduct(S_a, S_b):
    '''
        Inputs:
            S_a, S_b: List containing the S-matrices
            [S11, S12, S21, S22]

        Output:
            S_ab: the star product of the two S-matrices
    '''
    N = S_a.shape[0]
    I = np.identity(N)
    F = S_a[1] @ np.linalg.inv(I - S_b[0] @ S_a[3])
    D = S_b[2] @ np.linalg.inv(I - S_a[3] @ S_b[0])

    S11 = S_a[0] + F @ S_b[0] @ S_a[2]
    S12 = F @ S_b[1]
    S21 = D @ S_a[2]
    S22 = S_b[3] + D @ S_a[3] @ S_b[1]

    return [S11, S12, S21, S22]

def computeSMatrixUniform(k_x, k_y, er, ur, k_0, L):
    '''
    Inputs:
        parameters required to calculate the S-matrix for a given uniform layer
        k_x, k_y: normalised incident wavevector components
        er: relative permittivity
        ur: relative permeability
        k0: free space wavevector
        L: thickness of the layer

    Output:
        S: the scattering matrix [S11, S12, S21, S22]
    '''
    I = np.identity(2)
    
    k_z_0 = np.sqrt(1-k_x**2-k_y**2)
    Q_0 = np.array([[k_x*k_y, 1 - k_x**2], [k_y**2-1, -k_x*k_y]])
    omega_0_inv = -1j/k_z_0*I
    V_0 = Q_0 @ omega_0_inv

    k_z_i = np.sqrt(ur*er-k_x**2-k_y**2)
    Q_i = 1/ur * np.array([[k_x*k_y, ur*er - k_x**2], [k_y**2-ur*er, -k_x*k_y]])
    omega_i_inv = -1j/k_z_i*I
    V_i = Q_i @ omega_i_inv
    V_i_inv = np.linalg.inv(V_i)
    X = np.array([[np.exp(1j*k_z_i*k_0*L), 0], [0, np.exp(-1j*k_z_i*k_0*L)]])
    
    A = I + V_i_inv @ V_0
    A_inv = np.linalg.inv(A) 
    B = I - V_i_inv @ V_0
    F = np.linalg.inv(A - X @ B @ A_inv @ X @ B) 
    S11 = F @ (X @ B @ A_inv @ X @ A - B)
    S22 = S11
    S12 = F @ X @ (A - B @ A_inv @ B)
    S21 = S12

    return [S11, S12, S21, S22]

def computeGlobalSMatrix(k_x, k_y, N, er_ref, ur_ref, er_trn, ur_trn, S_device):
    '''
    Inputs:
        k_x, k_y: normalised incident wavevector components
        N: number of modes
        er_ref, ur_ref: reflection side material
        er_trn, ur_trn: transmission side material
        S_device: S matrix of multilayered device

    Outputs:
        S_global: the global scattering matrix
    '''
    I = np.identity(N)

    k_z_0 = np.sqrt(1-k_x**2-k_y**2+0j)
    Q_0 = np.array([[k_x*k_y, 1 - k_x**2], [k_y**2-1, -k_x*k_y]])
    omega_0_inv = -1j/k_z_0*I
    V_0_inv = np.linalg.inv(Q_0 @ omega_0_inv)

    k_z_trn = np.sqrt(ur_trn*er_trn-k_x**2-k_y**2+0j)
    Q_trn = 1/ur_trn * np.array([[k_x*k_y, ur_trn*er_trn - k_x**2], [k_y**2-ur_trn*er_trn, -k_x*k_y]])
    omega_trn_inv = -1j/k_z_trn*I
    V_trn = Q_trn @ omega_trn_inv
    
    k_z_ref = np.sqrt(ur_ref*er_ref-k_x**2-k_y**2+0j)
    Q_ref = 1/ur_ref * np.array([[k_x*k_y, ur_ref*er_ref - k_x**2], [k_y**2-ur_ref*er_ref, -k_x*k_y]])
    omega_ref_inv = -1j/k_z_ref*I
    V_ref = Q_ref @ omega_ref_inv

    A_ref = I + V_0_inv @ V_ref
    B_ref = I - V_0_inv @ V_ref
    A_ref_inv = np.linalg.inv(A_ref)
    S11_ref = -A_ref_inv @ B_ref
    S12_ref = 2*A_ref_inv
    S21_ref = 0.5*(A_ref - B_ref @ A_ref_inv @ B_ref)
    S22_ref = B_ref @ A_ref_inv
    S_ref = [S11_ref, S12_ref, S21_ref, S22_ref]

    A_trn = I + V_0_inv @ V_trn
    B_trn = I - V_0_inv @ V_trn
    A_trn_inv = np.linalg.inv(A_trn)
    S11_trn = B_trn @ A_trn_inv
    S12_trn = 0.5*(A_trn - B_trn @ A_trn_inv @ B_trn)
    S21_trn = 2*A_trn_inv
    S22_trn = -A_trn_inv @ B_trn
    S_trn = [S11_trn, S12_trn, S21_trn, S22_trn]

    return starProduct(S_ref, starProduct(S_device, S_trn))