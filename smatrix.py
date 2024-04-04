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
    V0 = solve_eigenproblem_uniform(K_x, K_y, 1)[1] 

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