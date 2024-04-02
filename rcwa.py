import numpy as np

def create_k_space(Px, Py, k0_x, k0_y, M, N, alpha=np.pi/2):
    '''
    Initialises the K matrices for the plane wave expansion

    Inputs:
    Px, Py: periodicities, the #1 reciprocal lattice vector is assumed to align with the x-axis
    alpha: angle between the first and second lattice vectors.
    k0_x, k0_y: x and y components of the Floquet wave vector, defined by incident angle of the incident wave.
    M: number of plane waves in X
    N: number of plane waves in Y

    Outputs:
    K_x, K_y - diagonal matrices with the wavevectors of the expansion rasterised.
    '''
    p = np.arange(-M, M+1)
    q = np.arange(-N, N+1)

    P, Q = np.meshgrid(p, q)

    if abs(alpha-np.pi/2) < 0.01:
        k_x = k0_x + 2*np.pi/Px * P
        k_y = k0_y + 2*np.pi/Py * Q
    else:
        k_x = k0_x + 2*np.pi/Px * P
        k_y = k0_y + 2*np.pi/Py * Q - 2*np.pi/Px/np.tan(alpha) * P 

    return np.diag(k_x.flatten()), np.diag(k_y.flatten())

def get_Fourier_coefficients(input, alpha=np.pi/2):
    '''
    Inputs:
        input - a square matrix corresponding to a function defined over the transformed orthogonal coordinates
    '''

    return output

def get_convolution_matrix(input, Gs):
    '''
    Inputs:
        
    '''
    return convolution_matrix