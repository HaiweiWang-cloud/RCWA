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

def get_Fourier_coefficients(material_function):
    '''
    Inputs:
        material_function - sampled complex-valued function of the material over the lattice coordinate system. Size NxN, where N is a power of 2. 
        
    Output:
        Fourier coefficients of the material function
    '''
    return (1/material_function.size) * np.fft.fftshift(np.fft.fft2(material_function))


def get_convolution_matrix(a_mn, M, N):
    '''
    Inputs:
        a_mn - Fourier coefficients of the material function. Size NxN, where N is a power of 2. 
        M, N - number of plane waves to expand to.

    Output:
        convolution_matrix - convolution matrix over (p,q) of the Fourier space of the material function.
    '''
    
    N_half = int(a_mn.shape[0]/2)

    p = np.arange(-M, M+1)
    q = np.arange(-N, N+1)

    convolution_matrix = np.zeros((p.size * q.size, p.size * q.size), dtype=complex)

    P, Q = np.meshgrid(p,q)

    for i, p_i in enumerate(p):
        for j, q_j in enumerate(q):
            convolution_matrix[(j)*p.size + i, :] = a_mn[q_j-Q+N_half,p_i-P+N_half].flatten()

    return convolution_matrix

def solve_eigenproblem(K_x, K_y, conv_er, conv_er_inv):
    '''
    Calculates the eigenmodes and eigenvalues of the propagation through a patterend layer.

    Inputs:
        K_x, K_y - Wavevectors of the plane wave expansion, rasterised.
        conv_er - convolution matrix of the relative permittivity
        conv_er_inv - inverse of the convolution matrix of the relative permittivity

    Outputs:
        W - Electric field amplitudes of the eigenmodes, stored in each column, with their respective propagation constant.
        V - Magnetic field amplitudes ...
        prop_constants - Ordered vector of propagation constants corresponding to each eigenmode.
    '''
    I = np.eye(K_x.shape[0])
    P = np.vstack((np.hstack((K_x @ conv_er_inv @ K_y, I - K_x @ conv_er_inv @ K_x)), np.hstack((K_y @ conv_er_inv @ K_y - I, -K_y @ conv_er_inv @ K_x))))
    Q = np.vstack((np.hstack((K_x @ K_y, conv_er - K_x**2)), np.hstack((K_y**2 - conv_er, -K_y @ K_x))))
    omega = P @ Q

    eig_values, W = np.linalg.eig(omega)
    prop_constants = np.sqrt(eig_values)
    V = Q @ W @ np.diag(1/prop_constants)

    return W, V, prop_constants

def solve_eigenproblem_uniform(K_x, K_y, er):
    '''
    Calculates the eigenmodes and eigenvalues of the propagation through a homogeneous layer.

    Inputs:
        K_x, K_y - Wavevectors of the plane wave expansion, rasterised.
        er - relative permittivity of the layer

    Outputs:
        V - Magnetic field amplitudes of each mode, the electric field amplitudes are simply 1.
        prop_constants - Ordered vector of propagation constants corresponding to each eigenmode.
    '''
    I = np.conjugate(np.eye(K_x.shape[0]))
    Z = np.zeros(K_x.shape)

    Q = np.vstack((np.hstack((K_x @ K_y, I*er - K_x**2)), np.hstack((K_y**2 - I*er, -K_y @ K_x))))
    K_z = np.emath.sqrt(er*I - K_x**2 - K_y**2)
    prop_constants = np.vstack((np.hstack((1j*K_z, Z)), np.hstack((Z, 1j*K_z))))
    V = Q @ np.linalg.inv(prop_constants)

    return np.eye(2*K_x.shape[0]), V, np.diag(prop_constants)

