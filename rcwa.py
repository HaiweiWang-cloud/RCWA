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
    P, Q - indices of the Floquet modes.
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

    return np.diag(k_x.flatten()), np.diag(k_y.flatten()), p, q

def get_convolution_matrix(material_function, p, q):
    '''
    Inputs:
        material_function - sampled complex-valued function of the material over the lattice coordinate system. Size NxN, where N is a power of 2. 
        p, q - plane wave expansion index vectors
    '''
    N2 = material_function.size
    N_half = int(material_function.shape[0]/2)
    
    a_mn = (1/N2) * np.fft.fftshift(np.fft.fft2(material_function))

    convolution_matrix = np.zeros((p.size * q.size, p.size * q.size), dtype=complex)

    P, Q = np.meshgrid(p,q)

    for i, p_i in enumerate(p):
        for j, q_j in enumerate(q):
            convolution_matrix[(j)*p.size + i, :] = a_mn[q_j-Q+N_half,p_i-P+N_half].flatten()

    return convolution_matrix