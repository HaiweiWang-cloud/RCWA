import numpy as np
import rcwa
import smatrix

class UniformLayer:
    _ID = 0
    def __init__(self, L, er, name=None) -> None:
        if name:
            self.name = name
        else:
            self.name = "uniform"+str(self.__class__._ID)
        self.__class__._ID += 1

        self.L = L
        self.er = er

    def get_Smatrix(self, k_0, K_x, K_y, M, N):
        W, V, eig = solve_eignproblem_uniform(K_x, K_y, er)
        return build_layer_S_matrix(K_x, K_y, W, V, eig, k_0, self.L)

class PatternedLayer:
    _ID = 0
    def __init__(self, L, er_func, name=None) -> None:
        if name:
            self.name = name
        else:
            self.name = "patterned"+str(self.__class__._ID)
        self.__class__._ID += 1

        self.L = L
        self.er_func = er_func
        self.er_Fourier = get_Fourier_coefficients(er_func)
        self.convolution_matrices = []

    def change_er_func(self, er_func):
        self.er_func = er_func
        self.er_Fourier = get_Fourier_coefficients(er_func)
        self.convolution_matrices = [] # reset precomputed convolution matrices, for they are no longer valid.

    def get_Smatrix(self, k_0, K_x, K_y, M, N):
        er_conv = None
        for item in self.convolution_matrices: # find in stored
            if item["M"] == M and item["N"] == N:
                er_conv = item["matrix"]
                er_conv_inv = item["matrix_inv"]

        if not er_conv: # compute and store if it has not been computed before
            er_conv = get_convolution_matrix(self.er_Fourier, M, N)
            er_conv_inv = np.linalg.inv(er_conv)
            self.convolution_matrices.append({'M': M, 'N': N, 'matrix': er_conv, 'matrix_inv': er_conv_inv})

        W, V, eig = solve_eigenproblem(K_x, K_y, er_conv, er_conv_inv)

        return build_layer_S_matrix(K_x, K_y, W, V, eig, k_0, self.L)
        
class Device:

    def __init__(self) -> None:
        self.layers = []
        self.id_count = 0
        
    def set_k_space(self, Px, Py, k_0, k0_x, k0_y, M, N, alpha=np.pi/2):
        self.k_0 = k_0
        self.M = M
        self.N = N
        self.K_x, self.K_y = create_k_space(Px, Py, k0_x, k0_y, M, N, alpha)

    def add_layer_patterned(self, er_func):
        '''
        Inputs:
            er_func - Dielectric function of the layer.

        Returns a reference to the layer object added.
        '''
        layer = PatternedLayer(er_func)
        self.layers.append(layer)
        return layer

    def add_layer_uniform(self, er):
        '''
        Inputs:
            er - Uniform dielectric constant of the layer.

        Returns a reference to the layer object added.
        '''
        layer = UniformLayer(er)
        self.layers.append(layer)
        return layer
    
    def get_layer_by_name(self, name):
        found = None

        for layer in self.layers:
            if layer.name == name:
                found = layer

        return found
    
    def get_device_Smatrix(self):
        S = getInitialSMatrix(2*self.K_x.shape[0])

        for layer in self.layers:
            S_add = layer.getSmatrix(self.k_0, self.K_x, self.K_y, self.M, self.N)
            S = starProduct(S_add, S)

        return S
    
    def get_global_Smatrix(self, er_ref, er_trn):

        I, V0 = solve_eigenproblem_uniform(self.K_x, self.K_y, 1)[0:2]
        V0_inv = np.linalg.inv(V0)
        # Compute the reflection side S matrix
        V_ref = solve_eigenproblem_uniform(self.K_x, self.K_y, er_ref)[1]
        A_ref = I + V0_inv @ V_ref
        B_ref = I - V0_inv @ V_ref
        A_ref_inv = np.linalg.inv(A_ref)
        S_ref = (-A_ref_inv @ B_ref, 2*A_ref_inv, 0.5*(A_ref - B_ref @ A_ref_inv @ B_ref), B_ref @ A_ref_inv)

        # Compute the transmission side S matrix
        V_trn = solve_eigenproblem_uniform(self.K_x, self.K_y, er_trn)[1]
        A_trn = I + V_0_inv @ V_trn
        B_trn = I - V_0_inv @ V_trn
        A_trn_inv = np.linalg.inv(A_trn)
        S_trn = (B_trn @ A_trn_inv, 0.5*(A_trn - B_trn @ A_trn_inv @ B_trn), 2*A_trn_inv, -A_trn_inv @ B_trn)

        return starProduct(S_ref, starProduct(S_device, S_trn))

def computeIncidentWave(wl, n_inc, theta, phi, p_TE, p_TM):
    '''
    Inputs:
        Incident wave parameters
            wl: vacuum wavelength, arbitrary units
            n_inc: incident medium refractive index
            theta: incident angle, in radians
            phi: azimuthal angle, in radians
            p_TE: TE field amplitude
            p_TM: TM field amplitude
    
    Output:
        Incident wavevectors and incident mode coefficients (k_x, k_y, c_inc_x, c_inc_y)
    '''
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    k_x = n_inc * s_theta * c_phi
    k_y = n_inc * s_theta * s_phi
    c_inc_x = -p_TE * s_phi + p_TM * c_theta * c_phi 
    c_inc_y = p_TE * c_phi + p_TM * c_theta * s_phi
    c_inc_z = -p_TM * s_theta

    N = 1/np.sqrt(c_inc_x**2+c_inc_y**2+c_inc_z**2)

    return 2*np.pi / wl, k_x, k_y, c_inc_x/N, c_inc_y/N