import numpy as np
from rcwa import *
from smatrix import *

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
        W, V, eig = solve_eigenproblem_uniform(K_x, K_y, self.er)
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

        if er_conv is None: # compute and store if it has not been computed before
            er_conv = get_convolution_matrix(self.er_Fourier, M, N)
            er_conv_inv = np.linalg.inv(er_conv)
            self.convolution_matrices.append({'M': M, 'N': N, 'matrix': er_conv, 'matrix_inv': er_conv_inv})

        W, V, eig = solve_eigenproblem(K_x, K_y, er_conv, er_conv_inv)

        return build_layer_S_matrix(K_x, K_y, W, V, eig, k_0, self.L)
        
class Device:

    def __init__(self) -> None:
        self.layers = []
        self.id_count = 0

    def set_lattice(self, Px, Py, M, N, alpha=np.pi/2):
        self.M = M
        self.N = N
        self.alpha = alpha
        self.Px = Px
        self.Py = Py

    def set_incidence(self, theta, phi, n_inc):
        
        self.theta = theta
        self.phi = phi
        s_theta = np.sin(self.theta)
        c_phi = np.cos(self.phi)
        s_phi = np.sin(self.phi)
        self.k0_x = n_inc * s_theta * c_phi
        self.k0_y = n_inc * s_theta * s_phi

        self.K_x, self.K_y = create_k_space(self.Px, self.Py, self.k0_x, self.k0_y, self.M, self.N, self.alpha)
        
    def compute_RT(self, wl0, p_TE, p_TM, er_ref, er_trn):
        k_0 = 2 * np.pi / wl0
        s_theta = np.sin(self.theta)
        c_theta = np.cos(self.theta)
        c_phi = np.cos(self.phi)
        s_phi = np.sin(self.phi)
        c_inc_x = -p_TE * s_phi + p_TM * c_theta * c_phi 
        c_inc_y = p_TE * c_phi + p_TM * c_theta * s_phi
        c_inc_z = -p_TM * s_theta
        
        N = 1/np.sqrt(c_inc_x**2+c_inc_y**2+c_inc_z**2)

        s_inc = np.zeros(2*(2*self.M+1)*(2*self.N+1))
        s_inc[self.N*(2*self.M+1)+self.M] = c_inc_x / N
        s_inc[(2*self.M+1)*(2*self.N+1) + self.N*(2*self.M+1)+self.M] = c_inc_y / N
        k_z_inc = np.sqrt(er_ref - self.k0_x**2 - self.k0_y**2)

        S11, S12, S21, S22 = self.get_global_Smatrix(er_ref, er_trn, self.get_device_Smatrix(k_0))

        s_ref = S11 @ s_inc
        s_trn = S21 @ s_inc

        I = np.eye(self.K_x.shape[0])
        K_z_ref = -np.sqrt(er_ref*I - self.K_x**2 - self.K_y**2 + 0j)
        r_x = s_ref[0:(2*self.N+1)*(2*self.M+1)]
        r_y = s_ref[(2*self.N+1)*(2*self.M+1):2*(2*self.N+1)*(2*self.M+1)]
        r_z = -np.linalg.inv(K_z_ref) @ (self.K_x @ r_x + self.K_y @ r_y) 
        r2 = r_x*np.conjugate(r_x) + r_y*np.conjugate(r_y) + r_z*np.conjugate(r_z)

        K_z_trn = np.sqrt(er_trn*I - self.K_x**2 - self.K_y**2 +0j)
        t_x = s_trn[0:(2*self.N+1)*(2*self.M+1)]
        t_y = s_trn[(2*self.N+1)*(2*self.M+1):2*(2*self.N+1)*(2*self.M+1)]
        t_z = -np.linalg.inv(K_z_trn) @ (self.K_x @ t_x + self.K_y @ t_y)
        t2 = t_x*np.conjugate(t_x) + t_y*np.conjugate(t_y) + t_z*np.conjugate(t_z)

        R = np.real(-K_z_ref) / k_z_inc @ r2
        T = np.real(K_z_trn) / k_z_inc @ t2

        return R.reshape((2*self.N+1, 2*self.M+1)), T.reshape((2*self.N+1, 2*self.M+1))

    def add_layer_patterned(self, L, er_func):
        '''
        Inputs:
            er_func - Dielectric function of the layer.

        Returns a reference to the layer object added.
        '''
        layer = PatternedLayer(L, er_func)
        self.layers.append(layer)
        return layer

    def add_layer_uniform(self, L, er):
        '''
        Inputs:
            er - Uniform dielectric constant of the layer.

        Returns a reference to the layer object added.
        '''
        layer = UniformLayer(L, er)
        self.layers.append(layer)
        return layer
    
    def get_layer_by_name(self, name):
        found = None

        for layer in self.layers:
            if layer.name == name:
                found = layer

        return found
    
    def get_device_Smatrix(self, k_0):
        S = getInitialSMatrix(2*self.K_x.shape[0])

        for layer in self.layers:
            S_add = layer.get_Smatrix(k_0, self.K_x, self.K_y, self.M, self.N)
            S = starProduct(S_add, S)

        return S
    
    def get_global_Smatrix(self, er_ref, er_trn, S_device):

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
        A_trn = I + V0_inv @ V_trn
        B_trn = I - V0_inv @ V_trn
        A_trn_inv = np.linalg.inv(A_trn)
        S_trn = (B_trn @ A_trn_inv, 0.5*(A_trn - B_trn @ A_trn_inv @ B_trn), 2*A_trn_inv, -A_trn_inv @ B_trn)

        return starProduct(S_ref, starProduct(S_device, S_trn))