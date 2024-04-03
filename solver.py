import numpy as np
import rcwa
import smatrix

class Device:

    def __init__(self) -> None:
        self.uniform_layers = []
        self.patterned_layers = []
        self.stored_convolution_matrices = []
        self.id_count = 0
        
    def set_k_space(self, Px, Py, k0_x, k0_y, M, N, alpha=np.pi/2):
        self.K_x, self.K_y, self.p, self.q = create_k_space(Px, Py, k0_x, k0_y, M, N, alpha)

    def add_layer_patterned(self, er_func):
        self.patterned_layer.append({'er_fourier': get_Fourier_coefficients(er_func), 'id': self.id_count})
        self.id_count += 1

    def add_layer_uniform(self, er):
        self.uniform_layers.append({'er': er, 'id': self.id_count})
        self.id_count += 1

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