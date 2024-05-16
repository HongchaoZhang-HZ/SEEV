import torch

from .Case import case

class ObsAvoid(case):
    '''
    Define classical control case Obstacle Avoidance
    x0_dot = v sin(phi) + 0
    x1_dot = v cos(phi) + 0
    phi_dot = 0         + u
    '''
    def __init__(self):
        DOMAIN = [[-2, 2], [-2, 2], [-2, 2]]
        SSpace = [[-2, -2, -2], [2, 2, 2]]
        CTRLDOM = [[-2, 2]]
        discrete = False
        self.v = 1
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)
        self.SSpace = SSpace
        self.is_gx_linear = True
        self.is_fx_linear = False
        self.is_u_cons = True
        self.is_u_cons_interval = True
        self.pos_h_x_is_safe = True
        self.NChx = False
        self.reverse_flag = True
        
        # self.A = [-1, 1]
        self.A = []
        # self.c = [-2, -2]
        self.c = []

    def f_x(self, x):
        '''
        Control affine model f(x)
        f0 = v sin(phi)
        f1 = v cos(phi)
        f2 = 0
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''

        v = self.v
        x0_dot = v * np.sin(x[2])
        x1_dot = v * np.cos(x[2])
        phi_dot = 0
        x_dot = np.vstack([x0_dot, x1_dot, phi_dot])
        return x_dot

    def f_x_dreal(self, x):
        v = self.v
        x0_dot = v * dr.sin(x[2])
        x1_dot = v * dr.cos(x[2])
        phi_dot = 0
        return x0_dot, x1_dot, phi_dot
    
    def g_x(self, x):
        '''
        Control affine model g(x)=[0 0 1]'
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        g_x0 = 0
        g_x1 = 0
        g_phi = 1
        gx = np.vstack([g_x0, g_x1, g_phi])
        return gx
    
    def g_x_dreal(self, x):
        g_x0 = 0
        g_x1 = 0
        g_phi = 1
        return g_x0, g_x1, g_phi

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        The safe region is a pole centered at (0,0,any) with radius 0.2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        hx = (x[0]**2 + x[1]**2) - 0.04
        return hx
