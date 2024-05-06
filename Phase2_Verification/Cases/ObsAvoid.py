import torch

from Cases.Case import *

class ObsAvoid(case):
    '''
    Define classical control case Obstacle Avoidance
    x0_dot = v sin(phi) + 0
    x1_dot = v cos(phi) + 0
    phi_dot = 0         + u
    '''
    def __init__(self):
        DOMAIN = [[-2, 2], [-2, 2], [-2, 2]]
        CTRLDOM = [[-2, 2]]
        discrete = False
        self.v = 1
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)

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

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        The safe region is a pole centered at (0,0,any) with radius 0.2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        hx = (x[:, 0]**2 + x[:, 1] ** 2) - 0.04
        return hx
