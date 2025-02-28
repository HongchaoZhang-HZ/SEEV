import torch
import numpy as np
from .Case import case

class HighO(case):
    '''
    Define classical control case Darboux
    '''
    def __init__(self):
        '''
        Define classical control case Darboux.
        The system is 2D open-loop nonlinear CT system
        '''
        DOMAIN = [[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]]
        SSpace = [[-2, -2, -2, -2, -2, -2, -2, -2], [2, 2, 2, 2, 2, 2, 2, 2]]
        CTRLDOM = [[0, 0]]
        discrete = False
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)
        self.SSpace = SSpace
        self.is_gx_linear = True
        self.is_fx_linear = False
        self.is_u_cons = False
        self.is_u_cons_interval = False
        self.pos_h_x_is_safe = True
        self.NChx = False
        self.reverse_flag = True
        

    def f_x(self, x):
        # x^8 + 20 x^7 + 170 x^6 + 800 x^5 + 2273 x^4 + 3980 x^3 + 4180 x^2 + 2400 x + 576
        # is stable with roots in -1, -2, -3, -4
        x_dot = np.vstack([x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                           -20*x[7] - 170*x[6] - 800*x[5] - 2273*x[4] - 3980*x[3] - 4180*x[2] - 2400*x[1] - 576*x[0]])
        return x_dot

    def g_x(self, x):
        '''
        Control affine model g(x)=[0 0 0]'
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        # gx = torch.zeros([self.DIM, 1])
        gx = np.zeros([self.DIM, 1])
        return gx

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        # hx = 0.16 - np.sum((x+2) ** 2) 
        hx = np.sum((x+2) ** 2) - 3
        return hx
