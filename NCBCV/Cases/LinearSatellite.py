import torch
import numpy as np
from .Case import case

class LinearSat(case):
    '''
    Define classical control case Darboux
    '''
    def __init__(self):
        '''
        Define classical control case Darboux.
        The system is 2D open-loop nonlinear CT system
        '''
        DOMAIN = [[-5, 5], [-5, 5], [-5, 5], [-1, 1], [-1, 1], [-1, 1]]
        SSpace = [[-5, -5, -5, -1, -1, -1], [5, 5, 5, 1, 1, 1]]
        CTRLDOM = [[-2, 2], [-2, 2], [-2, 2]]
        discrete = False
        super().__init__(DOMAIN, CTRLDOM, discrete=discrete)
        self.SSpace = SSpace
        self.is_gx_linear = True
        self.is_fx_linear = True
        self.is_u_cons = False
        self.is_u_cons_interval = False
        # whether h(x) > 0 is safe
        self.pos_h_x_is_safe = True
        self.NChx = False
        # if b(x) > 0 is safe, then no need reverse (i.e. reverse_flag = False)
        # if b(x) > 0 is unsafe, then need to reverse (i.e. reverse_flag = True)
        self.reverse_flag = True
        

    def f_x(self, x):
        '''
        Control affine model f(x)
        f0 = x0 + 2*x0*x1
        f1 = -x0 + 2*x0^2 - x1^2
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        n = 0.5
        F = np.array([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,0],
                      [3*n**2,0,0,0,2*n,0],
                      [0,0,0,-2*n,0,0],
                      [0,0,0,-n**2,0,0]])
        fx = F @ x
        return fx

    def g_x(self, x):
        '''
        Control affine model g(x)=[0 0 0]'
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] output in R^n
        '''
        # gx = torch.zeros([self.DIM, 1])
        G = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,0,1]])
        return G

    def h_x(self, x):
        '''
        Define safe region C:={x|h_x(x) >= 0}
        :param x: [np.array/torch.Tensor] input state x in R^n
        :return: [np.array/torch.Tensor] scalar output in R
        '''
        hx = (x[0]**2 + x[1]**2 + x[2]**2) - 0.25**2
        # hx = abs(x[0]) + abs(x[1]) + abs(x[2]) - 0.25
        return hx
