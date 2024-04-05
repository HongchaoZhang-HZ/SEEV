from re import S
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch

class visualization_module:
    def __init__(self, shape):
        self.DOMAIN = shape
        self.DIM = len(shape)
        self.PLOT_LEN_B = [100, 100]
        self.INIT = [[0, 1], [1, 2]] # set the initial in super-rectangle
        self.INIT_SHAPE = 1 # 1 for rectangle; 2 for cycle
        self.SUB_INIT = []
        self.SUB_INIT_SHAPE = []
        self.UNSAFE = [[-2, 0], [-1.5, 1.5]] # the the unsafe in super-rectangle
        self.UNSAFE_SHAPE = 3 # parabola

    def plot_samples(self, fig, sample_set) -> plt.Figure:
        x = sample_set[:, 0]
        y = sample_set[:, 1]
        fig.scatter(x, y)
        return fig
    
    def heatmap(nx, ny, model, shape):
        fig, ax = plt.subplots()
        vx, vy = torch.meshgrid(nx, ny)
        data = np.dstack([vx.reshape([shape[0], shape[1], 1]), vy.reshape([shape[0], shape[1], 1])])
        data = torch.Tensor(data.reshape(shape[0] * shape[1], 2))
        output = (model.forward(data)).detach().numpy()
        z = output.reshape(shape)
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        c = ax.pcolormesh(vx, vy, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('pcolormesh')
        # set the limits of the plot to the limits of the data
        ax.axis([vx.min(), vx.max(), vy.min(), vy.max()])
        fig.colorbar(c, ax=ax)
        for i in range(100):
            for j in range(100):
                if np.linalg.norm(z[i][j])<0.003:
                    fig.scatter(nx[i], ny[j])
        return fig
    
    def gen_plot_data(self, region, len_sample):
        grid_sample = [torch.linspace(region[i][0], region[i][1], int(len_sample[i])) for i in range(self.DIM)] # gridding each dimension
        mesh = torch.meshgrid(grid_sample) # mesh the gridding of each dimension
        flatten = [torch.flatten(mesh[i]) for i in range(len(mesh))] # flatten the list of meshes
        plot_data = torch.stack(flatten, 1) # stack the list of flattened meshes
        return plot_data
    
    def plot_boundary(self, model): # barrier boundary: contour plotting
        barrier_plot_nn_input = self.gen_plot_data(self.DOMAIN, self.PLOT_LEN_B)
        # apply the nn model but do not require gradient
        with torch.no_grad():
            barrier_plot_nn_output = model(barrier_plot_nn_input).reshape(self.PLOT_LEN_B[1], self.PLOT_LEN_B[0]) # y_size * x_size
        plot_Z = barrier_plot_nn_output.numpy()
        plot_sample_x = np.linspace(self.DOMAIN[0][0], self.DOMAIN[0][1], self.PLOT_LEN_B[0])
        plot_sample_y = np.linspace(self.DOMAIN[1][0], self.DOMAIN[1][1], self.PLOT_LEN_B[1])
        plot_X, plot_Y = np.meshgrid(plot_sample_x, plot_sample_y)
        #plt.contourf(plot_X, plot_Y, plot_Z, [0], color='k')
        barrier_contour = plt.contour(plot_X.T, plot_Y.T, plot_Z, [-self.TOL_BOUNDARY, 0, self.TOL_BOUNDARY], \
            linewidths = 3, colors=('k', 'b', 'y'))
        plt.clabel(barrier_contour, fontsize=20, colors=('k', 'b', 'y'))
        return barrier_contour
    
    def plot_init(init_range, init_shape):
        if init_shape == 1: # rectangle
            init = matplotlib.patches.Rectangle((init_range[0][0], init_range[1][0]), \
                init_range[0][1] - init_range[0][0], init_range[1][1] - init_range[1][0], facecolor='green')
        if init_shape == 2: # circle
            init = matplotlib.patches.Circle(((init_range[0][1] +  init_range[0][0]) / 2.0, \
                    (init_range[1][1] + init_range[1][0]) / 2.0), (init_range[1][1] - init_range[1][0]) / 2.0, facecolor='green')
        return init

    def plot_unsafe(unsafe_range, unsafe_shape):
        if unsafe_shape == 1: # rectangle
            unsafe = matplotlib.patches.Rectangle((unsafe_range[0][0], unsafe_range[1][0]), \
                unsafe_range[0][1] - unsafe_range[0][0], unsafe_range[1][1] - unsafe_range[1][0], facecolor='red')
        elif unsafe_shape == 2: # circle
            unsafe = matplotlib.patches.Circle(((unsafe_range[0][1] + unsafe_range[0][0]) / 2.0, \
                    (unsafe_range[1][1] + unsafe_range[1][0]) / 2.0), (unsafe_range[1][1] - unsafe_range[1][0]) / 2.0, facecolor='red')
        else: # a parabola?
            y = np.linspace(-np.sqrt(2), np.sqrt(2), 1000)
            x = - y ** 2
            unsafe = plt.fill(x, y, 'r')

        return unsafe

    def plot_barrier_2d(self, model):
        fig, ax = plt.subplots()
        boundary = self.plot_boundary(model) # plot boundary of barrier function
        
        # plot sub_init
        if len(self.SUB_INIT) == 0:
            init = self.plot_init(self.INIT, self.INIT_SHAPE) # plot initial
            ax.add_patch(init)
        else:
            for i in range(len(self.SUB_INIT)):
                init = self.plot_init(self.SUB_INIT[i], self.SUB_INIT_SHAPE[i]) # plot initial
                ax.add_patch(init)

        # plot sub_unsafe
        if len(self.SUB_UNSAFE) == 0:
            unsafe = self.plot_unsafe(self.UNSAFE, self.UNSAFE_SHAPE) # plot unsafe
            if self.UNSAFE_SHAPE == 1 or self.UNSAFE_SHAPE == 2:
                ax.add_patch(unsafe)
        else:
            for i in range(len(self.SUB_UNSAFE)):
                unsafe = self.plot_unsafe(self.SUB_UNSAFE[i], self.SUB_UNSAFE_SHAPE[i]) # plot unsafe
                ax.add_patch(unsafe)

        plt.axis([self.DOMAIN[0][0], self.DOMAIN[0][1], self.DOMAIN[1][0], self.DOMAIN[1][1]])
        # plt.axis('equal') ##PLOT_VEC_SCALE = None
        plt.show()


    # def plot_scatter(): # scatterring sample points
    #     scatter_plot_nn_input = gen_plot_data(self.DOMAIN, superp.PLOT_LEN_P)
    #     x_values = (scatter_plot_nn_input[:, 0]).numpy()
    #     y_values = (scatter_plot_nn_input[:, 1]).numpy()
    #     scattering_points = plt.scatter(x_values, y_values)
    #     return scattering_points

    # def plot_vector_field(): # vector field
    #     vector_plot_nn_input = gen_plot_data(self.DOMAIN, superp.PLOT_LEN_V)
    #     vector_field = self.vector_field(vector_plot_nn_input)

    #     vector_x_values = (vector_field[:, 0]).numpy()
    #     vector_y_values = (vector_field[:, 1]).numpy()
    #     vector_x_positions = (vector_plot_nn_input[:, 0]).numpy()
    #     vector_y_positions = (vector_plot_nn_input[:, 1]).numpy()

    #     vector_plot = plt.quiver(vector_x_positions, vector_y_positions, vector_x_values, vector_y_values, \
    #                     color='pink', angles='xy', scale_units='xy', scale=superp.PLOT_VEC_SCALE)
    #     return vector_plot