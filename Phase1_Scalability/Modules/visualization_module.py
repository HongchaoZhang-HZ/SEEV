from re import S
from matplotlib import pyplot as plt

class visualization_module:
    def __init__(self, shape):
        self.domain = shape

    def plot_samples(self, fig, sample_set) -> plt.Figure:
        x = sample_set[:, 0]
        y = sample_set[:, 1]
        fig.scatter(x, y)
        return fig
