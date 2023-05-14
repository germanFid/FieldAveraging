import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
import numpy as np
from matplotlib.colors import Normalize


def plot_2d(figure, title=None, xlabel=None, ylabel=None,
            show_colorbar=False, figsize=(6, 6), font_size=12,
            normalize=None, extent=None, cmap='Spectral'):
    mplstyle.use('fast')
    fig, ax = plt.subplots()

    plt.rcParams.update({'font.size': font_size})
    if normalize is not None:
        vmin = normalize[0]
        vmax = normalize[1]
        im = ax.imshow(figure, cmap=cmap, interpolation='none',
                       vmin=vmin, vmax=vmax, extent=extent)
    else:
        im = ax.imshow(figure, cmap=cmap, interpolation='none', extent=extent)
    fig.set_size_inches(figsize)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if show_colorbar:
        fig.colorbar(im, ax=ax, shrink=1)
    return fig


def plot_2d_in_row(figures, titles=None, xlabels=None, ylabels=None,
                   show_colorbar=False, figsize=(6, 6), font_size=12,
                   nrows=1, normalize=None, extent=None, cmap='Spectral'):
    num_figures = len(figures)
    ncols = (num_figures + nrows - 1) // nrows
    mplstyle.use('fast')
    fig, axs = plt.subplots(nrows, ncols)

    plt.rcParams.update({'font.size': font_size})

    for i, ax in enumerate(axs.flat):
        if i >= num_figures:
            break
        if normalize is not None:
            vmin = normalize[0]
            vmax = normalize[1]
            im = ax.imshow(figures[i], cmap=cmap, interpolation='none',
                           vmin=vmin, vmax=vmax, extent=extent)
        else:
            im = ax.imshow(figures[i], cmap=cmap, interpolation='none', extent=extent)
        fig.set_size_inches(figsize)
        if titles is not None:
            ax.set_title(titles[i])
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
    if show_colorbar:
        fig.colorbar(im, ax=axs, shrink=1)
    return fig


def scatter_3d_array(data: np.ndarray, treeshold_up: float = None, treeshold_down: float = None,
                     normalize=None, title: str = None, xlabel: str = None, ylabel: str = None,
                     zlabel: str = None, colorbar: bool = False, cmap: str = "Spectral"):
    mplstyle.use('fast')
    n, m, k = data.shape
    x, y, z = np.meshgrid(
        np.linspace(0, n - 1, n), np.linspace(0, m - 1, m), np.linspace(0, k - 1, k)
    )
    if treeshold_up is not None:
        alphas = np.where(data < treeshold_up, 0, 1)
        if treeshold_down is not None:
            alphas = np.where(data > treeshold_down, 0, alphas)
    else:
        alphas = 1
    if normalize is not None:
        norm = Normalize(vmin=normalize[0], vmax=normalize[1])
    else:
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    sc = ax.scatter(x, y, z, s=100, c=data, marker='o', cmap=cmap, norm=norm, alpha=alphas)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    if colorbar:
        fig.colorbar(sc, shrink=.65)
    plt.show()
    plt.savefig("scatter_hue", bbox_inches='tight')


def create_gif(data, filename, name='', duration=100, vmin=None, vmax=None, figsize=(8, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data[0], cmap='Spectral', interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(name + str(1), fontsize=14)  # add name to the title of the figure
    fig.colorbar(im)  # add colorbar to the figure

    def update(i):
        im.set_data(data[i])
        ax.set_title(name + str(i + 1))
        return [im]

    ani = FuncAnimation(fig, update, frames=len(data), interval=duration, blit=True)
    ani.save(filename, writer='imagemagick')
