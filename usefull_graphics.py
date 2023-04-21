import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def plot_2d_in_row(figures, titles=None, xlabels=None, ylabels=None,
                        show_colorbar=False, figsize=(6, 6), font_size=12, 
                        nrows=1, normalize=None, cmap='Spectral'):
    num_figures = len(figures)
    ncols = (num_figures + nrows - 1) // nrows

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    plt.rcParams.update({'font.size': font_size})

    for i, ax in enumerate(axs.flat):
        if i >= num_figures:
            break
        if normalize is not None:
            vmin = normalize[0]
            vmax = normalize[1]
            im = ax.imshow(figures[i], cmap=cmap, interpolation='none',
                           vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(figures[i], cmap=cmap, interpolation='none')
            ax.set_axis_off()
        if titles is not None:
            ax.set_title(titles[i])
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
        if show_colorbar:
            fig.colorbar(im, ax=ax)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate


def plot_3d_in_row(figures, titles=None, xlabels=None, ylabels=None, zlabels=None,
                   show_colorbar=False, figsize=(6, 6), font_size=12, nrows=1, 
                   normalize=None, exclude=None, cmap='Spectral', edgecolor=None):
    
    num_figures = len(figures)
    ncols = (num_figures + nrows - 1) // nrows

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, subplot_kw={'projection': '3d'})

    plt.rcParams.update({'font.size': font_size})
    
    # Ensure that a valid colormap is passed
    try:
        cmap_obj = cm.get_cmap(cmap)
    except ValueError:
        raise ValueError("Invalid colormap: {}".format(cmap))

    for i, ax in enumerate(np.ravel(axs)):
        if i >= num_figures:
            break
        data = figures[i]
        if exclude is not None:
            data = np.where(data < exclude, np.nan, data)
        if normalize is not None:
            norm = plt.Normalize(vmin=normalize[0], vmax=normalize[1])
            colors = cmap_obj(norm(data.flatten()))
        else:
            norm = plt.Normalize(vmin=np.min(data), vmax=np.max(data))
            colors = cmap_obj(data.flatten())
        if len(data.shape) == 3:
            filled = data > 0  # Create a boolean array indicating which voxels are filled
            surf = ax.voxels(filled, facecolors=colors.reshape(data.shape[0], data.shape[1], data.shape[2], -1), edgecolor=edgecolor)
        else:
            raise ValueError("Data must be 3D array")
        if titles is not None:
            ax.set_title(titles[i])
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
        if zlabels is not None:
            ax.set_zlabel(zlabels[i])
        if show_colorbar:
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


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
