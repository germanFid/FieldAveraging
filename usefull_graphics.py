import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as mplstyle
import numpy as np
from matplotlib.colors import Normalize, ListedColormap

def plot_2d_in_row(figures, titles=None, xlabels=None, ylabels=None,
                        show_colorbar=False, figsize=(6, 6), font_size=12, 
                        nrows=1, normalize=None, cmap='Spectral'):
    num_figures = len(figures)
    ncols = (num_figures + nrows - 1) // nrows
    mplstyle.use('fast')
    fig, axs = plt.plot(figsize=figsize)

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


def plot_3d_voxels(figure, title=None, xlabel=None, ylabel=None, zlabel=None, 
                   show_colorbar=False, font_size=12, normalize=None, exclude=None, cmap='Spectral'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_size, y_size, z_size = figure.shape
    
    plt.rcParams.update({'font.size': font_size})
    # Ensure that a valid colormap is passed
    try:
        cmap_obj = plt.get_cmap(cmap)
    except ValueError:
        raise ValueError("Invalid colormap: {}".format(cmap))
    
    data = figure
    if exclude is not None:
        data = np.where(abs(data) < exclude, np.nan, data)
    if normalize is not None:
        norm = Normalize(vmin=normalize[0], vmax=normalize[1])
    else:
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    if len(data.shape) == 3:
        # Create a voxel grid and set the facecolors and edgecolors based on the normalized data
        ax.voxels(data, facecolors=cmap_obj(norm(data)), edgecolor='k')
    else:
        raise ValueError("Data must be 3D array")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    if show_colorbar:
        # Create a colorbar for the normalized data
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(data)
        plt.colorbar(mappable)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


def plot_3d_voxels_gpu(figure, title=None, xlabel=None, ylabel=None, zlabel=None, 
                       show_colorbar=False, font_size=12, normalize=None, exclude=None, cmap='Spectral'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_size, y_size, z_size = figure.shape
    
    plt.rcParams.update({'font.size': font_size})
    # Ensure that a valid colormap is passed
    try:
        cmap_obj = plt.get_cmap(cmap)
    except ValueError:
        raise ValueError("Invalid colormap: {}".format(cmap))
    
    data = cp.asarray(figure) # convert input array to CuPy array
    if exclude is not None:
        data = cp.where(cp.absolute(data) < exclude, cp.nan, data)
    if normalize is not None:
        norm = Normalize(vmin=normalize[0], vmax=normalize[1])
    else:
        norm = Normalize(vmin=cp.nanmin(data), vmax=cp.nanmax(data))
    if len(data.shape) == 3:
        # Create a voxel grid and set the facecolors and edgecolors based on the normalized data
        ax.voxels(data.get(), facecolors=cmap_obj(norm(data.get())), edgecolor='k') # convert CuPy array back to NumPy for plotting
    else:
        raise ValueError("Data must be 3D array")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        ax.set_zlabel(zlabel)
    if show_colorbar:
        # Create a colorbar for the normalized data
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(data.get()) # convert CuPy array back to NumPy for colorbar
        plt.colorbar(mappable)
    
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
