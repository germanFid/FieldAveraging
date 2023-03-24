import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


def plot_figures_in_row(figures, titles=None, xlabels=None, ylabels=None, show_axes=True, show_colorbar=False, figsize=(6, 6), font_size=12, nrows=1, normalize=None):
    # Determine the number of figures in the list
    num_figures = len(figures)
    # Determine the number of rows and columns of subplots
    ncols = num_figures // nrows
    if num_figures % nrows > 0:
        ncols += 1

    # Create a new figure with a specified size and subplot layout
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Set font size for all elements in the figure
    plt.rcParams.update({'font.size': font_size})

    # Add each figure to a subplot
    for i in range(num_figures):
        row = i // ncols
        col = i % ncols
        if nrows == 1:
            ax = axs[col]
        else:
            ax = axs[row, col]
        if show_axes:
            if normalize is not None:
                im = ax.imshow(figures[i], cmap='Spectral', interpolation='none', vmin=normalize[0], vmax=normalize[1])
            else:
                im = ax.imshow(figures[i], cmap='Spectral')
        else:
            if normalize is not None:
                im = ax.imshow(figures[i], cmap='Spectral', interpolation='none', vmin=normalize[0], vmax=normalize[1])
                ax.set_axis_off()
            else:
                im = ax.imshow(figures[i], cmap='Spectral', interpolation='none')
                ax.set_axis_off()
        if titles is not None:
            ax.set_title(titles[i])
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
        if show_colorbar:
            fig.colorbar(im, ax=ax)

    # Adjust the spacing between subplots and show the figure
    plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(hspace=0.05)
    return fig


def create_gif(data, filename, name='', duration=100, vmin=None, vmax=None, figsize=(8,10)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data[0], cmap='Spectral', interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(name + str(1), fontsize=14) # add name to the title of the figure
    fig.colorbar(im)  # add colorbar to the figure
    animations = []

    def update(i):
        im.set_data(data[i])
        ax.set_title(name + str(i + 1))
        return [im]

    ani = FuncAnimation(fig, update, frames=len(data), interval=duration, blit=True)
    ani.save(filename, writer='imagemagick')