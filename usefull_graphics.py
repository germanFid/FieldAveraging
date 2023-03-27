import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_figures_in_row(figures, titles=None, xlabels=None, ylabels=None,
                        show_colorbar=False, figsize=(6, 6), font_size=12, nrows=1, normalize=None):
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
            im = ax.imshow(figures[i], cmap='Spectral', interpolation='none',
                           vmin=vmin, vmax=vmax)
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
