import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from math import floor, ceil, log


def plot_2d(xdata, ydata, data, ylabel=None, xlim=None, ylim=None, invert_x=False, invert_y=True, colorbar=True,
            tick_params=None, colorbar_params=None, title=None, fontsize_title=None, vmin=0, vmax=None,
            data_transpose=True, data_flipud=False, data_fliplr=False, plotbox_position=None, fontsize_labels=11):
    """
        2d-plotting interface
    :param xdata: x-axis data
    :param ydata: y-axis data
    :param data: 2d-data
    :param ylabel: y-axis label
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param invert_x: Set True to invert x-axis
    :param invert_y: Set True to invert y-axis
    :param colorbar: Set True to plot colorbar
    :param tick_params: Dict with kwargs for plt.tick_params if not None
    :param colorbar_params: Dict with kwargs for the colorbar if not None
    :param title: plot title
    :param fontsize_title: fontisze of the plot title
    :param vmin: Specify minimum of the color range
    :param vmax: Specify maximum of the color range
    :param data_transpose: Transpose data array (also flips x-axis and y-axis)
    :param data_flipud: Flips data w.r.t x-axis
    :param data_fliplr: Flips data w.r.t y-axis
    :param plotbox_position: Set the position of the plot box and adjust if space left and right of it is desired
    :param fontsize_labels: Fontsize of the axis labels
    """
    tick_params_default = {'direction': 'in', 'bottom': True, 'top': True, 'left': True, 'right': True}
    colorbar_params_default = {'orientation': 'vertical', 'aspect': 15, 'pad': 1.8}

    xlabel = 't'
    xticks_int = True
    xticks_nbins = 2
    yticks_int = True
    yticks_nbins = 2
    tick_params = tick_params if tick_params is not None else tick_params_default
    disable_minor_ticks = True

    colorbar = None if not colorbar else \
        (colorbar_params if colorbar_params is not None else colorbar_params_default)
    disable_colorbar_minortics = True
    cmap = 'YlOrBr'

    # Bring ydata and xdata in the correct format if they are not already:
    if len(ydata) == data.shape[0]:
        dt_y = ydata[-1] - ydata[-2]
        ygrid = np.concatenate((ydata, np.array([ydata[-1] + dt_y])))
    elif len(ydata) == data.shape[0]+1:
        ygrid = ydata
    else:
        raise AssertionError('Incompatible shape of ydata and data_2d')
    if len(xdata) == data.shape[1]:
        dt_y = xdata[-1] - xdata[-2]
        xgrid = np.concatenate((xdata, np.array([xdata[-1] + dt_y])))
    elif len(xdata) == data.shape[1]+1:
        xgrid = xdata
    else:
        raise AssertionError('Incompatible shape of xdata and data_2d')
    # Bring 2d data in the desired format
    if data_transpose:
        data = data.T
        xgrid, ygrid = ygrid, xgrid
    if data_flipud:
        data = np.flipud(data)
    if data_fliplr:
        data = np.fliplr(data)

    # get position of the lower left corner of the plot-box
    ax = plt.gca()
    box = ax.get_position()

    # set the position of the plotbox and adjust if space left and right of it is wanted
    if plotbox_position is None:

        # plotbox is multiplied to the current position, so array of 1's has no effect
        # ax.set_position needs an array like: [x0, y0, width, height]
        plotbox_position = [1, 1, 1, 1]

        # adjust height of plotbox to keep the plotbox-ratio
        plotbox_position[3] = plotbox_position[2]

    # set title, xlabel, ylabel, xscale, yscale
    if plt.title is not None:
        plt.title(title, fontdict={'fontsize': fontsize_title})

    # set title, xlabel, ylabel, xscale, yscale
    if plt.title is not None:
        plt.title(title, fontdict={'fontsize': fontsize_title})

    # set the xlabel at the wanted position
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)

    # set the ylabel at the wanted position
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize_labels)

    # set the limits for the x-axis
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    ax.xaxis.set_major_locator(MaxNLocator(integer=xticks_int, nbins=xticks_nbins))

    # set the limits for the y-axis
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    ax.yaxis.set_major_locator(MaxNLocator(integer=yticks_int, nbins=yticks_nbins))

    if invert_y:
        top = ygrid[0]
        bottom = min(ygrid[-1], ylim[1] if ylim is not None else ygrid[-1])
    else:
        top = min(ygrid[-1], ylim[1] if ylim is not None else ygrid[-1])
        bottom = ygrid[0]
    if invert_x:
        left = min(xgrid[-1], xlim[1] if xlim is not None else xgrid[-1])
        right = xgrid[0]
    else:
        left = xgrid[0]
        right = min(xgrid[-1], xlim[1] if xlim is not None else xgrid[-1])

    # Set box extent of the axes:
    extent = [left, right, bottom, top]
    ax.axis(extent)

    # Plotting core:
    X, Y = np.meshgrid(xgrid, ygrid)

    data_min = data.min()
    plt.pcolormesh(X, Y, data, cmap=cmap, vmin=data_min if vmin is None else vmin,
                   vmax=data.max() if vmax is None else vmax)

    if colorbar is not None:
        cb = plt.colorbar(**colorbar)


    # set the position of the plot box
    ax.set_position([box.x0*plotbox_position[0], box.y0*plotbox_position[1],
                    box.width*plotbox_position[2], box.height*plotbox_position[3]])

    # set ticks to custom directions or by default all inside the plot
    if tick_params is not None:
        plt.tick_params(**tick_params)

    if disable_minor_ticks:
        plt.minorticks_off()

    if colorbar is not None:
        if disable_colorbar_minortics:
            cb.ax.tick_params(which='minor', length=0)

    ax.tick_params(axis='x', pad=7)


def plot_1d(xdata, ydata, yticks_tilde, yticks_labels=None, ylabel=None, xlim=None, ylim=None,
            tick_params=None, title=None, fontsize_title=None,
            plotbox_position=None, fontsize_labels=11,
            legend=(None,), legendloc='bottom right', fontsize_legend=None, linewidth=(1.0,), linestyle=('-',),
            color_list=('blue', 'orange'), xscale='linear', yscale='log'):
    """
        1d-plotting interface
    :param xdata: x-axis data
    :param ydata: y-axis data (possibly multiple as list with ydata index: [(0, arr1), (1, arr2), ...])
    :param yticks_tilde: Points on the y-axis where ytick-labels should be set
    :param yticks_labels: label-strings for the ytics_tilde labels
    :param ylabel: label of the y-axis
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param tick_params: Dict with kwargs for plt.tick_params if not None
    :param title: Title of the plot
    :param fontsize_title: Fontsize of the title
    :param plotbox_position: Set the position of the plot box and adjust if space left and right of it is desired
    :param fontsize_labels: Fontsize of the axis labels
    :param legend: Tuple/List of legend descriptions, pass (None, ) if no legend is desired
    :param legendloc: Location of the legend
    :param fontsize_legend: Fontsize of the legend
    :param linewidth: Linewidth of the plots(float)
    :param linestyle: Matplotlib linestyle
    :param color_list: Tuple/List of colors in the rotation
    :param xscale: Scale of the x-axis ('linear', 'log', ...)
    :param yscale: Scale of the y-axis ('linear', 'log', ...)
    """
    tick_params_default = {'direction': 'in', 'bottom': True, 'top': True, 'left': True, 'right': True}

    xlabel = 't'
    xticks_int = True
    xticks_nbins = 2
    tick_params = tick_params if tick_params is not None else tick_params_default
    disable_minor_ticks = True

    ax = plt.gca()
    box = ax.get_position()

    # set the position of the plotbox and adjust if space left and right of it is wanted
    if plotbox_position is None:

        # plotbox is multiplied to the current position, so array of 1's has no effect
        # ax.set_position needs an array like: [x0, y0, width, height]
        plotbox_position = [1, 1, 1, 1]

        # adjust height of plotbox to keep the plotbox-ratio
        plotbox_position[3] = plotbox_position[2]

    # set title, xlabel, ylabel, xscale, yscale
    if title is not None:
        plt.title(title, fontdict={'fontsize': fontsize_title})

    # set the xlabel at the wanted position
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)

    # set the xlabel at the wanted position
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize_labels)

    # set the xscale and the yscale
    plt.xscale(xscale)
    plt.yscale(yscale)

    # set the limits for the x-axis
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    ax.xaxis.set_major_locator(MaxNLocator(integer=xticks_int, nbins=xticks_nbins))

    # set the limits for the y-axis
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    if yticks_labels is not None:
        plt.yticks(yticks_tilde, labels=yticks_labels)

    for id, ydat in ydata:
        xdat = xdata
        plt.plot(xdat, ydat, linewidth=linewidth[id % len(linewidth)],
                 linestyle=linestyle[id % len(linestyle)],
                 color=color_list[id % len(color_list)], label=legend[id % len(legend)])

    # set the legend if at least one not None value was passed
    plot_legend = [element for element in legend if element is not None]
    if plot_legend:
        plt.legend(loc=legendloc, prop={'size': fontsize_legend})

    # set the position of the plot box
    ax.set_position([box.x0 * plotbox_position[0], box.y0 * plotbox_position[1],
                     box.width * plotbox_position[2], box.height * plotbox_position[3]])

    # set ticks to custom directions or by default all inside the plot
    if tick_params is not None:
        plt.tick_params(**tick_params)

    if disable_minor_ticks:
        plt.minorticks_off()

    ax.tick_params(axis='x', pad=7)


def multiplot(root, plot_name, nof_coefficients, times, chain_ranks, star_ranks, chain_size, star_size, rel_diff):
    """
        Main plotting interface
    :param root: Root folder in which the plot is saved
    :param plot_name: Name of the plot
    :param nof_coefficients: Number of bath sites
    :param times: Time-grid
    :param chain_ranks: Bond dimensions in the chain geometry
    :param star_ranks: Bond dimensions in the star geometry
    :param chain_size: TN-size of the chain
    :param star_size: TN-size of the star
    :param rel_diff: Relative difference between chain and star
    """
    # set global matplotlb rc parameters:
    rc = {'font.family': "serif",
          'font.weight': "normal",
          'font.size': 11,
          'xtick.major.width': 1,
          'xtick.major.size': 3,
          'ytick.major.width': 1,
          'ytick.major.size': 3,
          'axes.linewidth': 1,
          'legend.frameon': True,
          'legend.framealpha': 0}
    plt.figure(1)
    plt.rcParams.update(rc)
    fig = plt.gcf()
    nof_rows, nof_cols = 1, 4
    fig_format = 'pdf'

    # get some parameters to adjust the suplots
    adjust_subplots = True
    hspace = 0.45
    wspace = 0.6
    top = 0.93
    bottom = 0.07
    left = None
    right = None

    # tight bbox for the figure:
    bbox_inches = 'tight'

    # figure width is a DINA4-width, figure height is kept to give the plot-ratio, but maximally it's a DINA4-length
    DINA4_length = 29.7
    DINA4_width = 21.0
    plot_ratio = DINA4_length / (2*DINA4_width)
    in_per_cm = 0.39

    # take as figure a folded DINA4-sheet
    figwidth = DINA4_width * in_per_cm
    figheight = figwidth * plot_ratio * nof_rows / nof_cols

    if figheight > DINA4_length * in_per_cm:
        figheight = DINA4_length * in_per_cm

    # set the variable figsize
    figsize = (figwidth, figheight)

    # adjust the suplots
    if adjust_subplots:
        plt.subplots_adjust(hspace=hspace, wspace=wspace, top=top, bottom=bottom, left=left, right=right)


    plot_2d_xdata = np.arange(nof_coefficients)
    vmax = max(np.max(chain_ranks), np.max(star_ranks))

    # Plot chain ranks
    plt.subplot(nof_rows, nof_cols, 1)
    plot_2d(plot_2d_xdata, times, chain_ranks, ylabel='bond', xlim=(times[0], times[-1]),
            ylim=None, invert_x=False, invert_y=True, colorbar=False,
            tick_params=None, colorbar_params=None, title='(I)', fontsize_title=11, vmin=0, vmax=vmax,
            data_transpose=True, data_flipud=False, data_fliplr=False, plotbox_position=None, fontsize_labels=11)

    # Plot star ranks
    plt.subplot(nof_rows, nof_cols, 2)
    if vmax >= 10:
        colorbar_params = {'orientation': 'vertical', 'aspect': 15, 'pad': 1.8,
                           'ticks': [0, int(floor(vmax/10)*10)/2, int(floor(vmax/10)*10)]}
    else:
        colorbar_params = {'orientation': 'vertical', 'aspect': 15, 'pad': 1.8}
    plot_2d(plot_2d_xdata, times, star_ranks, ylabel=None, xlim=(times[0], times[-1]),
            ylim=None, invert_x=False, invert_y=True, colorbar=True,
            tick_params={'direction': 'in', 'bottom': True, 'top': True, 'left': True, 'right': True, 'labelleft': False},
            colorbar_params=colorbar_params,
            title='(II)', fontsize_title=11, vmin=0, vmax=vmax,
            data_transpose=True, data_flipud=False, data_fliplr=False, plotbox_position=[0.89, 1, 1, 1],
            fontsize_labels=11)

    # Plot chain and star size
    plt.subplot(nof_rows, nof_cols, 3)
    ymin = int(min(floor(log(np.min(chain_size), 10)), floor(log(np.min(star_size), 10))))
    ymax = int(max(ceil(log(np.max(chain_size), 10)), ceil(log(np.max(star_size), 10))))
    ylim = [10**ymin, 10**ymax]
    ytics_tilde = [10**x for x in range(ymin, ymax+1)]
    ytics_labels = ['1e'+str(int(x)) for x in range(ymin, ymax+1)]
    plot_1d(times, ((0, chain_size), (1, star_size)), ytics_tilde, ytics_labels, ylabel=None, xlim=(times[0], times[-1]),
            ylim=ylim, tick_params=None, title='size', fontsize_title=11,
            plotbox_position=None, fontsize_labels=11,
            legend=('(I)', '(II)'), legendloc='best', fontsize_legend=10, linewidth=(1.8, 1.8),
            linestyle=('-', '--'), color_list=('blue', 'orange'), xscale='linear', yscale='log')

    # Plot rel. diff between chain and star
    plt.subplot(nof_rows, nof_cols, 4)
    ymin = int(floor(log(np.min(rel_diff[rel_diff != 0]), 10)))
    ymax = int(ceil(log(np.max(rel_diff), 10)))
    ylim = [10 ** ymin, 10 ** ymax]
    ytics_tilde = [10 ** x for x in range(ymin, ymax+1)]
    ytics_labels = ['1e' + str(int(x)) for x in range(ymin, ymax+1)]
    if len(ytics_tilde) > 3:
        ytics_tilde = ytics_tilde[::2]
    if len(ytics_labels) > 3:
        ytics_labels = ytics_labels[::2]
    plot_1d(times, ((0, rel_diff), ), ytics_tilde, yticks_labels=ytics_labels, ylabel=None, xlim=(times[0], times[-1]),
            ylim=ylim, tick_params=None, title='rel. diff.', fontsize_title=11,
            plotbox_position=None, fontsize_labels=11,
            legend=(None, ), legendloc='best', fontsize_legend=10, linewidth=(1.8, ),
            linestyle=('-',), color_list=('black',), xscale='linear', yscale='log')

    # set the figure size of the image
    fig.set_size_inches(figsize[0], figsize[1])

    # save the figure
    plt.savefig(join(root, plot_name + '.' + fig_format), bbox_inches=bbox_inches)
    plt.rcdefaults()
    plt.clf()  # clear current figure
    plt.cla()  # clear axes
