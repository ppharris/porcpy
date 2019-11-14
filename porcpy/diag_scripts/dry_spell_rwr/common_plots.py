#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

from cartopy.crs import EqualEarth as PlotProj
from cartopy.feature import LAND, OCEAN
from matplotlib import rc
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import netCDF4 as nc
import numpy as np

from .event import iterevents
from .geo_grid import read_grid
from . import file_io as fio
from . import srex as srex
from . import logger


rc("font", family="DejaVu Sans")

_NEVE_MIN = 10
_SE_MAX = 0.03  # K/day


class Plotable(object):
    def __init__(self, pattern, label=None, color=None):
        self.pattern = pattern
        self.label = label
        self.color = color

    def get_filenames(self, regions):
        """Generate file names for a list of
        <dry_spell_rwr.srex.Region>."""
        for r in regions:
            yield self.pattern.format(region=r.short_name)


def nc_rwr_read(filename, grid, name_rwr="rwr", name_neve="neve",
                name_se="stderr",
                neve_min=_NEVE_MIN, se_max=_SE_MAX):
    """
    Read RWR from a netCDF file produced by <file_io.nc_rwr_write>
    and apply some extra data rejection based on the number of events.

    Parameters
    ----------

    filename : str
        Name of input netCDF file containing RWR data.
    grid : <geo_grid.LandGrid> instance
        Object describing the RWR spatial grid.
    name_rwr : str, optional
        Name of relative_warming_rate variable in input netCDF file.
    name_neve : str, optional
        Name of number_of_dry_spells variable in input netCDF file.
    name_se : str, optional
        Name of p_value variable in input netCDF file.
    neve_min : int, optional
        Minimum number of dryspells below which a gridbox RWR will be
        masked.  Threshold is compared with the minimum number over dry
        spell day 2 onwards.
    se_max : float, optional
        Standard error value (K/day) above which RWR values are
        considered bad.

    Returns
    -------

    rwr : <numpy.ma.MaskedArray> instance, shape(lat, lon)
        Relative warming rate (K/day).
    bad : <numpy.array> instance, shape(lat, lon)
        Array defining grid boxes that will be greyed-out, i.e., where
        var is poorly estimated.

    """

    with nc.Dataset(filename, "r") as ncid:
        rwr = ncid.variables[name_rwr][:]
        neve = ncid.variables[name_neve][6:, ...].min(axis=0)
        rwr.mask[neve < neve_min] = True

        se = ncid.variables[name_se][:]

    # Make a boolean array showing which gridboxes rwr estimates have standard
    # errors that exceed a threshold value (i.e, are too large).  This is used
    # to grey-out grid boxes on an RWR map.
    bad = se > se_max

    return rwr, bad


def _get_cmap(name, levs, bad=None, under=None, over=None):
    """
    Return a Matplotlib colormap subject to some constraints.

    Parameters
    ----------

    name : str
        Name of Matplotlib colormap to return.
    levs : list or ndarray
        N tick levels.
    bad : Matplotlib color, optional
        Color to use for missing values.
    under : Matplotlib color, optional
        Color to use for under-scale values z < min(levs).
    over : Matplotlib color, optional
        Color to use for over-scale values z > max(levs).

    Returns
    -------
    cmap : <matplotlib.colors.Colormap> instance.
        A colormap of N-1 colors.

    """

    cmap = cm.get_cmap(name, lut=len(levs)-1)
    if bad is not None:
        cmap.set_bad(bad)
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)

    extend = {(True, True): "neither",
              (False, True): "min",
              (True, False): "max",
              (False, False): "both"}

    cmap.colorbar_extend = extend[(under is None, over is None)]

    return cmap


def _plot_map(ax, grid, var, levs, cmap, ticks=None, labelled=False,
              ocean=None):
    """
    Plot data on a map to an existing set of axes.

    Parameters
    ----------

    ax : A <matplotlib.axes.Axes> instance.
        Axes to which var will be drawn.
    grid : <geo_grid.LandGrid> instance
        Object describing the spatial grid of var.
    var : ndarray, shape(lat, lon)
        Data to be plotted.
    levs : list or ndarray
        Set of N contour levels to plot the data.
    cmap : <matplotlib.colors.Colormap> instance.
        A colormap of N-1 colors.
    ticks : list or ndarray, optional
        If present draw a colorbar marked with these tick levels.
    labelled : bool, optional
        If True, draw lon/lat gridlines and labels on the map.
    ocean : Matplotlib color, optional
        Color to mask the ocean.

    Returns
    -------

    PCM : mappable
        E.,g, <matplotlib.collections.QuadMesh>

    """

    kw_plot = dict(vmin=min(levs), vmax=max(levs), cmap=cmap)

    PCM = grid.plot_var(ax, var, labelled=labelled, kw_plot=kw_plot)

    if ticks is not None:
        plt.colorbar(PCM, fraction=0.05, ticks=ticks)

    if ocean is not None:
        ax.add_feature(OCEAN, facecolor=ocean)

    return PCM


def _plot_hist(ax, z, cmap, bad=None):
    """
    Add a histogram to an existing set of axes.

    Parameters
    ----------

    ax : <matplotlib.axes.Axes> instance or similar.
        The axes on which the inset historam is drawn.  The new inset
        axes are defined relative to this axes.
    z : <numpy.MaskedArray> instance.
        Possibly masked data to plot on as a histogram.
    cmap : <matplotlib.colors.Colormap> instance.
        Colormap used to derive colors for positive and negative bins.
    bad : <numpy.array> instance, shape(lat, lon), optional
        Array indicating additional gridboxes to exclude from the
        histogram.

    Returns
    -------

    bx : <matplotlib.axes.Axes> instance.

    """

    if bad is None:
        zc = z.compressed()
    else:
        zc = z[~bad].compressed()

    bx = inset_axes(ax, width="15%", height=0.7, loc=6,
                    bbox_to_anchor=(0.10, 0.1, 1.1, 0.5),
                    bbox_transform=ax.transAxes)

    bins = np.linspace(-1.05, 1.05, 42)
    n, bins, patches = bx.hist(zc, bins, density=False, histtype="bar",
                               edgecolor="k")

    msg = ("Data percentiles: %s", np.percentile(zc, [0, 25, 50, 75, 100]))
    if zc.min() < bins[0] or zc.max() > bins[-1]:
        logger.warn("Some data are outside of requested bins.")
        logger.warn(*msg)
    else:
        logger.info(*msg)

    col_neg, col_pos = cmap(0), cmap(cmap.N-1)
    for patch, right, left in zip(patches, bins[1:], bins[:-1]):
        if left < 0.0 < right:
            patch.set_facecolor('w')
        elif left < 0.0:
            patch.set_facecolor(col_neg)
        elif left > 0.0:
            patch.set_facecolor(col_pos)

    bx.set_facecolor("none")
    bx.set_xlim(-0.5, 0.5)
    bx.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    bx.set_yticks([0, 1000, 2000])

    bx.spines["top"].set_visible(False)
    bx.spines["right"].set_visible(False)
    bx.tick_params("both", labelsize=7, top=False, right=False)
    bx.tick_params("x", direction="out")

    return bx


def _finalize_plots(axs, regions, xlim, ylim, titles=None,
                    xlabel=None, ylabel=None,
                    xclean=False, yclean=False):
    """
    Draw some common components and tidy a group of existing axes.

    This includes drawing zero lines, setting common xaxis and yaxis
    limits, adding plot titles and axis labels, removing unnecessary
    tick labels some rows/columns, adding a legend.

    Parameters
    ----------

    axs : Array of <matplotlib.axes.Axes> instances.
        Group of axes to be manipulated.
    regions : Iterable of <dry_spell_rwr.srex.Region> instances.
        Description of region for each plot, used to generate default
        subplot titles.
    xlim : tuple
        Plot limits for xaxes, is applied to all axes.
    ylim : tuple
        Plot limits for yaxes, is applied to all axes.
    titles : iterable of strings, optional
        If present, replaces region names as the default subplot titles.
    xlabel : str, optional
        If present, write this label to the xaxes of the bottom row of
        subplots.
    ylabel : str, optional
        If present, write this label to the yaxes of the left column of
        subplots.
    xclean : bool, optional
        If True, remove the xticklabels from all subplots except the
        bottom row.
    yclean : bool, optional
        If True, remove the yticklabels from all subplots except the
        left column.

    """

    if titles is None:
        titles = [r.long_name for r in regions]

    for ax in axs.flat:
        ax.axhline(0, color="k", alpha=0.3, lw=1)
        ax.axvline(0, color="k", alpha=0.3, lw=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for ax, title in zip(axs.flat, titles):
        ax.set_title(title, fontsize=10, pad=2)

    if xclean:
        for ax in axs[:-1, :].flat:
            ax.set_xticklabels([])
    if yclean:
        for ax in axs[:, 1:].flat:
            ax.set_yticklabels([])

    if xlabel is not None:
        for ax in axs[-1, :]:
            ax.set_xlabel(xlabel, fontsize=10)

    if ylabel is not None:
        for ax in axs[:, 0]:
            ax.set_ylabel(ylabel, fontsize=10)

    axs[0, 0].legend(loc=0, fontsize=6)

    return


def _events_add_title(axs, file_eves, events, years):
    """
    Add a particular title to a global dry spell events figure.

    Parameters
    ----------

    axs : Array of <matplotlib.axes.Axes> instances.
        Group of axes to which the figure title will be added.
    file_eves : str
        Name of input text file containing catalogue of dry spell events.
    events : list of lists of <event.Event> instances
        Catalogue of dry spell events from file_eves.
    years : set
        List of years with at least one dry spell event in file_eves.

    """

    try:
        F = axs.get_figure()
    except AttributeError:
        F = axs[0].get_figure()

    neve = sum(1 for _ in iterevents(events))
    title = "%s\nTotal number of events %d between %d and %d" % (file_eves,
                                                                 neve,
                                                                 min(years),
                                                                 max(years))
    F.suptitle(title, fontsize=10)
    return


def _plot_rwr_ptiles(ax, filename, color=None, label=None, mark_upper=True):
    """
    Plot a single RWR file to an existing set of axes.

    NB The highest percentile of antecedent precip is often very broad
    compared with the lower deciles, so this routine will reduce it's
    upper bound to make the overall plot more readable.

    Parameters
    ----------

    ax : A <matplotlib.axes.Axes> instance.
        Axes to which the RWR data will be drawn.
    filename : str
        Name of input text file containing RWR data.
    color : Matplotlib color, optional.
        Color to plot data.
    label : str, optional
        Text label for these data that will be added to the legend.
    mark_upper : bool, optional
        If True, write the actual upper bound of the highest percentile
        on the plot if the function has modified it to improve plot
        readability.

    """

    # Read the input RWR data.
    header, compos = fio.asc_ptiles_read(filename)

    # Regenerate the RWR statistics in the composites.
    for c in compos:
        days_rwr = slice(c.ndays_ante + c.regress_day_start,
                         c.ndays_ante + c.regress_day_end + 1)
        c.calc_rwr(days_rwr)

    # Extract bounds suitable for plotting with plt.errorbar().
    x = [(c.bounds[1]+c.bounds[0])/2 for c in compos]
    xb = [(c.bounds[1]-c.bounds[0])/2 for c in compos]
    y = [c.stats[0] for c in compos]
    yb = [c.stats[-1] for c in compos]

    # Wrangle the final ptile location if it's very wide.
    if xb[-1] > xb[-2]*2:
        xu_lab = x[-1] + xb[-1]  # Save max bound for labelling.
        xb[-1] = xb[-2]*2
        x[-1] = x[-2] + xb[-2] + xb[-1]
        xy_tic = x[-1] + xb[-1], y[-1]
    else:
        xu_lab, xy_tic = False, False

    # Plot the ptile RWR.
    kw_draw = dict(lw=0, elinewidth=1, marker="o", mfc=None, ms=2)
    ax.errorbar(x, y, xerr=xb, yerr=yb, color=color, label=label,
                **kw_draw)

    # Possibly mark the true upper bound of the upper ptile.
    if mark_upper and xy_tic and not getattr(ax, "xannotated", False):
        ax.annotate("%4.f" % xu_lab,
                    xy=xy_tic,
                    xytext=(0, -30), textcoords="offset points",
                    ha="right", va="baseline",
                    fontsize=6, color=color, alpha=0.3,
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.3,
                                    relpos=(1, 1), shrinkA=1, patchA=None))
        ax.xannotated = True

    return


def _plot_events_neve(ax, grid, events, scale_factor=1.0):
    """
    Plot a map of the total number of dry spell events.

    Parameters
    ----------

    ax : <matplotlib.axes.Axes> instance.
        The axes to which the map will be drawn.
    grid : <geo_grid.LandGrid> instance
        Object describing the spatial grid.
    events : list of lists of <event.Event> instances
        Catalogue of dry spell events from file_eves.
    scale_factor : float, optional
        Totals are multipled by this number before plotting.  Typically
        used to convert from total to per year.

    Returns
    -------

    PCM : mappable
        E.,g, <matplotlib.collections.QuadMesh>.

    """

    neve = np.ma.masked_less([len(e) for e in events[0]], 1)
    neve = grid.expand(neve) * scale_factor

    levs = np.linspace(0, 10, 21)
    cmap = _get_cmap("cividis", levs, over="orange")
    PCM = _plot_map(ax, grid, neve, levs, cmap, ticks=levs[::4])
    ax.set_title("Number of dry spell events per year")
    ax.add_feature(LAND, facecolor="lightgrey")

    return PCM


def _plot_events_nday(ax, grid, events, scale_factor=1.0):
    """
    Plot a map of the total number of days spent in dry spell events.

    Parameters
    ----------

    ax : <matplotlib.axes.Axes> instance.
        The axes to which the map will be drawn.
    grid : <geo_grid.LandGrid> instance
        Object describing the spatial grid.
    events : list of lists of <event.Event> instances
        Catalogue of dry spell events from file_eves.
    scale_factor : float, optional
        Totals are multipled by this number before plotting.  Typically
        used to convert from total to per year.

    Returns
    -------

    PCM : mappable
        E.,g, <matplotlib.collections.QuadMesh>.

    """

    nday = []
    for eves in events[0]:
        nday.append(sum(e.duration for e in eves))
    nday = np.ma.masked_less(nday, 1)
    nday = grid.expand(nday) * scale_factor

    levs = np.linspace(0, 360, 13)
    cmap = _get_cmap("cividis", levs, over="orange")
    PCM = _plot_map(ax, grid, nday, levs, cmap, ticks=levs[::3])
    ax.set_title("Number of days per year spent in dry spells")
    ax.add_feature(LAND, facecolor="lightgrey")

    return PCM


def _plot_events_mdur(ax, grid, events, scale_factor=1.0):
    """
    Plot a map of the median duration of dry spell events.

    Parameters
    ----------

    ax : <matplotlib.axes.Axes> instance.
        The axes to which the map will be drawn.
    grid : <geo_grid.LandGrid> instance
        Object describing the spatial grid.
    events : list of lists of <event.Event> instances
        Catalogue of dry spell events from file_eves.
    scale_factor : float, optional
        Totals are multipled by this number before plotting.  Typically
        used to convert from total to per year.

    Returns
    -------

    PCM : mappable
        E.,g, <matplotlib.collections.QuadMesh>.

    """

    mdur = []
    for eves in events[0]:
        if eves:
            md = np.median([e.duration for e in eves])
        else:
            md = -1
        mdur.append(md)
    mdur = np.ma.masked_less(mdur, 0)
    mdur = grid.expand(mdur)

    levs = np.linspace(10, 35, 11)
    cmap = _get_cmap("cividis", levs, over="orange")
    PCM = _plot_map(ax, grid, mdur, levs, cmap, ticks=levs[::2])
    ax.set_title("Median duration of dry spell")
    ax.add_feature(LAND, facecolor="lightgrey")

    return PCM


def plot_events(axs, grid, events, file_events=""):
    """
    Make a standard set of plots summarising dry spell events.

    Currently, these are maps of (1) total number of events per year,
    (2) number of days per year spend in dry spells, (3) median duration
    of a dry spell event.

    Parameters
    ----------

    axs : Array of <matplotlib.axes.Axes> instances.
        Group of axes to which maps will be drawn.
    grid : <geo_grid.LandGrid> instance
        Object describing the spatial grid.
    events : list of lists of <event.Event> instances
        Catalogue of dry spell events.
    file_events : str, optional
        Name of input text file containing catalogue of dry spell events.
        Used to make figure title.

    """

    idx = set(e.start_index for e in iterevents(events))
    for eve in iterevents(events):
        break

    years = range(eve._ut.num2date(min(idx)).year,
                  eve._ut.num2date(max(idx)).year + 1)
    scale_factor = 1. / len(years)

    _events_add_title(axs, file_events, events, years)
    _plot_events_neve(axs[0], grid, events, scale_factor)
    _plot_events_nday(axs[1], grid, events, scale_factor)
    _plot_events_mdur(axs[2], grid, events, scale_factor)
    return


def plot_composites(axs, regions, plotables, **kwargs):
    """
    Draw plots of dry spell composites of some variables for a set of
    regions on a set of existing figure axes.

    Parameters
    ----------

    axs : Array of <matplotlib.axes.Axes> instances.
        Axes to which the data will be drawn.
    regions : Iterable of <dry_spell_rwr.srex.Region> instances.
        Definition of regions, used to generate input file names, titles,
        etc.
    plotables : list of <Plotable> instances.
        Information (input file pattern, plot characteristics) about each
        set of composite input files, e.g., one instance for each model or
        experiment.

    """

    # lims are fixed across all plots, so must be specified.
    xlim = kwargs.pop("xlim", [-5, 12])
    ylim = kwargs.pop("ylim", [-3, 4])
    titles = kwargs.pop("titles", None)

    for p in plotables:
        files = p.get_filenames(regions)
        for ax, file_in in zip(axs.flat, files):
            days, var, nvar, wvar = fio.asc_comp_read(file_in)
            ax.plot(days+0.5, var, label=p.label, color=p.color, **kwargs)

    _finalize_plots(axs, regions, xlim, ylim, xclean=True, yclean=True,
                    titles=titles, xlabel="Day of dry spell")

    return


def plot_rwr_ptiles(axs, regions, plotables, **kwargs):
    """
    Draw plots of RWR as a function of antecedent precipitation for a set
    of regions on a set of existing figure axes.

    Parameters
    ----------

    axs : Array of <matplotlib.axes.Axes> instances.
        Axes to which the RWR data will be drawn.
    regions : Iterable of <dry_spell_rwr.srex.Region> instances.
        Definition of regions, used to generate input file names, titles,
        etc.
    plotables : list of <Plotable> instances.
        Information (input file pattern, plot characteristics) about each
        set of RWR input files, e.g., one instance for each model or
        experiment.

    """

    # ylims are fixed across all plots, so must be specified.
    ylim = kwargs.pop("ylim", [-0.5, 0.5])
    xlim = kwargs.pop("xlim", None)

    # Read data and plot each panel.
    for p in plotables:
        files = p.get_filenames(regions)
        for ax, file_in in zip(axs.flat, files):
            _plot_rwr_ptiles(ax, file_in,
                             label=p.label, color=p.color, **kwargs)

    # Tidy the axes limits and labels across all the plots.
    _finalize_plots(axs, regions, xlim, ylim, yclean=True,
                    xlabel="Antecedent precip (mm)", ylabel="RWR (K/day)")

    return


def plot_rwr_map(ax, grid, var, bad=None, title=None, levs=None, cmap=None,
                 ocean=None, labelled=False, add_hist=True,
                 add_colorbar=False):
    """
    Make a standard global RWR map.

    Parameters
    ----------

    ax : <matplotlib.axes.Axes> instance.
        The axes to which the map will be drawn.
    grid : <geo_grid.LandGrid> instance
        Object describing the RWR spatial grid.
    var : <numpy.ma.MaskedArray> instance, shape(lat, lon).
        RWR data to plot on the map and histogram.
    bad : <numpy.array> instance, shape(lat, lon), optional.
        Array defining grid boxes that will be greyed-out, e.g., where
        var is poorly estimated.
    title : str, optional
        Text to replace the standard plot title.
    levs : list, optional
        Tick levels to use on the colorbar.  Currently this is only used
        to normalise the data between [min(levs), max(levs)] for plotting.
    cmap : str or <matplotlib.colors.Colormap> instance, optional
        Colormap or colormap name.
    ocean : Matplotlib color, optional
        Color to mask the ocean.
    labelled : bool, optional
        If True, draw lon/lat gridlines and labels on the map.
    add_hist : bool, optional
        If True, a histogram of the data is added to the map.
    add_colorbar : bool, optional
        If True, a colorbar is added stealing from axes ax.

    Returns
    -------

    PCM : mappable
        E.,g, <matplotlib.collections.QuadMesh>

    """

    if title is None:
        title = "Dry spell RWR (°C day⁻¹)"

    if levs is None:
        levs = np.linspace(-0.27, 0.27, 10).tolist()

    if cmap is None:
        cmap = ListedColormap(['#b2182b', '#d6604d',
                               '#f4a582', '#fddbc7',
                               '#f7f7f7',
                               '#d1e5f0', '#92c5de',
                               '#4393c3', '#2166ac'][::-1])

        cmap.set_over('darkred')
        cmap.set_under('darkblue')

    if add_colorbar:
        cmap.colorbar_extend = "both"
        ticks = levs[:4] + [0.0, ] + levs[-4:]
    else:
        ticks = None

    PCM = _plot_map(ax, grid, var, levs, cmap, ticks=ticks, labelled=labelled,
                    ocean=ocean)

    ax.set_title(title, fontsize=12)

    if bad is not None:
        kw_plot_greyed = {
            "cmap": ListedColormap(["0.5", ]),
        }
        greyed = np.ma.ones(bad.shape, dtype=int)
        greyed.mask = ~bad
        I2 = grid.plot_var(ax, greyed, labelled=False, kw_plot=kw_plot_greyed)

    if add_hist:
        bx = _plot_hist(ax, var, cmap, bad=bad)

    return PCM


def common_plot_events(file_events, file_grid, file_out):
    """
    Make a standard set of dry spell events plots and save as a PNG
    image.

    Parameters
    ----------

    file_events : str
        Name of input text file containing dry spell events data.
    file_grid : str
        Name of input netCDF file describing the input grid.
    file_out : str
        Output image file name.  Any existing file will be clobbered.

    """

    grid = read_grid(file_grid)
    events, ut_events = fio.asc_events_read(file_events)

    F, A = plt.subplots(nrows=3, figsize=(8, 10),
                        subplot_kw=dict(projection=PlotProj()))
    F.subplots_adjust(top=0.9, bottom=0.026,
                      left=0.02, right=0.95,
                      hspace=0.177, wspace=0.2)
    plot_events(A, grid, events, file_events=file_events)
    plt.savefig(file_out)

    return


def common_plot_composites(file_patterns, file_out):
    """
    Make a standard plot of a dry spell composite variable with one panel
    for each SREX region.

    Parameters
    ----------

    file_patterns : list of str
        Input composites text file name patterns.  Each must contain the
        string '{region:s}', which is replaced with SREX region short
        names to yield actual input file names.
    file_out : str
        Output image file name.  Any existing file will be clobbered.

    """

    plotables = [Plotable(p, label="Test %d" % k)
                 for k, p in enumerate(file_patterns)]

    F, A = plt.subplots(nrows=5, ncols=5, figsize=(12, 9))

    F.suptitle("Dry spell composite LSTA (K)")
    F.subplots_adjust(top=0.9, bottom=0.065,
                      left=0.04, right=0.985,
                      hspace=0.3, wspace=0.2)

    plot_composites(A, srex.regions, plotables)

    plt.savefig(file_out)

    return


def common_plot_rwr_ptiles(file_patterns, file_out):
    """
    Make a standard plot of RWR as a function of antecedent precipitation
    decile, with one panel for each SREX region.

    Parameters
    ----------

    file_patterns : list of str
        Input RWR text file name patterns.  Must contain the string
        '{region:s}', which is replaced with SREX region short names to
        yield actual input file names.
    file_out : str
        Output image file name.  Any existing file will be clobbered.

    """

    plotables = [Plotable(p, label="Test %d" % k)
                 for k, p in enumerate(file_patterns)]

    F, A = plt.subplots(nrows=5, ncols=5, figsize=(12, 9))
    F.subplots_adjust(top=0.93, bottom=0.065,
                      left=0.06, right=0.985,
                      hspace=0.4, wspace=0.2)

    plot_rwr_ptiles(A, srex.regions, plotables,
                    ylim=[-0.1, 0.25])

    plt.savefig(file_out)

    return


def common_plot_rwr_map(file_rwr, file_grid, file_out, **kwargs):
    """
    Make a standard RWR map and save as a PNG image.

    Parameters
    ----------

    file_rwr : str
        Name of input netCDF file containing RWR data.
    file_grid : str
        Name of input netCDF file describing the input grid.
    file_out : str
        Output image file name.  Any existing file will be clobbered.

    """

    grid = read_grid(file_grid)
    rwr, bad = nc_rwr_read(file_rwr, grid)

    F, ax = plt.subplots(figsize=(7, 4),
                         subplot_kw=dict(projection=PlotProj()))
    F.subplots_adjust(left=0.01, right=0.94)

    kw = kwargs.copy()
    kw.setdefault("ocean", "lightgrey")
    kw.setdefault("add_colorbar", True)

    PCM = plot_rwr_map(ax, grid, rwr, bad=bad, **kw)

    plt.savefig(file_out, dpi=150)

    return


if __name__ == "__main__":
    pass
