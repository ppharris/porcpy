#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cartopy.feature import COASTLINE, BORDERS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import iris
import iris.plot as iplt
from iris.analysis.stats import pearsonr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from shapely.geometry import box
import xarray as xr


def is_in_period(month, period):
    """Return which months fall within a specified group of calendar months.

    Parameters
    ----------
    month : int or iterable of ints
        One or a series of calendar month numbers [1..12].
    period : tuple of ints
        Group of calendar month numbers to match against.  These are usually
        consecutive, e.g., [3, 4, 5] for MAM, but they don't have to be.

    Returns
    -------
    return : bool or iterable of bools
        True for month numbers that are in the group.

    """
    try:
        return [m in period for m in month]
    except TypeError:
        return month in period


def select_period(tdat, period):
    """Return parts of an xarray that fall within a requested set of months."""
    return tdat.sel(time=is_in_period(tdat['time.month'], period))


def calc_nhd(tdiff):
    """Return the number of positive elements of an xarray along the time dim.

    """
    nhd_yr = tdiff.where(tdiff > 0).count(dim='time')
    return nhd_yr


def calc_hot_days(tas_filename, tas_varname, nhd_filename, quantile=0.9):
    """Driver routine for calculating the number of hot JJA days each year.

    Hot days are defined relative to the percentile 100*quantile of all JJA
    days in the input dataset.

    Parameters
    ----------
    tas_filename : str
        Name of input netCDF file containing daily air temperature time series.
    tas_varname : str
        NetCDF variable name for input air temperature data.
    nhd_filename : str
        Name of output netCDF file containing annual time series of the number
        of hot JJA days.
    quantile : float, optional
        Quantile of daily air temperature above which days are considered hot.
        Values in the range (0-1).

    """
    dset_in = xr.open_mfdataset(tas_filename, combine='by_coords')
    tas = dset_in[tas_varname]
    tas_jja = select_period(tas, [6, 7, 8])

    # Calculate 90th percentile from JJA values for the full period.
    tas_jja_90 = tas_jja.load().quantile(quantile, dim='time')

    tas_diff = tas_jja - tas_jja_90

    # Calculate the number of hot days year-wise.
    nhd = tas_diff.groupby('time.year').map(calc_nhd)
    nhd.attrs["long_name"] = "Number of hot days"
    nhd.attrs["units"] = "days"

    dset_out = xr.Dataset({
        "nhd": nhd,
        "lon_bnds": dset_in["lon_bnds"],
        "lat_bnds": dset_in["lat_bnds"],
    }, attrs=dset_in.attrs)

    dset_out.to_netcdf(nhd_filename)

    return


def calc_acc_pr(pr_filename, pr_varname, prm_filename, season):
    """Driver routine for calculating the mean precipitation rate in a season.

    Parameters
    ----------
    pr_filename : str
        Name of input netCDF file containing daily precipitation rate time
        series (kg m-2 s-1).
    pr_varname : str
        NetCDF variable name for input precipitation rate.
    prm_filename : str
        Name of output netCDF file containing annual time series of
        precipitation rate for the requested season.
    season : int or iterable of ints
        Month numbers over which to calculate the seasonal mean.  Values in the
        range [1-12].

    """
    dset_in = xr.open_mfdataset(pr_filename, combine='by_coords')
    pr = dset_in[pr_varname]
    pr_seas = select_period(pr, season)

    # Calculate the number of hot days year-wise.
    prm = pr_seas.groupby('time.year').mean(dim='time')

    prm.attrs["long_name"] = "precipitation"
    prm.attrs["cell_methods"] = f"time: mean over months {season}"

    dset_out = xr.Dataset({
        "pr": prm.compute(),
        "lon_bnds": dset_in["lon_bnds"],
        "lat_bnds": dset_in["lat_bnds"],
    }, attrs=dset_in.attrs)

    dset_out.to_netcdf(prm_filename)

    return


def regional_mean(cube, attrs):
    """Return the area mean of an Iris cube over a lon/lat sub-domain."""
    cube_out = cube.intersection(longitude=attrs["longitudes"],
                                 latitude=attrs["latitudes"])
    grid_areas = iris.analysis.cartography.area_weights(cube_out)
    cube_out = cube_out.collapsed(["longitude", "latitude"],
                                  iris.analysis.MEAN,
                                  weights=grid_areas)
    return cube_out


def get_label(name, attrs):
    """Return a string with nicely formatted lons and lats."""
    lon0, lon1 = attrs["longitudes"]
    lat0, lat1 = attrs["latitudes"]
    return "{} ({}-{}, {}-{})".format(name,
                                      LONGITUDE_FORMATTER(lon0),
                                      LONGITUDE_FORMATTER(lon1),
                                      LATITUDE_FORMATTER(lat0),
                                      LATITUDE_FORMATTER(lat1))


def _add_gridlines(ax):
    try:
        ggl = ax.gridlines(color="0.5",
                           linestyle="--",
                           draw_labels=True)

        ggl.xlocator = MultipleLocator(30)
        ggl.ylocator = MultipleLocator(15)
        ggl.xlabels_top = False
        ggl.ylabels_right = False
        ggl.xlabel_style = {"size": 10}
        ggl.ylabel_style = {"size": 10}
        ggl.xformatter = LONGITUDE_FORMATTER
        ggl.yformatter = LATITUDE_FORMATTER
    except TypeError as e:
        # Catch attempts to gridline unsupported projections.
        # logger.warn(str(e))
        print(str(e))

    return


def plot_scatter(file_in_nhd, files_in_prm, file_out_plot, title=None):
    """Driver routine for making standard precip vs hot day scatter plots.

    Parameters
    ----------
    file_in_nhd : str
        Name of input netCDF file containing annual time series of number of
        hot JJA days.
    files_in_prm : dict
        Dictionary of season_name:filename pairs for input netCDF files
        containing seasonal mean precipitation rate.
    file_out_plot : str
        Name of output image file.  Any existing file will be clobbered.
    title : str, optional
        Title to add to the overall multi-plot figure.
    """

    regions = {
        "NE China": {
            "longitudes": [110, 120],
            "latitudes": [30, 40],
            "color": "tab:red",
        },
        "Tibet": {
            "longitudes": [80, 95],
            "latitudes": [29, 37],
            "color": "tab:blue",
        },
        "SE Asia": {
            "longitudes": [100, 110],
            "latitudes": [10, 20],
            "color": "tab:green",
        },
    }

    if title is None:
        title = "Number of hot days in JJA vs antecedent precipitation"

    nhd = iris.load_cube(file_in_nhd, "Number of hot days")

    # Initialise a multipanel figure.
    nr, nc = 2, 3
    F, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(12, 8))
    F.subplots_adjust(left=0.03,
                      right=0.97,
                      bottom=0.07,
                      wspace=0.3,
                      hspace=0.3)
    F.suptitle(title)

    # Scatter plot for each season.
    for ax, (season, file_prm) in zip(axs[:, 1:].flat, files_in_prm.items()):

        prm = iris.load_cube(file_prm, "precipitation")

        lines = []
        for name, attrs in regions.items():
            nhd_reg = regional_mean(nhd, attrs)
            prm_reg = regional_mean(prm, attrs)

            L = ax.plot(prm_reg.data, nhd_reg.data,
                        lw=0, marker="o", color=attrs["color"])[0]
            lines.append((L, get_label(name, attrs)))

            regress = np.poly1d(np.polyfit(prm_reg.data, nhd_reg.data, 1))
            ax.plot(prm_reg.data, regress(prm_reg.data),
                    marker=None, color=attrs["color"])

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 30)
        ax.set_xlabel("Mean antecedent precipitation (mm/day)")
        ax.set_ylabel("Number of hot days")
        ax.set_title("Season: %s" % season)

    # Map of regions.
    ax = plt.subplot(nr, nc, 1, projection=ccrs.PlateCarree())
    axs[0, 0] = ax

    ax.set_extent([65, 140, 0, 50])
    ax.add_feature(COASTLINE)
    ax.add_feature(BORDERS, edgecolor="0.5")

    for region in regions.values():
        lons, lats = region["longitudes"], region["latitudes"]
        reg_box = box(lons[0], lats[0], lons[1], lats[1])
        ax.add_geometries([reg_box, ], ccrs.PlateCarree(),
                          lw=1.5, edgecolor=region["color"], facecolor="None")

    ax.legend(*zip(*lines), bbox_to_anchor=(0.5, -0.05), loc="upper center")

    # No plot in the bottom left.
    F.delaxes(axs[1, 0])

    plt.savefig(file_out_plot)

    return


def plot_rmaps(file_in_nhd, files_in_prm, file_out_plot, title=None):

    region = {
        "longitudes": (60-1e-3, 150+1e-3),
        "latitudes": (-5, 55),
    }

    if title is None:
        title = "Correlation between antecedent precipitation and number of JJA hot days"

    nhd = iris.load_cube(file_in_nhd, "Number of hot days")

    nr, nc = 2, 2
    F, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(12, 8),
                          subplot_kw=dict(projection=ccrs.PlateCarree()))
    F.suptitle(title)

    cticks = np.linspace(-1, 1, 11)
    cmap = cm.get_cmap("coolwarm_r", lut=2*(len(cticks)-1))

    for ax, (season, file_prm) in zip(axs.flat, files_in_prm.items()):
        prm = iris.load_cube(file_prm, "precipitation")
        corr = pearsonr(prm, nhd, corr_coords="year")

        PCM = iplt.pcolormesh(corr, axes=ax,
                              vmin=min(cticks), vmax=max(cticks), cmap=cmap)

        ax.set_title("Season: %s" % season)
        ax.set_extent(region["longitudes"] + region["latitudes"])
        ax.add_feature(COASTLINE)
        _add_gridlines(ax)

    cax = F.add_axes([0.90, 0.53, 0.02, 0.35])
    F.colorbar(PCM, cax=cax, ticks=cticks)

    plt.savefig(file_out_plot)

    return
