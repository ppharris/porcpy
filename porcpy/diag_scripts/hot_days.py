#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cartopy.feature import COASTLINE, BORDERS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import iris
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
import xarray as xr


def is_in_period(month, period):
    try:
        return [m in period for m in month]
    except TypeError:
        return month in period


def select_period(tdat, period):
    return tdat.sel(time=is_in_period(tdat['time.month'], period))


def calc_nhd(tdiff):
    nhd_yr = tdiff.where(tdiff > 0).count(dim='time')
    return nhd_yr


def calc_hot_days(tas_filename, tas_varname, nhd_filename, quantile=0.9):

    tas = xr.open_mfdataset(tas_filename)[tas_varname]
    tas_jja = select_period(tas, [6, 7, 8])

    # Calculate 90th percentile from JJA values for the full period.
    tas_jja_90 = tas_jja.load().quantile(quantile, dim='time')

    tas_diff = tas_jja - tas_jja_90

    # Calculate the number of hot days year-wise.
    nhd = tas_diff.groupby('time.year').map(calc_nhd)
    nhd.attrs["long_name"] = "Number of hot days"
    nhd.attrs["units"] = "days"

    nhd.to_netcdf(nhd_filename)

    return


def calc_acc_pr(pr_filename, pr_varname, prm_filename, season):

    pr = xr.open_mfdataset(pr_filename)[pr_varname]
    pr_seas = select_period(pr, season)

    # Calculate the number of hot days year-wise.
    prm = pr_seas.groupby('time.year').mean(dim='time')

    prm.attrs["long_name"] = "precipitation"
    prm.attrs["cell_methods"] = f"time: mean over months {season}"

    prm.to_netcdf(prm_filename)

    return


def regional_mean(cube, attrs):
    cube_out = cube.intersection(longitude=attrs["longitudes"],
                                 latitude=attrs["latitudes"])
    cube_out = cube_out.collapsed(["longitude", "latitude"],
                                  iris.analysis.MEAN)
    return cube_out


def get_label(name, attrs):
    lon0, lon1 = attrs["longitudes"]
    lat0, lat1 = attrs["latitudes"]
    return "{} ({}-{}, {}-{})".format(name,
                                      LONGITUDE_FORMATTER(lon0),
                                      LONGITUDE_FORMATTER(lon1),
                                      LATITUDE_FORMATTER(lat0),
                                      LATITUDE_FORMATTER(lat1))


def plot_scatter(file_in_nhd, files_in_prm, file_out_plot, title=None):

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
    F.subplots_adjust(left=0.03, right=0.97, bottom=0.07, wspace=0.3, hspace=0.3)
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
