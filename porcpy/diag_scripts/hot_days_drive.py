#!/usr/bin/env python
# -*- coding: utf-8 -*-

from esmvaltool.diag_scripts.shared import (run_diagnostic,
                                            group_metadata,
                                            select_metadata,
                                            get_diagnostic_filename,
                                            get_plot_filename)

import warnings
import hot_days as hot


def _get_filename(var_meta, cfg, extension="nc"):
    """Return a filename for output data, e.g., number of hot days."""
    basename = "_".join([
        var_meta["project"],
        var_meta["dataset"],
        var_meta["exp"],
        var_meta["ensemble"],
        var_meta["short_name"],
        str(var_meta["start_year"]),
        str(var_meta["end_year"]),
    ])

    filename = get_diagnostic_filename(basename, cfg, extension=extension)
    return filename


def _get_plot_filename(var_meta, cfg, label):
    """Return an output filename for plots."""
    basename = "_".join([
        var_meta["project"],
        var_meta["dataset"],
        var_meta["exp"],
        var_meta["ensemble"],
        label,
        str(var_meta["start_year"]),
        str(var_meta["end_year"]),
    ])

    filename = get_plot_filename(basename, cfg)
    return filename


def make_hot_days(cfg, dataset, input_data):
    """Shim routine between ESMValTool and hot_days.calc_hot_days().

    Calculates the number of hot days (NHD) in JJA each year, where "hot" days
    are relative to a threshold quantile specified in the ESMValTool recipe.
    The NHD fields for each year are written to a netCDF file.
    """

    tas_meta = select_metadata(input_data, short_name="tas")[0]

    # Generate an NHD output filename.
    keys = ["project", "dataset", "exp", "ensemble", "start_year", "end_year"]
    nhd_meta = {k: tas_meta[k] for k in keys}
    nhd_meta["short_name"] = "nhd"
    nhd_filename = _get_filename(nhd_meta, cfg, extension="nc")

    hot.calc_hot_days(tas_meta["filename"],
                      tas_meta["short_name"],
                      nhd_filename,
                      quantile=tas_meta["quantile"])

    return nhd_filename


def make_pr_mean(cfg, dataset, input_data):
    """Shim routine between ESMValTool and hot_days.calc_acc_pr().

    Calculates the mean precipitation for several periods ("seasons")
    antecedent to JJA for each year.  These seasons are typically April-May,
    May, March-May, and Jun-Aug.  The mean pr annual time series are written to
    a separate netCDF file for each season type.
    """

    seasons = {
        "am": [4, 5],
        "may": [5, ],
        "mam": [3, 4, 5],
        "jja": [6, 7, 8],
    }

    pr_files = {}

    pr_meta = select_metadata(input_data, short_name="pr")[0]
    keys = ["project", "dataset", "exp", "ensemble", "start_year", "end_year"]

    for name, season in seasons.items():

        # Generate precipitation output filename.
        prm_meta = {k: pr_meta[k] for k in keys}
        prm_meta["short_name"] = f"prm_{name}"
        prm_filename = _get_filename(prm_meta, cfg, extension="nc")

        # Calculate the mean precip for this season.
        hot.calc_acc_pr(pr_meta["filename"],
                        pr_meta["short_name"],
                        prm_filename,
                        season)

        # Save the output filename for use in subsequent plottings steps.
        pr_files[name] = prm_filename

    return pr_files


def make_plots(cfg, dataset, data, nhd_file, prm_files):
    """Shim routine between ESMValTool and hot_days.plot_scatter().

    Makes standard scatter plots of precip vs NHD for each season and for
    several pre-defined regions in East Asia.  The plot format (e.g., png) and
    file location are specified through the ESMValTool recipe and user
    configuration files.
    """

    tas_meta = select_metadata(data, short_name="tas")[0]
    filename_scatter = _get_plot_filename(tas_meta, cfg, "scatter")
    filename_maps = _get_plot_filename(tas_meta, cfg, "rmaps")

    model_desc = "{:s}, {:s}, {:s}, {:s}, {:d}-{:d}".format(
        tas_meta["project"],
        tas_meta["exp"],
        tas_meta["dataset"],
        tas_meta["ensemble"],
        tas_meta["start_year"],
        tas_meta["end_year"],
    )

    args = [tas_meta["quantile"] * 100, model_desc]

    title_scatter = ("Number of hot days (tas > Q{:2.0f}) vs antecedent mean precipitation"  # noqa
                     "\n{:s}".format(*args))

    title_rmaps = ("Correlation coefficient for number of hot days (tas > Q{:2.0f}) vs "  # noqa
                   "antecedent mean precipitation\n{:s}".format(*args))

    hot.plot_scatter(nhd_file, prm_files, filename_scatter, title=title_scatter)
    hot.plot_rmaps(nhd_file, prm_files, filename_maps, title=title_rmaps)

    return


def main(cfg):
    grouped_metadata = group_metadata(cfg["input_data"].values(), "dataset")

    for dataset, data in grouped_metadata.items():
        nhd_file = make_hot_days(cfg, dataset, data)
        prm_files = make_pr_mean(cfg, dataset, data)
        if cfg["write_plots"]:
            make_plots(cfg, dataset, data, nhd_file, prm_files)

    return


if __name__ == '__main__':

    # Suppress a warning about the Iris area averaging.  This could be fixed by
    # using weights from the CMIP areacella files.
    warnings.filterwarnings("ignore",
                            message=".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")

    with run_diagnostic() as cfg:
        main(cfg)
