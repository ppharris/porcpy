#!/usr/bin/env python
# -*- coding: utf-8 -*-

from esmvaltool.diag_scripts.shared import (run_diagnostic,
                                            group_metadata,
                                            select_metadata,
                                            get_diagnostic_filename,
                                            get_plot_filename)

import hot_days as hot


def _get_filename(var_meta, cfg, extension="nc"):
    """Return a filename for output data."""
    basename = "_".join([
        var_meta["short_name"],
        var_meta["project"],
        var_meta["dataset"],
        var_meta["exp"],
        var_meta["ensemble"],
        str(var_meta["start_year"]),
        str(var_meta["end_year"]),
    ])

    filename = get_diagnostic_filename(basename, cfg, extension=extension)
    return filename


def _get_plot_filename(var_meta, cfg, label):
    """Return an output filename for plots."""
    basename = "_".join([var_meta["project"],
                         var_meta["dataset"],
                         var_meta["exp"],
                         var_meta["ensemble"],
                         label])

    filename = get_plot_filename(basename, cfg)
    return filename


def make_hot_days(cfg, dataset, input_data):

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

    seasons = {
        "am": [4, 5],
        "may": [5, ],
        "mam": [3, 4, 5],
        "jja": [6, 7, 8],
    }

    pr_files = {}

    pr_meta = select_metadata(input_data, short_name="pr")[0]

    for name, season in seasons.items():

        # Generate precipitation output filename.
        keys = ["project", "dataset", "exp", "ensemble", "start_year", "end_year"]
        prm_meta = {k: pr_meta[k] for k in keys}
        prm_meta["short_name"] = f"prm_{name}"
        prm_filename = _get_filename(prm_meta, cfg, extension="nc")

        hot.calc_acc_pr(pr_meta["filename"],
                        pr_meta["short_name"],
                        prm_filename,
                        season)

        pr_files[name] = prm_filename

    return pr_files


def make_plots(cfg, dataset, data, nhd_file, prm_files):

    tas_meta = select_metadata(data, short_name="tas")[0]
    filename_scatter = _get_plot_filename(tas_meta, cfg, "scatter")

    title = ("Number of hot days (tas > Q{:2.0f}) vs antecedent mean precipitation\n"
             "{:s}, {:s}, {:s}, {:s}, {:d}-{:d}").format(
                 tas_meta["quantile"] * 100,
                 tas_meta["project"],
                 tas_meta["exp"],
                 tas_meta["dataset"],
                 tas_meta["ensemble"],
                 tas_meta["start_year"],
                 tas_meta["end_year"],
             )

    hot.plot_scatter(nhd_file, prm_files, filename_scatter, title=title)

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
    with run_diagnostic() as cfg:
        main(cfg)
