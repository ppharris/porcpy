#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import netCDF4 as nc
import numpy as np
from cftime import utime
import os

from esmvaltool.diag_scripts.shared import (
    run_diagnostic, ProvenanceLogger,
    group_metadata, select_metadata,
    get_diagnostic_filename, get_plot_filename)

from dry_spell_rwr.geo_grid import read_grid
import dry_spell_rwr.file_io as fio
import dry_spell_rwr.utc_to_lt as utc
from dry_spell_rwr.dry_spells import make_dry_spells
from dry_spell_rwr.common_analysis import make_analysis
from dry_spell_rwr.common_plots import common_plot_rwr_map


logger = logging.getLogger(os.path.basename(__file__))


def _get_time_axis(filename, means=False):
    """Get a utime based at the actual start date of the data."""

    _STEP = 3  # Time step length (hours).
    _SPD = 24 // _STEP  # Steps per day.

    with nc.Dataset(filename, "r") as ncid:
        t = ncid.variables["time"]
        ut = utime(t.units, t.calendar)
        t0 = ut.num2date(t[0])
        t1 = ut.num2date(t[-1])

        units = "days since {} 00:00:00".format(t0.strftime("%Y-%m-%d"))
        ut_out = utime(units, ut.calendar)

        pad_beg = t0.hour // _STEP
        pad_end = (_SPD - 1) - (t1.hour // _STEP) % _SPD

    return ut_out, (pad_beg, pad_end)


def _get_model_grid(var_meta):
    """Make a LandGrid object from stflf."""

    sftlf_file = var_meta['fx_files']['sftlf']
    logger.info(sftlf_file)

    model_grid = read_grid(sftlf_file)
    logger.info(model_grid)

    return model_grid, sftlf_file


def _getweights_i(*args, **kwargs):
    weights_lp = utc.getweights(*args, **kwargs)

    # These lines reduce the weights above to a single slot with weight
    # 1.0, i.e., they pick out the period mean rather than interpolating
    # it.  They do not account for the fact that getweights() assumes the
    # first time stamp is 0000 UTC (corresponding to means over
    # (2100, 0000] UTC), whereas in this CMIP5 example the first time stamp
    # is 0130 UTC (means over (0000, 0300] UTC).
    weights_lp[weights_lp > 0.0] = 1.0
    weights_lp = weights_lp - np.concatenate(
        (np.zeros((1, weights_lp.shape[1])), weights_lp[:-1, :]))
    weights_lp[weights_lp < 0.0] = 0.0

    return weights_lp


def _get_filename(var_meta, cfg, extension="nc"):
    """Return a filename for output data."""
    basename = "_".join([var_meta["project"],
                         var_meta["dataset"],
                         var_meta["exp"],
                         var_meta["ensemble"],
                         var_meta["short_name"]])

    filename = get_diagnostic_filename(basename, cfg, extension=extension)
    return filename


def _get_plot_filename(var_meta, cfg):
    """Return an output filename for RWR map plots."""
    basename = "_".join([var_meta["project"],
                         var_meta["dataset"],
                         var_meta["exp"],
                         var_meta["ensemble"],
                         "rwr"])

    filename = get_plot_filename(basename, cfg)
    return filename


def _get_provenance_record(attributes, ancestor_files):
    """Return a provenance record describing the diagnostic data."""
    record = attributes.copy()
    record.update({
        'ancestors': ancestor_files,
        'realms': ['land'],
        'domains': ['global'],
    })
    return record


def make_daily_var(cfg, input_data, short_name, getweights, metadata,
                   scale=1, offset=0):
    """Wrapper for dry_spell_rwr.utc_to_lt.make_daily_var() to derive some
    args from the ESMValTool config.
    """

    var_meta = select_metadata(input_data, short_name=short_name)[0]
    logger.info(var_meta)

    files_var = [var_meta["filename"], ]
    var_name = var_meta["short_name"]
    local_time = var_meta["local_time"]

    model_grid, file_sftlf = _get_model_grid(var_meta)

    ut_var, ts_pad = _get_time_axis(files_var[0])
    logger.info("ts_pad = %s", ts_pad)

    file_out = _get_filename(var_meta, cfg)

    utc.make_daily_var(files_var, var_name, local_time, getweights,
                       model_grid, metadata, ut_var, tsPad=ts_pad,
                       scale=scale, offset=offset,
                       file_out=file_out)

    record_var = _get_provenance_record({}, files_var + [file_sftlf, ])
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(file_out, record_var)

    return file_out


def make_rwr(cfg, dataset, input_data):
    """Calculate dry spell relative warming rate for a single dataset."""

    pr_rate_to_amount = 3 * 3600

    logger.info(input_data)

    ###########################################################################
    # Calculate daily local time files for input varibles.
    ###########################################################################
    file_pr = make_daily_var(cfg, input_data, "pr", utc.getweights_p,
                             fio.META_PR, scale=pr_rate_to_amount)
    file_tas = make_daily_var(cfg, input_data, "tas", utc.getweights,
                              fio.META_TAS)
    file_tslsi = make_daily_var(cfg, input_data, "tslsi", utc.getweights,
                                fio.META_TSLSI)
    file_rsds = make_daily_var(cfg, input_data, "rsds", _getweights_i,
                               fio.META_RSDS)
    file_rsdscs = make_daily_var(cfg, input_data, "rsdscs", _getweights_i,
                                 fio.META_RSDSCS)

    ###########################################################################
    # Calculate the dry spells from the daily files.
    ###########################################################################
    var_meta = select_metadata(input_data, short_name="pr")[0]
    keys = ["project", "dataset", "exp", "ensemble", "start_year", "end_year"]

    ds_meta = {k: var_meta[k] for k in keys}
    ds_meta["short_name"] = "ds01"
    file_out_ann = _get_filename(ds_meta, cfg, extension="asc")

    make_dry_spells(file_pr, file_tas, file_out_ann=file_out_ann)

    record_ds = _get_provenance_record(ds_meta, [file_pr, file_tas])
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(file_out_ann, record_ds)

    ###########################################################################
    # Calculate global dry spell RWR.
    ###########################################################################
    file_sftlf = var_meta["fx_files"]["sftlf"]
    rwr_meta = {k: var_meta[k] for k in keys}
    rwr_meta["short_name"] = "rwr"
    file_out_rwr = _get_filename(rwr_meta, cfg)

    kw_rwr = dict(ndays_ante=5, ndays_dry=10, file_out_rwr=file_out_rwr)

    make_analysis(file_out_ann, file_tslsi, file_tas,
                  file_rsds=file_rsds, file_rsdscs=file_rsdscs,
                  file_sftlf=file_sftlf,
                  kw_rwr=kw_rwr)

    record_rwr = _get_provenance_record(rwr_meta,
                                        [file_sftlf, file_pr,
                                         file_tas, file_tslsi,
                                         file_rsds, file_rsdscs])
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(file_out_rwr, record_rwr)

    return file_out_rwr


def make_plot(cfg, input_data, file_rwr):
    """Plot a global RWR map like those in Gallego-Elvira et al (2019)."""

    var_meta = select_metadata(input_data, short_name="pr")[0]

    keys = ["project", "dataset", "exp", "ensemble", "start_year", "end_year"]
    meta = {k: var_meta[k] for k in keys}

    file_sftlf = var_meta["fx_files"]["sftlf"]
    file_plot = _get_plot_filename(var_meta, cfg)
    title = ("Dry spell RWR (°C day⁻¹), {project}, {dataset}, {exp}, "
             "{ensemble}, {start_year}-{end_year}".format(**meta))

    common_plot_rwr_map(file_rwr, file_sftlf, file_plot, title=title)

    record_plot = _get_provenance_record(meta, [file_sftlf, file_rwr])
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(file_plot, record_plot)

    return


def main(cfg):
    grouped_metadata = group_metadata(cfg["input_data"].values(), "dataset")

    for dataset, data in grouped_metadata.items():
        file_rwr = make_rwr(cfg, dataset, data)
        if cfg["write_plots"]:
            make_plot(cfg, data, file_rwr)

    return


if __name__ == '__main__':
    with run_diagnostic() as cfg:
        main(cfg)
