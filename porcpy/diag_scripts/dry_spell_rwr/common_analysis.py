#!/usr/bin/env python

from __future__ import print_function, division

from cftime import utime
import netCDF4 as nc
import numpy as np
from os.path import join, split

from . import (FMDI, _RWR_BEG, _RWR_END, _NDAYS_ANTE, _NDAYS_DRY, _PTILES,
               logger)
from . import climatology_composites as cc
from .event import iterevents
from . import file_io as fio
from .geo_grid import read_grid
from . import srex
from . import modis
from . import wfdei


def make_analysis(file_events, file_tslsi, file_tas,
                  file_rsds=None, file_rsdscs=None,
                  file_sftlf=None, modis_dir=None,
                  tslsi_name="tslsi", tas_name="tas",
                  rsds_name="rsds", rsdscs_name="rsdscs",
                  ut_tslsi=None, ut_tas=None, ut_model=None,
                  kw_rwr=None, kw_srex=None, kw_srex_rwr=None):

    """
    Driver routine for running the dry spell temperature analysis for a
    single land surface model.  The various parts of this analysis are
    common to all models and are usually different ways of dividing up
    the events, e.g., by region/season/antecedent precipitation.

    Parameters
    ----------
    file_events : str
        Name of input text file containing the global catalogue of dry
        spells.
    file_tslsi : str
        Name of input netCDF file containing modelled daily land surface
        temperature (K) at a local overpass time.
    file_tas : str
        Name of input netCDF file containing modelled daily near-surface
        air temperature (K) at a local overpass time.
    file_rsds : str, optional
        Name of input netCDF file containing modelled daily surface
        downwelling shortwave radiation (W/m2) at a local overpass time.
    file_rsdscs : str, optional
        Name of input netCDF file containing modelled daily surface
        downwelling shortwave radiation under clear sky (W/m2) at a local
        overpass time.
    file_sftlf : str, optional
        Name of input netCDF file describing the model grid.
    modis_dir : str, optional
        Directory containing 0.5deg MODIS Terra or Aqua LST observations.
        This acts as a switch between offline LSM and online ESM analysis.
    tslsi_name : str, optional
        Name of land surface temperature variable in file_tslsi.
    tas_name : str, optional
        Name of near-surface air temperature variable in file_tas.
    rsds_name : str, optional
        Name of shortwave down variable in file_rsds.
    rsdscs_name : str, optional
        Name of clear sky shortwave down variable in file_rsdscs.
    ut_tslsi : <cftime.utime> instance, optional
        If set, this is used instead of the base time from the tslsi
        daily input file.
    ut_tas : <cftime.utime> instance, optional
        If set, this is used instead of the base time from the tas
        daily input file.
    ut_model : <cftime.utime> instance, optional
        If set, this is used instead of the base time from the rsds
        daily input file for the ESM analysis.
    kw_rwr : dict, optional
        List of keyword arguments that will be passed to
        calc_global_rwr().  If None then calc_global_rwr() will not be
        called.
    kw_srex : dict, optional
        List of keyword arguments that will be passed to
        calc_srex_compos().  If None then calc_srex_rwr() will not be
        called.
    kw_srex_rwr : dict, optional
        List of keyword arguments that will be passed to
        calc_srex_rwr().  If None then calc_srex_rwr() will not be
        called.

    """

    ###########################################################################
    # Argument sanity checking.
    ###########################################################################
    if not any((kw_rwr, kw_srex, kw_srex_rwr)):
        logger.warn("No analysis has been selected using kwargs.")
        return

    if modis_dir is None:
        use_modis = False
        if not all((file_rsds, file_rsdscs, file_sftlf)):
            logger.error("If modis_dir is not specified, then file_rsds, "
                         "file_rsdscs and file_sftlf must all be specified "
                         "instead.")
            return
    else:
        use_modis = True

    ###########################################################################
    # Read the global dry spell events file.
    ###########################################################################
    logger.info("Reading the dry spell events list from %s", file_events)
    events, ut_events = fio.asc_events_read(file_events)

    ###########################################################################
    # Make some calendar variables for the data with leap days removed. This
    # forces padding of the data to a full number of leapday-less years so that
    # climatologies can be calculated.
    ###########################################################################
    idx = set(e.start_index for e in iterevents(events))
    n_years = (ut_events.num2date(max(idx)).year
               - ut_events.num2date(min(idx)).year + 1)
    n_time = cc.year_lengths[ut_events.calendar] * n_years
    time_events = [ut_events.num2date(t) for t in range(n_time)]

    logger.info("Event analysis start / end dates: %s / %s",
                str(time_events[0]),
                str(time_events[-1]))

    ###########################################################################
    # Calculate the clear-sky day and the composite averaging weights.
    ###########################################################################
    if use_modis:
        # Get time series (excluding leap days) of MODIS LST, LST climatology
        # and 1 km pixel count.  The pixel count is used to cloud-screen and
        # weight the model climatologies and composites.  The MODIS LST
        # climatology has a greater number of missing data than the MODIS LST
        # itself, so we need to create a combined missing data mask.
        modis_vars = ["gridboxes", "lst", "climatology"]
        logger.info("Reading MODIS %s data from %s",
                    ", ".join(modis_vars),
                    modis_dir)

        (mod_pix, mod_lst, mod_clim) = modis.read_files(time_events,
                                                        modis_vars,
                                                        modis_dir)

        var_bad = (mod_pix < modis.NPIX_MIN)
        var_bad += (mod_lst < 0.0)
        var_bad += (mod_clim < 0.0)

        weights = mod_pix

        modis_anom = np.ma.masked_array(mod_lst - mod_clim, mask=var_bad,
                                        keep_mask=True, fill_value=FMDI)
    else:
        # Mimic the MODIS 1 km pixel count used to pick out clear-sky days in
        # the MODIS observations using CMIP5 model surface downward shortwave
        # radiation (rsds) and surface downward shortwave radiation under clear
        # sky (rsdscs). Model clear-sky days are those where rsds is at least
        # 90% of rsdscs.
        logger.info("Deriving weights from model output in %s",
                    split(file_rsds)[0])

        if ut_model is None:
            ut_model = read_ut_var(file_rsds)

        rsds = fio.nc_read_var(file_rsds, rsds_name)
        rsdscs = fio.nc_read_var(file_rsdscs, rsdscs_name)

        cloudy_days = (rsds < 0.9*rsdscs)
        cloudy_days, ut_model_final = cc.extract_period(cloudy_days, ut_model,
                                                        ut_events, time_events)
        cloudy_days_noleap = cc.removeleap(cloudy_days, ut_model_final)

        var_bad = cloudy_days_noleap

        weights = np.full_like(cloudy_days_noleap, modis.NPIX_MIN,
                               dtype=rsds.dtype)
        weights[var_bad] = 0.0

        # For clarity, remove no longer needed time axes from the namespace.
        del(ut_model, ut_model_final)

    ###########################################################################
    # Get the model time series to be composited.
    ###########################################################################
    tslsi_anom = calc_var_anom(file_tslsi, tslsi_name, ut_tslsi,
                               ut_events, time_events, weights, var_bad)

    tas_anom = calc_var_anom(file_tas, tas_name, ut_tas,
                             ut_events, time_events, weights, var_bad)

    ###########################################################################
    # Get a function for deriving RWR output grid info.
    ###########################################################################
    if file_sftlf is None:
        regrid_model = wfdei.regrid_wfdei
    else:
        def regrid_model(res):
            grid = read_grid(file_sftlf)
            land = [[i] for i in range(grid.nl)]
            return grid, land

    ###########################################################################
    # Calculate SREX region composites.
    ###########################################################################
    if kw_srex is not None:
        calc_srex_compos(events, tslsi_anom, "tslsi", weights, **kw_srex)
        calc_srex_compos(events, tas_anom, "tas", weights, **kw_srex)
        if use_modis:
            calc_srex_compos(events, modis_anom, "modis", weights, **kw_srex)

    ###########################################################################
    # Calculate composites for each antecedent precip decile for each SREX
    # region.
    ###########################################################################
    if kw_srex_rwr is not None:
        calc_srex_rwr(events, tslsi_anom, tas_anom, weights,
                      modis_out=False, **kw_srex_rwr)
        if use_modis:
            calc_srex_rwr(events, modis_anom, tas_anom, weights,
                          modis_out=True, **kw_srex_rwr)

    ###########################################################################
    # Calculate RWR for each gridbox.
    ###########################################################################
    if kw_rwr is not None:
        calc_global_rwr(events, tslsi_anom, tas_anom, weights, regrid_model,
                        modis_out=False, **kw_rwr)
        if use_modis:
            calc_global_rwr(events, modis_anom, tas_anom, weights,
                            regrid_model,
                            modis_out=True, **kw_rwr)

    return


def read_ut_var(file_var):
    """Return a cftime.utime from a netCDF file 'time' coord."""
    with nc.Dataset(file_var, "r") as ncid:
        tim = ncid.variables["time"]
        ut_var = utime(tim.units, calendar=tim.calendar)
    return ut_var


def calc_var_anom(file_var, var_name, ut_var, ut_events, time_events,
                  weights, weights_mask):

    if ut_var is None:
        ut_var = read_ut_var(file_var)

    var = fio.nc_read_var(file_var, var_name)

    # Remove leap days from the model time series.
    var, ut_var_final = cc.extract_period(var, ut_var, ut_events,
                                          time_events)
    var_noleap = cc.removeleap(var, ut_var_final)
    logger.info("Shape after leap removal: %s", str(var_noleap.shape))

    # Combine the MODIS mask is with any existing missing data mask in the
    # model output.
    var_noleap = np.ma.masked_array(var_noleap, mask=weights_mask,
                                    keep_mask=True, fill_value=FMDI)

    # Calculate model LST anomalies from the 365-day year time series and apply
    # the combined missing data mask.
    var_clim = cc.get_smooth_climatology(var_noleap, ut_events,
                                         weights=weights)
    var_anom = cc.get_var_anomaly(var_noleap, var_clim, ut_events)
    logger.info("%s anom shape: %s", var_name, str(var_anom.shape))

    return var_anom


def calc_srex_compos(events, var, var_name, weights,
                     ndays_dry=_NDAYS_DRY, ndays_ante=_NDAYS_ANTE,
                     patt_srex=None, patt_vars_out=None):
    """
    Calculate composite weighted means of an input variable over a set
    of events for each SREX region, and write the results to a text file
    for each region.

    Parameters
    ----------
    events : list of lists of <dry_spell_rwr.event.Event> instances
        Global dry spell event information.  Outer list is some
        grouping, e.g., season, inner list is land point.
    var : MaskedArray, shape(time, land)
        Variable to be composited.
    var_name : str
        Name of input variable, used for output file metadata.
    weights : MaskedArray, shape(time, land)
        Weights used in calculating the composite means.  This is
        typically the number of 1 km MODIS LST data per 0.5deg gridbox.
    ndays_dry : int, optional
        Number of dry spell days to composite over.
    ndays_ante : int, optional
        Number of days before dry spell started to composite over.
    patt_srex : str
        Pattern describing the names of input files listing the land
        points for each SREX region.  Must contain "%s" which will be
        replaced with an SREX short name.
    patt_vars_out : str
        Pattern describing the names of output files for each SREX
        region.  Must contain "%(var)s" and "%(reg)s" which will be
        replaced with var_name and the SREX short name.

    """

    srex_files = [patt_srex % r.short_name for r in srex.regions]
    srex_points = [np.loadtxt(f, dtype=int) for f in srex_files]

    var_zero = np.ma.zeros(var.shape, dtype=var.dtype)
    var_zero.mask = var.mask

    events = events[0]

    for region, points in zip(srex.regions, srex_points):
        logger.info("Region %s with %d points.", region, len(points))
        compos = cc.get_composite(events, var_zero, var, weights, points,
                                  ndays_dry=ndays_dry,
                                  ndays_ante=ndays_ante)

        var_composite = [compos[0].td, ]
        n = [compos[0].neve, ]
        wsum = [compos[0].wsum, ]

        filename = patt_vars_out % dict(var=var_name, reg=region.short_name)
        header = ("Composite %s anomaly (K) for SREX region %s."
                  % (var_name, region.long_name))
        logger.info(header)
        fio.asc_comp_write(var_composite, n, wsum, ndays_dry, filename,
                           var_name=var_name, header=header)

    return


def calc_srex_rwr(events, tslsi_anom, tas_anom, weights,
                  ndays_dry=_NDAYS_DRY, ndays_ante=_NDAYS_ANTE,
                  rwr_beg=_RWR_BEG, rwr_end=_RWR_END,
                  ptiles=_PTILES, patt_srex=None, patt_rwr_out=None,
                  patt_rwr_out_modis=None, modis_out=False,
                  patt_header="RWR (K/day) and TD (K) for region %s"):
    """
    Calculate RWR from composite weighted means over a set of events for
    each SREX region and over percentiles of antecedent precipitation,
    and write the results to a text file for each region.

    Parameters
    ----------
    events : list of lists of <dry_spell_rwr.event.Event> instances
        Global dry spell event information.  Outer list is some
        grouping, e.g., season, inner list is land point.
    tslsi_anom : MaskedArray, shape(time, land)
        Land surface temperature anomaly (K).
    tas_anom : MaskedArray, shape(time, land)
        Near-surface air temperature anomaly (K).
    weights : MaskedArray, shape(time, land)
        Weights used in calculating the composite means.  This is
        typically the number of 1 km MODIS LST data per 0.5deg gridbox.
    ndays_dry : int, optional
        Number of dry spell days to composite over.
    ndays_ante : int, optional
        Number of days before dry spell started to composite over.
    rwr_beg : int, optional
        The first dry spell day to use for the RWR regression.  This is
        zero-based, so rwr_beg=1 starts the regression on the second day
        of the dry spell.
    rwr_end : int, optional
        The last dry spell day to use for the RWR regression.
    ptiles : int, optional
        Number of antecedent precipitation percentiles over which RWR
        will be calculated.
    patt_srex : str
        Pattern describing the names of input files listing the land
        points for each SREX region.  Must contain "%s" which will be
        replaced with an SREX short name.
    patt_rwr_out : str
        Pattern describing the names of model output files for each SREX
        region.  Must contain "%s" which will be replaced with an SREX
        short name.
    patt_rwr_out_modis : str
        Pattern describing the names of MODIS output files for each SREX
        region.  Must contain "%s" which will be replaced with an SREX
        short name.
    modis_out : bool, optional
        If true, indicates that the input tslsi_anom is MODIS data, so
        patt_rwr_out_modis is used to generate the output file name.
    patt_header : str, optional
        Header to be written to the output text file.  Must contain "%s"
        which will be replaced with an SREX short name.

    """

    if modis_out:
        patt_out = patt_rwr_out_modis
        patt_header = "MODIS " + patt_header
    else:
        patt_out = patt_rwr_out

    # Ensure all header lines start with the right character.
    if not patt_header.startswith("#"):
        patt_header = "# " + patt_header.replace("\n", "\n# ")

    if patt_out is None:
        raise ValueError("An output file pattern is required.")

    if ptiles < 1:
        raise ValueError("Invalid ptiles: %i" % ptiles)

    srex_files = [patt_srex % r.short_name for r in srex.regions]
    srex_points = [np.loadtxt(f, dtype=int) for f in srex_files]

    # Annual events list.
    events = events[0]

    composites = cc.get_rwr(events, tas_anom, tslsi_anom, weights, srex_points,
                            ndays_dry=ndays_dry, ndays_ante=ndays_ante,
                            rwr_beg=rwr_beg, rwr_end=rwr_end,
                            ptiles=ptiles)

    for region, compos in zip(srex.regions, composites):
        file_out = patt_out % region.short_name
        header = patt_header % region.long_name
        logger.info(header)
        fio.asc_ptiles_write(compos, file_out, header=header)

    return


def calc_global_rwr(events, tslsi_anom, tas_anom, weights, regrid_model,
                    ndays_dry=_NDAYS_DRY, ndays_ante=_NDAYS_ANTE,
                    rwr_beg=_RWR_BEG, rwr_end=_RWR_END,
                    modis_out=False,
                    file_out_rwr=None, file_out_modis=None):
    """
    Calculate RWR from composite weighted means over a set of events for
    coarse (1 degree) grid boxes globally.  Resulting (lon, lat) fields
    are written to a netCDF file.

    Parameters
    ----------
    events : list of lists of <dry_spell_rwr.event.Event> instances
        Global dry spell event information.  Outer list is some
        grouping, e.g., season, inner list is land point.
    tslsi_anom : MaskedArray, shape(time, land)
        Land surface temperature anomaly (K).
    tas_anom : MaskedArray, shape(time, land)
        Near-surface air temperature anomaly (K).
    weights : MaskedArray, shape(time, land)
        Weights used in calculating the composite means.  This is
        typically the number of 1 km MODIS LST data per 0.5deg gridbox.
    regrid_model : function
        Returns info for mapping from the input grid to a coarser output
        grid.
    ndays_dry : int, optional
        Number of dry spell days to composite over.
    ndays_ante : int, optional
        Number of days before dry spell started to composite over.
    rwr_beg : int, optional
        The first dry spell day to use for the RWR regression.  This is
        zero-based, so rwr_beg=1 starts the regression on the second day
        of the dry spell.
    rwr_end : int, optional
        The last dry spell day to use for the RWR regression.
    modis_out : bool, optional
        If true, indicates that the input tslsi_anom is MODIS data, so
        patt_rwr_out_modis is used to generate the output file name.
    file_out_rwr : str, optional
        Output netCDF file name.  This is ignored if modis_out=T.
    file_out_modis : str, optional
        Output netCDF file name.  Only used if modis_out=T.
    """

    if modis_out:
        file_out = file_out_modis
    else:
        file_out = file_out_rwr

    # Annual events list.
    events = events[0]

    # Get info for mapping WFDEI grid boxes to larger boxes for compositing.
    coarse_grid, land = regrid_model(1)

    composites = cc.get_rwr(events, tas_anom, tslsi_anom, weights, land,
                            ndays_dry=ndays_dry, ndays_ante=ndays_ante,
                            rwr_beg=rwr_beg, rwr_end=rwr_end)

    fio.nc_rwr_write(coarse_grid, composites, file_out)

    return


if __name__ == "__main__":
    pass
