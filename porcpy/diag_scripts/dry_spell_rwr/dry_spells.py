"""
A module for calculating dry spell events from time series of
precipitation and near-surface air temperature.
"""

from __future__ import print_function, division

import numpy as np
import netCDF4 as nc
from cftime import utime

from . import (FMDI, _PR_MAX, _TAS_MIN, _DURATION_MIN, _DURATION_ANTE,
               logger)
from .event import Event
from . import climatology_composites as cc
from . import file_io as fio


def make_dry_spells(file_precip, file_tas, pr_max=_PR_MAX, tas_min=_TAS_MIN,
                    dur_min=_DURATION_MIN, ante_pr_days=_DURATION_ANTE,
                    file_out_ann=None, file_out_seas=None):
    """
    Driver routine for generating lists of dry spell events based on daily
    precipiation and air temperature.

    Parameters
    ----------

    file_precip : str
        Name of input netCDF file containing daily precipitation (mm).
    file_tas : str
        Name of input netCDF file containing daily air temperature (K).
    pr_max : float, optional
        A day is dry when pr is below this value (default=0.5 mm).
    tas_min : float, optional
        A day is warm whem tas is above this value (default=10 degC).
    dur_min : int, optional
        Minimum duration of dry spell in days (default=10).
    ante_pr_days : int, optional
        Numbers of days to cumulate antecedent rainfall (default=30).
    file_out_ann : str, optional
        Name of output text file containing dry spell event definitions
        for all months.
    file_out_seas : str, optional
        Name of output text file containing dry spell event definitions
        grouped by season.

    Notes
    -----
    A dry spell event is a period of at least dur_min days with daily
    precipitation P < pr_max and daily air temperature T > tas_min on
    every day.  It's up to the user to decide what daily air temperature
    is most appropriate: mean, max, at local noon, etc.

    """

    ###########################################################################
    # Read the daily precip time series including leap days.
    ###########################################################################
    ncid = nc.Dataset(file_precip, "r")
    pr = ncid.variables["pr"][:]

    model_time = ncid.variables["time"]
    ut_model = utime(model_time.units, calendar=model_time.calendar)

    ncid.close()

    logger.info("pr (min, med, max): %5.2f, %5.2f, %5.2f mm/day" % tuple(
        np.percentile(pr, [0, 50, 100])))

    ###########################################################################
    # Read the daily tas time series including leap days.
    ###########################################################################
    ncid = nc.Dataset(file_tas, "r")
    tas = ncid.variables["tas"][:]
    ncid.close()

    logger.info("tas (min, med, max): %5.2f, %5.2f, %5.2f K" % tuple(
        np.percentile(tas, [0, 50, 100])))

    ###########################################################################
    # Remove the leap days from the time series.
    ###########################################################################
    if ut_model.calendar in cc.calendars_without_leap:
        ut_noleap = ut_model
    else:
        logger.info("Using 'noleap' as default target events calendar.")
        ut_noleap = utime(ut_model.unit_string, calendar="noleap")

    pr = cc.removeleap(pr, ut_model)
    tas = cc.removeleap(tas, ut_model)

    ###########################################################################
    # Calculate the dry spells.  The time index of each corresponds to the time
    # axis with leap days removed, so the "noleap" calendar is used to append
    # the event dates.
    ###########################################################################
    spells = get_dryspells(pr, tas, ut_noleap, dur_min=dur_min, pr_max=pr_max,
                           tas_min=tas_min, ante_pr_days=ante_pr_days)
    spells_seas = get_dryspells_perseason(spells)

    logger.info("nland = %s", pr.shape[1])
    logger.info("ndays = %s", pr.shape[0])
    logger.info("neve  = %s", sum([len(l) for l in spells]))

    ###########################################################################
    # Dump dry spell catalogues to text files.
    ###########################################################################
    if file_out_seas is not None:
        seas = ["DJF", "MAM", "JJA", "SON"]
        fio.asc_events_write(spells_seas, seas, ut_noleap, file_out_seas)

    if file_out_ann is not None:
        fio.asc_events_write((spells,), ("All months (Jan-Dec).",), ut_noleap,
                             file_out_ann)

    return


def get_dryspells(pr, tas, ut_var, pr_max=_PR_MAX, tas_min=_TAS_MIN,
                  dur_min=_DURATION_MIN, ante_pr_days=_DURATION_ANTE):
    """
    Calculate a catalogue of dry spells based on daily precip data.

    Parameters
    ----------

    pr : ndarray, shape(NTIME,NLAND)
        Daily total precipitation (mm).
    tas : ndarray, shape(NTIME,NLAND)
        Daily air temperature at local overpass time (K).
    ut_var : <cftime.utime> instance.
        Converts from time index to model date, E.g,
        utime('days since 1979-1-1',calendar='gregorian')
    pr_max : float, optional
        A day is dry when pr is below this value (default=0.5 mm).
    tas_min : float, optional
        A day is warm whem tas is above this value (default=10 degC).
    dur_min : int, optional
        Minimum duration of dry spell in days (default=10).
    ante_pr_days : int, optional
        Numbers of days to cumulate antecedent rainfall (default=30).

    Returns
    -------

    dryspells : list
        List of dry spells event information for each land point.  Each
        event has information [start time index, duration, antecedent
        precipitation].

    Notes
    -----
    If the first day of the time series is dry, then any associated dry
    spell event is not included in the catalogue because it's not
    possible to determine when that event started.  Dry spells that have
    not completed by the end of the time series are also excluded from
    the catalogue.

    """

    # Make a binary mask of wet and dry days and another version offset by one
    # day.
    pc1 = np.where(pr <= pr_max, 1, 0)
    pc1c = np.concatenate((pc1[1:, :], np.ones_like(pc1[0:1, :])), axis=0)

    # Use these masks to determine the transitions from wet to dry days and
    # from dry to wet days, such that,
    #    bgg is the time indexes of wet days that precede dry days (yy == -1).
    #    enn is the time indexes of dry days that precede wet days (yy == 1).
    # Therefore dry periods run from time indexes [b+1, e], and can be
    # referenced using the Python half-open interval [b+1, e+1).
    yy = np.transpose(pc1 - pc1c)

    bgg, enn = [], []
    for y in yy:
        bgg.append(np.where(y == -1)[0])
        enn.append(np.where(y == 1)[0])

    dryspells = []
    for l, (en, bg, rain, ta) in enumerate(zip(enn, bgg, pr.T, tas.T)):
        pointlist = []
        if len(en) > 0:
            # Remove events at the beginning and end of the time series that
            # are only half defined, i.e., an event end with an unknown start
            # date or an event start with an unknown end date.
            if en[0] < bg[0]:
                en = np.delete(en, 0)
            if len(bg) > len(en):
                bg = np.delete(bg, -1)

            # Build a list of dry spells for which duration >= dur_min.
            # Antecedent precip values for the first 30 days are invalid,
            # because there will have been insufficient data
            for e, b in zip(en, bg):
                start_dry = b + 1
                start_ante = start_dry - ante_pr_days
                duration = e - b

                ante_days = slice(start_ante, start_dry)
                dry_days = slice(start_dry, start_dry+duration)
                if duration >= dur_min and not (ta[dry_days] < tas_min).any():
                    if start_ante > -1:
                        pr_ante = rain[ante_days].sum()
                    else:
                        pr_ante = FMDI

                    eve = Event(l, start_dry, ut_var, duration, pr_ante)
                    pointlist.append(eve)

        dryspells.append(pointlist)

    return dryspells


def get_dryspells_perseason(dryspells, seasons=((12, 1, 2), (3, 4, 5),
                                                (6, 7, 8), (9, 10, 11))):
    """
    Groups events by season.

    Parameters
    ----------

    dryspells :
        List of dry spell events from getdryspelldates(), i.e., the
        event dates must have been appended for this routine to work.

    seasons : sequence of sequences
        Sequence of groups of calendar month numbers into which the dry
        spell events will be regrouped.  The default behaviour is to group
        events by meteorological season (DJF, MAM, JJA, SON), but these
        groupings could be anything.  These groups need not be exclusive:
        month numbers may occur in more than one group.

    Returns
    -------

    dryspells_seasons :
        Nested lists of dry spell events with order [SEAS][LAND][EVENT].

    """
    dryspells_seasons = []
    for season in seasons:
        eveSeas = []
        for eveLand in dryspells:
            eves = [e for e in eveLand if e.start_date().month in season]
            eveSeas.append(eves)
        dryspells_seasons.append(eveSeas)

    return dryspells_seasons
