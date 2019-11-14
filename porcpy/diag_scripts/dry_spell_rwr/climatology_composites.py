"""
A module for (1) calculating smoothed, daily climatologies from variable
time series and (2) calculating dry spell composites of variables.
"""

from __future__ import print_function, division

import numpy as np
from cftime import utime
from scipy.stats import linregress

from . import (FMDI, _RWR_BEG, _RWR_END, _NDAYS_ANTE, _NDAYS_DRY,
               _PTILES, _WIN_LEN, logger)
from .event import iterevents

###############################################################################
# Time series and climatology functions.
###############################################################################
_calendars = ['standard', 'gregorian', 'proleptic_gregorian', 'noleap',
              'julian', 'all_leap', '365_day', '366_day', '360_day']

calendars_with_leap = ['standard', 'gregorian', 'proleptic_gregorian',
                       'julian', 'all_leap', '366_day']
calendars_without_leap = ['noleap', '365_day', '360_day']

year_lengths = {'standard': 365, 'gregorian': 365, 'proleptic_gregorian': 365,
                'julian': 365, 'noleap': 365, 'all_leap': 366, '366_day': 366,
                '365_day': 365, '360_day': 360}


def extract_period(var, ut_model, ut_events, time_events):
    """
    Trim and/or pad an input time series to match another time axis.

    Typically used to trim/pad a model time series so that it begins and
    ends on the same days as some analysis period, as defined by the
    time_events variable.  This will make the input array have full years,
    which is required by the dry_spells module.

    Parameters
    ----------
    var : ndarray, shape(time, land)
        Input array from which data will be extracted.
    ut_model : <cftime.utime> instance
        Base time of the incoming var array.
    ut_events : <cftime.utime> instance
        Base time of the analysis period, which should be the same base
        time used in the dry spell events lists.  This usually has a
        calendar that excludes leap days.
    time_events : list of <cftime.datetime> instances
        Time dimension variable covering the analysis period.  The input
        var array is trimmed and/or padded with MDIs so that the time
        series runs from time_events[0] to time_events[-1] inclusive.

    Returns
    -------
    var : ndarray, shape(time, land)
        Output array with the time axis trimmed/padded.
    ut_model_final : <cftime.utime> instance
        Base time of the output var array.  The calendar is the same as
        the input ut_model.

    """

    n_time_model = var.shape[0]

    logger.info("Model start / end dates: %s / %s" % (
        str(ut_model.num2date(0)),
        str(ut_model.num2date(n_time_model-1))))

    slice_beg = int(ut_model.date2num(time_events[0]))
    slice_end = int(ut_model.date2num(time_events[-1])) + 1
    logger.info("Initial slice begin / end: %d / %d" % (slice_beg, slice_end))

    # Prepend MDI fields to the model time series if it starts after the
    # analysis period, and make a new utime for this padded time series.
    if slice_beg < 0:
        nbeg = abs(slice_beg)
        logger.info("Prepending %d MDI fields." % abs(slice_beg))
        var_beg = np.full_like(var[:nbeg, :], FMDI)
        var = np.concatenate([var_beg, var])
        slice_beg = 0
    else:
        nbeg = 0

    # The model time series should now start at the same date as the event
    # ansysis period, so we need a new time axis object to reflect this.
    ut_model_final = utime(ut_events.unit_string,
                           calendar=ut_model.calendar)

    # Append MDI fields to the model time series if it ends before the analysis
    # period.
    if slice_end > n_time_model:
        nend = slice_end - n_time_model
        logger.info("Appending %d MDI fields." % nend)
        var_end = np.full_like(var[:nend, :], FMDI)
        var = np.concatenate([var, var_end])
        slice_end = var.shape[0]
    else:
        slice_end += nbeg

    # Remove any parts of the model time series that are before or after the
    # analysis period.
    time_slice = slice(slice_beg, slice_end)

    if slice_beg > 0:
        ut_report = ut_model
    else:
        ut_report = ut_model_final

    logger.info("Final slice begin / end: %d / %d" % (slice_beg, slice_end))
    logger.info("Extract model dates: %s / %s" % (
        str(ut_report.num2date(slice_beg)),
        str(ut_report.num2date(slice_end-1))))

    logger.info("Shape before: %s" % str(var.shape))
    var = var[time_slice, ...]
    logger.info("Shape after : %s" % str(var.shape))

    return var, ut_model_final


def removeleap(var, ut_var):
    """
    Remove fields on leap days (29th Feb) from an input time series.

    Parameters
    ----------
    var : ndarray, shape(time, land)
        Input array from which data will be removed.
    ut_var : <cftime.utime> instance.
        E.g. utime('days since 1979-1-1', calendar='standard').

    Returns
    -------
    var : ndarray
        The input array with leap days removed.

    """

    # Do nothing if the calendar is unknown or if it's one without leap days.
    if ((ut_var.calendar not in _calendars) or
            (ut_var.calendar in calendars_without_leap)):
        return var

    LeapDay = (29, 2)
    nt = var.shape[0]

    keep = [(d.day, d.month) != LeapDay
            for d in ut_var.num2date(list(range(nt)))]
    keep = np.array(keep, dtype="bool")
    var = var[keep, ...]

    return var


def get_smooth_climatology(var, ut_var, weights=None, window_len=_WIN_LEN):
    """
    Calculate a clear-sky, smoothed climatology from a daily time series.

    Parameters
    ----------
    var : MaskedArray, shape(time, land)
        Input array on a non-leap calendar.
    ut_var : <cftime.utime> instance
        E.g., utime('days since 1979-1-1',calendar='standard').
    weights : ndarray, shape(time, land)
        Weights to use when calculating the day of year mean.  E.g., the
        number of valid MODIS 0.01 degree pixels per model gridbox
        (fill_value=FMDI).
    window_len : int
        Window length for the smoother.

    Returns
    -------
    clim : ndarray
        Smoothed climatology.

    """

    days_peryear = year_lengths[ut_var.calendar]

    # New shape for local arrays to make day-of-year averaging easier.
    doyShape = (var.shape[0]//days_peryear, days_peryear, -1)

    var = var.reshape(doyShape)

    # NB Weights are not applied to the all-sky means used for gap filling even
    # if they are applied to the clear-sky means.  This should be changed.
    var_mean = np.mean(var.data, axis=0)
    if weights is None:
        var_mean_cs = np.mean(var, axis=0)
    else:
        weights = weights.reshape(doyShape)
        var_mean_cs = np.ma.average(var, weights=weights, axis=0)

    # Gap-fill missing data in the clear-sky aray with values from the all-sky
    # array.
    var_mean_cs[var_mean_cs.mask] = var_mean[var_mean_cs.mask]
    var_mean_cs = np.tile(var_mean_cs, (3, 1))

    # Run the smoother through the noisy, gap-filled climatology.
    clim = np.zeros_like(var_mean_cs)
    for i in range(var_mean_cs.shape[1]):
        temp = smooth(var_mean_cs[:, i], window_len=window_len,
                      window='bartlett')
        clim[:, i] = temp[window_len//2:days_peryear*3 + window_len//2]

    return clim[days_peryear:days_peryear*2]


def smooth(x, window_len=_WIN_LEN, window='hanning'):
    """
    Smooth noisy data
    http://wiki.scipy.org/Cookbook/SignalSmooth

    Parameters
    ----------
    x : ndarray
        Noisy data.
    window_len : int
        Window length for the smoother.
    window : str
        Type of window.

    Returns
    -------
    y : ndarray
        Smoothed data.

    """

    window_options = ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in window_options:
        msg = "window must be one of " + " ".join(window_options)
        raise ValueError(msg)
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # Moving average.
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def get_var_anomaly(var, var_clim, ut_var):
    """
    Calculate anomaly (actual value - climatology)

    Parameters
    ----------
    var : MaskedArray, shape(time, land).
    var_clim : ndarray, shape(days_peryear, land).
        Daily climatology.
    ut_var : <cftime.utime> instance
        Base time of the incoming var array.

    Returns
    -------
    var_anom : MaskedArray, shape(time, land).
        Anomaly.

    """

    # Save the output array shape.
    out_shape = var.shape

    days_peryear = year_lengths[ut_var.calendar]

    # Make the leading dimension of the input data the number of years, which
    # makes it easier to calculate the anomalies
    work_shape = (var.shape[0]//days_peryear,
                  days_peryear,
                  var.shape[1])

    var = var.reshape(work_shape)

    var_anom = np.ma.empty_like(var)
    var_anom.set_fill_value(var.fill_value)
    var_anom.mask = var.mask

    # Iterate over years and calculate anomalies.
    for v, a in zip(var, var_anom):
        a[:] = v[:] - var_clim[:]

    # Make the output array into a flat time series, i.e.,
    # (year, day of year) -> (time).
    var_anom = var_anom.reshape(out_shape)

    return var_anom


###############################################################################
# Composite mean functions.
###############################################################################
class Composite(object):
    """
    Dry spell composite data for a single antecedent precipitation bin.

    """
    def __init__(self, bounds, tasa, lsta, neve, wsum,
                 ndays_ante=0, ndays_dry=_NDAYS_DRY):
        self.bounds = bounds
        self.ndays_ante = ndays_ante
        self.stats = None
        self.regress_day_start = None
        self.regress_day_end = None

        if bounds is None:
            self.wsum = [FMDI, ]*(ndays_ante+ndays_dry)
            self.neve = [FMDI, ]*(ndays_ante+ndays_dry)
            self.td = [FMDI, ]*(ndays_ante+ndays_dry)
        else:
            self.wsum = wsum
            self.neve = neve
            self.td = lsta - tasa
        return

    def get_days(self):
        """Return a list of day-of-dry spell indexes."""
        return list(range(-self.ndays_ante, len(self.td)-self.ndays_ante))

    def calc_rwr(self, days):
        """Calculate RWR stats over some sub-period of the composite.

        Parameters
        ----------
        days : slice
            Subperiod over which RWR is calculated.
        """

        if self.bounds is None:
            self.stats = [FMDI, ]*5
        else:
            days_c = self.get_days()
            self.stats = linregress(days_c[days], self.td[days])
            self.regress_day_start = days_c[days][0]
            self.regress_day_end = days_c[days][-1]
        return


def get_events_data(dryspells_points, var1, var2, weights,
                    ndays_dry=_NDAYS_DRY, ndays_ante=_NDAYS_ANTE,
                    bounds_ante=None):
    """
    Extract data during dry spells from the full time series.

    Parameters
    ----------
    dryspells_points : list of list of <event.Event> instances.
        Descriptions of all dry spells for each land point to be
        composited.
    var1 : list of MaskedArrays
        Full time series of variable 1 for each land point to be
        composited.  (This is typically near-surface air temperature
        anomaly.)
    var2 : list of MaskedArrays
        Full time series of variable 2 for each land point to be
        composited.  (This is typically land surface temperature
        anomaly.)
    weights : list of MaskedArrays
        Weights used in calculating the composite means.  This is
        typically the number of 1 km MODIS LST data per gridbox.
        Fill value = FMDI.
    ndays_dry : int, optional
        Number of dry spell days to composite over.
    ndays_ante : int, optional
        Number of days before dry spell started to composite over.
    bounds_ante : tuple, optional
        Return data for dry spells with antecedent precip in this
        half-open interval (bl, bu].

    Returns
    -------
    These returned lists contain views of the data that may vary for each
    dry spell depending on the duration of the dry spell and the requested
    ndays_dry.

    var1_dp : list of MaskedArrays
        var1 during dry spells and antecedent period.
    var2_dp : list of MaskedArrays
        var2 during dry spells and antecedent period.
    w_dp : list of ndarrays
        Averaging weights during dry spells and antecedent period.
    """

    var1_dp, var2_dp, w_dp = [], [], []

    # Loop over grid boxes.
    for dryspells, v1, v2, w in zip(dryspells_points, var1, var2, weights):
        for dp in dryspells:
            if bounds_ante[0] < dp.antep <= bounds_ante[1]:
                days = slice(dp.start_index - ndays_ante,
                             dp.start_index + min(ndays_dry, dp.duration))
                if days.start >= 0 and days.stop <= len(v1):
                    var1_dp.append(v1[days])
                    var2_dp.append(v2[days])
                    w_dp.append(w[days])

    return var1_dp, var2_dp, w_dp


def get_antep_bounds(events, ptiles):
    pbounds = np.linspace(0, 100, ptiles+1).tolist()
    antep = [e.antep for e in iterevents([events, ]) if e.antep > 0.0]
    if len(antep) < ptiles:
        bounds = None
    else:
        try:
            # Force a lower bound of 0.0 to ensure that the lowest antep event
            # is included in the composite.
            b = np.percentile(antep, pbounds)
            b[0] = 0.0
            bounds = list(zip(b, b[1:]))
        except ValueError:
            bounds = None
    return bounds


def get_points_data(dryspells, var1, var2, weights, points):
    """
    Extract dryspells, full time-series of var1 and var2, and averaging
    weights for a list of land points from input data on the full domain.

    The variables are typically near-surface air temperature and land
    surface temperature, and the weights are 1 km MODIS pixel counts.
    The full domain is typically global.

    Parameters
    ----------
    dryspells : list of list of <event.Event> instances.
        Descriptions of all dry spells for each land point on the full
        domain.
    var1: MaskedArray, shape(time, land)
        First input variable.
    var2: MaskedArray, shape(time, land)
        Second input variable.
    weights : ndarray, shape(time, land)
        Weights used in calculating the composite means.
    points : list of ints
       Zero-based list of land points to be extracted.

    Returns
    -------
    dryspells_p : list of list of <event.Event> instances.
        Descriptions of all dry spells for each requested land point.
    var1_p : list of MaskedArrays
        Full time series of variable 1 for each requested land point.
    var2_p : list of MaskedArrays
        Full time series of variable 2 for each requested land point.
    weights_p : list of ndarrays
        Full time series of weights for each requested land point.
    """

    dryspells_p = []
    var1_p = []
    var2_p = []
    weights_p = []

    for p in points:
        dryspells_p.append(dryspells[p])
        var1_p.append(var1[:, p])
        var2_p.append(var2[:, p])
        weights_p.append(weights[:, p])

    return dryspells_p, var1_p, var2_p, weights_p


def maskpad(x, width):
    """Return a MaskedArray from a ragged-ended input source."""
    xm = np.ma.zeros([len(x), width], dtype=x[0].dtype)
    xm.mask = True
    for k, xx in enumerate(x):
        xm[k, :len(xx)] = xx
    return xm


def get_composite(dryspells, tas_anom, lst_anom, weights, points,
                  ndays_dry=_NDAYS_DRY, ndays_ante=_NDAYS_ANTE,
                  ptiles=_PTILES):
    """
    Get composites of tas_anom and lst_anom (along with the associated td,
    RWR, Nobs, Wsum).

    Parameters
    ----------
    dryspells : list or ndarray
        Iterable containing the dry spell information for every land
        point.
    tas_anom: MaskedArray, shape(time, land)
        Near-surface air temperature anomaly (K).  Fill value = FMDI.
    lst_anom: MaskedArray, shape(time, land)
        Land surface temperature anomaly (K).  Fill value = FMDI.
    weights : ndarray, shape(time, land)
        Weights used in calculating the composite means.  This is
        typically the number of 1 km MODIS LST data per 0.5deg gridbox.
        Fill value = FMDI.
    points : list of ints
        Calculate composite mean using only dry spell events on land
        points in this list.
    ndays_dry : int, optional
        Number of dry spell days to composite over.
    ndays_ante : int, optional
        Number of days before dry spell started to composite over.
    ptiles : int, optional
        Number of antecedent precipitation percentiles over which
        composites will be calculated.  Default is 1, i.e., produce one
        composite ignoring antecedent precipitation.

    Returns
    -------
    compos : List of <Composite> instances.
        Dry spell mean composites of tas_anom and lst_anom, one for each
        of the requested antecedent precipitation percentiles.

    """

    if ptiles < 1:
        raise ValueError("Invalid ptiles: %d" % ptiles)
    elif ptiles > 20:
        raise ValueError("Too many ptiles requested: %d" % ptiles)

    # Extract events for land points of selected region.
    dryspells_p, tas_anom_p, lst_anom_p, weights_p = get_points_data(
        dryspells, tas_anom, lst_anom, weights, points)

    compos = []
    mdi_composite = Composite(None, None, None, None, None,
                              ndays_dry=ndays_dry, ndays_ante=ndays_ante)

    # Calculate antecedent precipitation percentile bounds for this region.
    antep_bounds = get_antep_bounds(dryspells_p, ptiles)

    if antep_bounds is None:
        compos.append(mdi_composite)
    else:
        # Calculate composites for each percentile.
        for bounds in antep_bounds:
            # Add events to list.
            tasa, lsta, nobs = get_events_data(dryspells_p,
                                               tas_anom_p,
                                               lst_anom_p,
                                               weights_p,
                                               ndays_dry=ndays_dry,
                                               ndays_ante=ndays_ante,
                                               bounds_ante=bounds)

            if len(nobs) == 0:
                compos.append(mdi_composite)
            else:
                # Compute composite (mean) dryspell tas and lst anomalies.
                # NB The output neve is not masked, so has neve=0 rather than
                # neve=MDI on composite days that have no data.  The tasa, lsta
                # and wsum arrays are masked, so their MDI masks in any output
                # file will differ from neve mask.
                mtasa = maskpad(tasa, ndays_ante+ndays_dry)
                mlsta = maskpad(lsta, ndays_ante+ndays_dry)
                mnobs = maskpad(nobs, ndays_ante+ndays_dry)
                mnobs.mask = mtasa.mask

                kw = dict(weights=mnobs, axis=0, returned=True)
                tasa_c, wsum = np.ma.average(mtasa, **kw)
                lsta_c, wsum = np.ma.average(mlsta, **kw)
                neve = mtasa.count(axis=0)

                compos.append(Composite(bounds, tasa_c, lsta_c, neve, wsum,
                                        ndays_ante=ndays_ante))

    return compos


def get_rwr(dryspells, tas_anom, lst_anom, weights, points_boxes,
            ndays_dry=_NDAYS_DRY, ndays_ante=_NDAYS_ANTE,
            rwr_beg=_RWR_BEG, rwr_end=_RWR_END,
            ptiles=_PTILES):
    """
    Return TD and RWR for a set small regions.

    Typically used to return TD and RWR for 1 degree regions containing
    about four grid boxes.  TD is calculated for ten dry spell days and
    five antecedent days.  RWR is calculated over dry spell days 2 to 10.
    Currently, these periods are hard-coded into this function.

    Parameters
    ----------
    dryspells : list or ndarray
        Iterable containing the dry spell information for every land
        point.
    tas_anom: MaskedArray, shape(time, land)
        Near-surface air temperature anomaly (K).  Fill value = FMDI.
    lst_anom: MaskedArray, shape(time, land)
        Land surface temperature anomaly (K).  Fill value = FMDI.
    weights : ndarray, shape(time, land)
        Weights used in calculating the composite means.  This is
        typically the number of 1 km MODIS LST data per 0.5deg gridbox.
        Fill value = FMDI.
    points_boxes : list of lists of ints
        List of regions each defined as a list of land points.  Separate
        composites are calculated for each region.
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
        Number of antecedent precipitation percentiles over which
        composites will be calculated.  Default is 1, i.e., produce one
        composite ignoring antecedent precipitation.

    Returns
    -------
    all_composites : List of lists of <Composite> instances.
        Dry spell mean composites of tas_anom and lst_anom, one for each
        of the requested antecedent precipitation percentiles.

    """

    # Indexes of days used to calculate RWR allowing for the antecedent wet
    # days in the composite.
    days_rwr = slice(ndays_ante+rwr_beg, ndays_ante+rwr_end)

    all_composites = []

    for points in points_boxes:
        composites = get_composite(
            dryspells, tas_anom, lst_anom, weights, points,
            ndays_dry=ndays_dry, ndays_ante=ndays_ante, ptiles=ptiles)

        for c in composites:
            c.calc_rwr(days_rwr)

        all_composites.append(composites)

    return all_composites
