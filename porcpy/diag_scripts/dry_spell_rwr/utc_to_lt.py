"""
A module providing functions for calculating daily time series at a
particular local solar time from sub-daily time series at UTC times.
"""

from __future__ import print_function, division

import dask
import dask.array as da
import numpy as np
import netCDF4 as nc4

from . import FMDI, logger
from . import file_io as fio

TS_INST = 0
TS_MEAN = 1
_TS_ALL = (TS_INST, TS_MEAN)


def make_daily_var(files_var, var_name, local_time, weights_func, land_grid,
                   metadata, ut_var,
                   file_reader=None, tsPad=(0, 0), tsType=TS_INST,
                   mdi=FMDI, scale=1.0, offset=0.0, file_out=None):
    """
    Driver routine for generating a daily output file for a single
    variable from sub-daily input files.

    Parameters
    ----------
    files_var : list of str
        List of input netCDF file names sorted by time.  It is assumed
        that time is the record dimension and that these files describe a
        complete time series when the data are concatenated.
    var_name : str or list of str
        Name(s) of variable(s) to be read from the input netCDF file.  If
        more than one name is passed, then the daily fields for those
        variable are added together (np.sum) before being written to the
        netCDF file.  This is typically used to create total precip output
        from separate rainfall and snowfall inputs.
    local_time : float, hours in [0., 24.).
        The local time of day that is passed to weights_func() and may be
        used as part of the daily reduction (e.g., interpolation to
        local_time, daily accumulation from local_time to local_time).
    weights_func : function
        Function that is used to calculate the reduction (e.g.,
        interpolation) weights for each land point using a 3-day window.
        Users will typically pass in other functions provided by this
        module (e.g., <dry_spell_rwr.utc_to_lt.getweights()>), but are
        free to define their own as long as those functions provide the
        same call signature.
    land_grid : <dry_spell_rwr.geogrid.LandGrid> instance
        Object that describes the input grid and the mapping from input
        grid to a vector of land points.
    metadata : dict
        Dictionary of metadata used to define the output netCDF file.
        Some common examples are defined in <dry_spell_rwr.file_io>.  The
        dictionary must contain "name", "dtype", "dims", and "fill_value"
        fields.  It may optionally contain a dict "attributes", the
        contents of which will be added as variable attributes.
    ut_var : <cftime.utime> instance
        Base time used for the time coordinate variable in the OUTPUT
        netCDF file.  It is really important that this is set correctly,
        as it's read by subsequent steps of the dry spell analysis code.
        It's up the the user to set this basetime correctly (see notes
        below).
    file_reader : function, optional
        User supplied function to read the input netCDF files.  Must have
        the same call signature as
        <dry_spell_rwr.utc_to_lt.default_file_reader()>.
    tsPad : tuple, (npre,npost), optional
        Number of times the first (last) time series field should be
        duplicated at the start (end) of the series to create a time
        series length that is an integer multiple of steps_perday (see
        notes below).
    tsType : int, optional
        Indicate whether the input variable is a time series of means
        (TS_MEAN) or instantaneous values (TS_INST, default).
    mdi : float, optional
        Missing data value of input netCDF variables. NB The missing data
        value in the output netCDF file is dry_spell_rwr.FMDI.
    scale, offset : floats, optional
        Adjustments made to the daily data (xout = x * scale + offset)
        before they are written to the output file.  NB It's best to make
        sure that the units metadata passed to this routine is consistent
        with the adjusted data.
    file_out : str, optional
        Name for output netCDF file.  If the file already exists, it will
        be clobbered.

    Notes
    -----
    The arguments weights_func, tsPad and ut_var must all be consistent
    for this driver routine to give the correct output.  Common values of
    weights_func (such as <dry_spell_rwr.utc_to_lt.getweights()>) return
    interpolation weights assuming that the subdaily time series start at
    0000 UTC.  If the time series start at a different time of day, e.g.,
    0300 UTC, then either,

      (1) tsPad should be used to force padding of the time series so that
          it starts at a time consistent with weights_func, or,
      (2) a weights_func should be passed that produces weights starting
          at the correct time of day.

    In practice, (1) is easier to do than (2).  In either case, ut_var
    should have a base time corresponding to midnight on the first date of
    the interpolated output.  This argument is never interrogated by the
    interpolation code, it's simply written into the time coordinate
    variable of the OUTPUT netCDF file.  So, two time series starting at
    0000 UTC and 0300 UTC respectively, would pass the same ut_var (e.g.,
    "days since 2000-1-1 0:0:0") and use one of the methods above to
    account for the different start times.

    """

    ###########################################################################
    # Sanity check of arguments.
    ###########################################################################
    if not 0.0 <= local_time < 24.0:
        raise ValueError("local_time must be in the range [0, 24).")

    if any(p < 0 for p in tsPad):
        raise ValueError("tsPad values must all be positive: %s" % str(tsPad))

    if tsType not in _TS_ALL:
        raise ValueError("Only tsType values %s are permitted." % str(_TS_ALL))

    if file_reader is None:
        file_reader = default_file_reader

    file_reader = dask.delayed(file_reader)

    ###########################################################################
    # Calculate the interpolation/summation weights through days [D-1, D, D+1]
    # that will be used to reduce the time series from sub-daily to daily.
    ###########################################################################
    check_weights(weights_func, local_time, tsType=tsType)

    weights = weights_func(local_time, land_grid.llon, tsType=tsType)

    land = land_grid.land_indexes

    # wg.shape == (24,NLAND)
    logger.info("Land: %s, %s, min=%s, max=%s",
                land.shape, land.dtype, land.min(axis=1), land.max(axis=1))
    logger.info("Weights: %s, %s", weights.shape, weights.dtype)

    ###########################################################################
    # Calculate the daily time series from sub-daily input data.
    ###########################################################################
    args = (mdi, weights, land, file_reader)
    kwargs = dict(tsPad=tsPad, tsType=tsType, scale=scale, offset=offset)

    if isinstance(var_name, str):
        var = get_var_atlocaltime(files_var, var_name, *args, **kwargs)
        vara = np.ma.array(var)
    else:
        var = [get_var_atlocaltime(*(fv + args), **kwargs)
               for fv in zip(files_var, var_name)]
        vara = np.ma.sum(var, axis=0)

    vara = vara.filled(metadata["fill_value"])

    if file_out:
        fio.nc_write(vara, file_out, metadata, ut_var, local_time)

    return


def default_file_reader(filename, varname, land):
    """
    Read data from a netCDF file and return it in an array with shape
    (NTIME,NLAND).

    Parameters
    ----------

    filename : str
        Input netCDF filename.
    varname : str
        Name of input variable in the netCDF file: e.g., "tas", "pr", etc.
    land : ndarray, shape(2,NLAND)
        Rows and columns of land points on the 2D model grid.

    """

    logger.info("Reading %s", filename)
    with nc4.Dataset(filename, "r") as ncid:
        var = ncid.variables[varname][:]
        if var.ndim == 3:
            var = var[:, land[0], land[1]]
    return var


def getInstantFromMeans(var, N=2, k=None):
    """
    From a time series of period means, estimate instantaneous values at
    the k-th of N sub time steps, such that the period means are
    preserved.

    Parameters
    ----------
    var : list of ndarrays
        Time series of period means.
    N : int
        Number of substeps to use when interpolating.  This must be an
        even number for the interpolation to conserve the means.
    k : int
        The substep to be returned.  Must be in the range [0, N-1].

    Returns
    -------
    varout : list of ndarrays
        Time series of instantaneous values valid at sub-step k.

    Notes
    -----

    This method was copied from JULES v4.9 file interpolate.inc for
    backward means ending at the time stamp.  Sub-step k=0 is the first
    value within the meaning period, and k=N-1 is the final step in the
    meaning period valid at the time stamp.  It looks like the input
    variable must be a strictly positive quantity, else the method become
    unstable; i.e., temperatures should be in kelvin not celsius.

    Missing data fields are returned for the first and last time steps
    because the interpolation uses data from the preceding and succeeding
    steps, which are unavailable.

    This function removes items from var once they are no longer needed,
    which will free memory if there are no external references to the
    items.

    """

    nsteps = 3

    if N < 1:
        raise ValueError("Interpolation N must be positive:" % N)
    elif N % 2 != 0:
        raise ValueError("Interpolation N must be even:" % N)

    if k is None:
        k = N - 1
    elif k < 0 or k >= N:
        raise ValueError("k must be in range (0, N-1).")

    mdi = np.ma.masked_all_like(var[0])
    mdi.set_fill_value(var[0].fill_value)

    varout = [mdi, ]
    for i in range(len(var)-(nsteps-1)):
        x0, x1, x2 = var[:nsteps]
        wx1 = x1 * (1. - abs(2*k - N + 1) / (2.*N))
        wx0 = x0 * (max(1. - (2*k + N + 1) / (2.*N), 0.))
        wx2 = x2 * (max(1. - (3*N - 2*k - 1) / (2.*N), 0.))
        xt = 4.*x1 * (wx0 + wx1 + wx2) / (0.5*x0 + 3.0*x1 + 0.5*x2)
        varout.append(xt)
        var.pop(0)
    varout += [mdi, ]
    return varout


def fix_longitudes(lons):
    """
    Return longitudes in the range (-180, 180] degE.

    The antimeridian at -180 degE is excluded from this interval and is
    returned as 180 degE.  This means that local time at the antimeridian
    is considered to be ahead of local time at the meridian.

    """
    lono = []
    for lon in lons:
        if lon == -180.0:
            lon = 180.0
        elif lon > 180.0:
            lon = lon - 360.0
        lono.append(lon)
    return np.array(lono)


def check_weights(weights_func, local_time, tsType=TS_INST, lons=None):
    """
    Print to stdout some typical UTC to LT interpolation weights as a
    function of longitude by calling a user-supplied weights function.

    """

    logger.info("Chosen local time: %s h", local_time)

    if lons is None:
        lons = (-179.0, -157.5, -115.0, -100.0, 0.0, 100.0, 115.0,
                157.5, 179.0)

    wg = weights_func(local_time, lons, tsType=tsType)

    days = ["D-1"]*8 + ["D  "]*8 + ["D+1"]*8
    times = ["0000", "0300", "0600", "0900",
             "1200", "1500", "1800", "2100"] * 3

    logger.info(("Day, Hour" + ", {:8.2f}"*len(lons)).format(*lons))
    for d, t, w in zip(days, times, wg):
        logger.info(("{:s}, {:s}" + ", {:8.4f}"*len(w)).format(d, t, *w))

    return


def getweights(time_loc_sol, lon, steps_perday=8, tsType=TS_INST):
    """
    Calculate the UTC to local time interpolation weights.

    Parameters
    ----------
    time_loc_sol : float
        Selected local solar time, e.g., MODIS-Terra view time of 10h30.
    lon : ndarray
        Vector of longitudes in degrees east.
    steps_perday : int
        Number of values per day, e.g., 3 hourly values, 8 steps per day.
    tsType : int
        Indicate whether the input variable is a time series of means
        (TS_MEAN) or instantaneous values (TS_INST).

    Returns
    -------
    weight_lon : ndarray
        Weights used to calculate instant value at local time.

    Notes
    -----
    The kindex values of 0 and 23 correspond respectively to the first
    time slot of day i-1 and the last time slot of day i+1.

    When the argument tsType=TS_INST, the weights are calculated assuming
    that the time series for interpolation start at 0000 UTC.

    When the argument tsType=TS_MEAN, the weights are calculated to
    account for the fact that get_var_atlocaltime() calls
    getInstantFromMeans() to generate an instantaneous time series with
    validity times at the mid-points of the input time series; i.e., at
    [2230, 0130, 0430,...] UTC for an input series with time stamps
    [0000, 0300, 0600,...] UTC.

    """
    steps_window = steps_perday * 3
    glob_degrees = 360.0
    hours_perday = 24.0

    # Convert longitudes to the range (-180,180] regardless of grid type.
    lon = fix_longitudes(lon)

    # Calculate the fractional indices of [0,23] that correspond to the
    # interpolation points for each longitude.  Also account for the mid-step
    # validity times that will arise if these weights are used for input time
    # series of backwards means.
    utc_time = time_loc_sol - (lon*hours_perday/glob_degrees)
    if tsType == TS_MEAN:
        utc_time += 0.5*hours_perday/steps_perday
    kindex = utc_time*steps_perday/hours_perday + steps_perday

    # Calculate the interpolation weights.  NB For each longitude no more than
    # two weights should be non-zero and their sum should be 1.
    weight_lon = np.zeros((steps_window, len(lon)), dtype=lon.dtype)
    for i, (kf, kw) in enumerate(zip(*np.modf(kindex))):
        weight_lon[int(kw), i] = 1.0 - kf
        weight_lon[int(kw)+1, i] = kf

    return weight_lon


def getweights_p(time_loc_sol, lon, steps_perday=8, tsType=TS_INST):
    """
    Calculate weights for 24-hour precip totals.

    Parameters
    ----------
    time_loc_sol : float
        Start/end local time of day for the 24 weights.
    lon : ndarray
        Vector of longitudes in degrees east.
    steps_perday : int
        Number of values per day, e.g., 3 hourly values, 8 steps per day.
    tsType : int
        Not used, just present to keep a common interface with
        getweights().

    Returns
    -------
    weight_lon : ndarray
        Weights used to calculate 24 total or means starting at a
        particular local time of day.

    Notes
    -----
    The kindex values of 0 and 23 correspond respectively to the first
    time slot of day i-1 and the last time slot of day i+1.  This routine
    also assumes that time stamps in the input file correspond to the end
    of each 3h period.  This limits how the 24h weights can be used with a
    [D-1, D, D+1] data window.

    """

    if not 0.0 <= time_loc_sol < 12.0:
        raise ValueError("Start/end time for 24 totals must be in the range "
                         "[0000,1200) LT.")

    # steps_perday = 8 # 3h data (CMIP5 and WFDEI)
    steps_window = steps_perday * 3
    glob_degrees = 360.0
    hours_perday = 24.0

    # Convert longitudes to the range (-180,180] regardless of grid type.
    lon = fix_longitudes(lon)

    # Calculate the indices of [0,23] that correspond to the interpolation
    # points for each longitude.
    utc_time = time_loc_sol - (lon*hours_perday/glob_degrees)
    kbeg = utc_time*steps_perday/hours_perday

    # Calculate the 24h total weights accounting for requested time_loc_sol
    # values that fall within the input data steps.  NB For each longitude no
    # more than nine weights should be non-zero and their sum should be 8.
    weight_lon = np.zeros((steps_window, len(lon)), dtype=lon.dtype)
    for i, kb in enumerate(kbeg):
        ib = int(np.floor(kb) + 1 + steps_perday)
        ie = int(ib + steps_perday)
        weight_lon[ib:ie, i] = 1.0
        weight_lon[ib, i] = 1.0 - (kb-np.floor(kb))
        weight_lon[ie, i] = (kb-np.floor(kb))

    return weight_lon


def var_meta(filename, varname, nland=None):
    """Return some basic variable metadata from a netCDF file."""
    with nc4.Dataset(filename, "r") as ncid:
        dtype = ncid.variables[varname].dtype
        if nland is None:
            shape = ncid.variables[varname].shape
        else:
            dname = "time" if "time" in ncid.dimensions else "tstep"
            shape = (ncid.dimensions[dname].size, nland)

    return {"shape": shape, "dtype": dtype}


def get_var_atlocaltime(ffiles, varname, fillvalue, weights_lon,
                        land_indexes, file_reader,
                        steps_perday=8, precip=0, tsPad=(0, 0),
                        tsType=TS_INST, scale=1.0, offset=0.0):
    """
    Calculate a daily local time value (snapshot, sum) from a sub-daily,
    UTC time series.

    Parameters
    ----------

    ffiles : list
        File names containing sub-daily input data.  Must be in time
        order.
    varname : str
        Name of input variable in the netCDF file: e.g., "tas", "pr", etc.
    fillvalue : float
        FillValue in the netCDF file, e,g., 1e+20 for AMIP, CMIP5.
    weights_lon : ndarray
        Weights returned from one of the getweights() functions.
    land_indexes : ndarray, shape(2,NLAND)
        Rows and columns of land points on the 2D model grid.
    file_reader : function
        User supplied function to read the input netCDF files.  Must have
        the same call signature as
        <dry_spell_rwr.utc_to_lt.default_file_reader()>.
    steps_perday : int
        Number of values per day, e.g., 3 houly values, 8 steps per day
    precip : int
        Not currently used, retained for back compatibility.
    tsPad : tuple, (npre,npost)
        Number of times the first (last) time series field should be
        duplicated at the start (end) of the series to create a time
        series length that is an integer multiple of steps_perday.
    tsType : int
        Indicate whether the input variable is a time series of means
        (TS_MEAN) or instantaneous values (TS_INST).
    scale, offset : floats, optional
        Adjustments made to the daily data (xout = x * scale + offset)
        before they are written to the output file.  NB It's best to make
        sure that the units metadata passed to this routine is consistent
        with the adjusted data.

    Returns
    -------

    varday : ndarray
        Time series of daily values adjusted to local time from UTC.

    """

    # Pre-read some metadata info to help with delayed reading of full data.
    nland = land_indexes.shape[1]
    meta = [var_meta(f, varname, nland=nland) for f in ffiles]

    # Construct a Dask virtual array of the whole time series.
    read_tasks = [file_reader(f, varname, land_indexes) for f in ffiles]
    arra_tasks = [da.from_delayed(t, **m) for t, m in zip(read_tasks, meta)]
    var = da.concatenate(arra_tasks, axis=0)

    # Apply scale and offset.
    var = var * scale
    var = var + offset

    # Manually account for time series lengths that are not an exact multiple
    # of steps_per_day by duplicating fields at the beginning and/or end.
    var_pre = [var[:1, ...], ] * tsPad[0]
    var_post = [var[-1:, ...], ] * tsPad[1]
    var = da.concatenate(var_pre + [var, ] + var_post, axis=0)

    # Estimate instantaneous time series values valid at the time step
    # mid-points that can be used to interpolate to the required daily output
    # time.
    #
    # NB This routine applies the weights assuming that the time series start
    # at 0000 UTC, so, when the input data are period means, the weights passed
    # in must account for the mid-point validity times of the estimated
    # instantaneous time series.
    if tsType == TS_MEAN:
        # lista = getInstantFromMeans(lista, N=6, k=2)
        raise NotImplementedError("TS_MEAN with Dask is not implemented")

    # Put the data into a single Numpy array with shape (NTIME,NLAND).
    logger.info("All data: %s, %s", var.shape, var.dtype)

    # Prevent upcasting of the input data when applying the weights, as the
    # weights often inherit a higher precision from the stored grid longitudes.
    weights = weights_lon.astype(var.dtype, casting="same_kind", copy=False)

    # Apply the weights to calculate a daily value (might be a sum or a
    # snapshot) including some local time adjustment.
    #
    # When a three-day window of weights is applied to a time series the
    # resulting value is assigned to the date of the middle day of the window.
    # This means that it is not possible to calculate correct values for the
    # first and last dates of the data, so they are simply padded with copies
    # of the second and second-to-last fields to give the output data the
    # expected length in days.
    win_steps = 3 * steps_perday
    win_stop = var.shape[0] - 2*steps_perday
    varday = []
    for i in range(0, win_stop, steps_perday):
        s = slice(i, i + win_steps)
        v = (var[s, :]*weights).sum(axis=0)
        varday.append(v)

    logger.info("Doing the compute.")
    varday = dask.compute(*varday)

    logger.info("Doing the concatenate.")
    varday = np.r_[varday[:1], varday, varday[-1:]]

    return varday
