"""
Module containing variables and functions for standardising the format
of output from the dry_spells code, and to help reading and writing
those files.
"""

from __future__ import print_function, division

import netCDF4 as nc
import numpy as np
from cftime import utime
from scipy.stats import linregress

from . import FMDI, logger
from .climatology_composites import Composite
from .event import Event
from .geo_grid import read_grid


# Metadata for daily output files.
META_TIME = dict(name="time", dtype=np.float64, dims=("time",),
                 fill_value=FMDI,
                 attributes=dict(standard_name="time",
                                 long_name="time",
                                 missing_value=FMDI))

META_TAS = dict(name="tas", dtype=np.float32, dims=("time", "land"),
                fill_value=FMDI,
                attributes=dict(standard_name="tas",
                                long_name="air_temperature",
                                units="K",
                                cell_methods="time: point (local time)",
                                missing_value=FMDI))

META_TSLSI = dict(name="tslsi", dtype=np.float32, dims=("time", "land"),
                  fill_value=FMDI,
                  attributes=dict(standard_name="surface_temperature",
                                  long_name="surface_temperature_where_land",
                                  units="K",
                                  cell_methods="time: point (local time)",
                                  missing_value=FMDI))

META_PR = dict(name="pr", dtype=np.float32, dims=("time", "land"),
               fill_value=FMDI,
               attributes=dict(standard_name="precipitation_amount",
                               long_name="precipitation_amount",
                               units="kg m-2",
                               cell_methods="time: sum (local time)",
                               missing_value=FMDI))

META_RSDS = dict(name="rsds", dtype=np.float32, dims=("time", "land"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="surface_downwelling_shortwave_flux_in_air",  # noqa
                                 long_name="surface_downwelling_shortwave_flux_in_air",  # noqa
                                 units="W m-2",
                                 cell_methods="time: point (local time)",
                                 missing_value=FMDI))

META_RSDSCS = dict(name="rsdscs", dtype=np.float32, dims=("time", "land"),
                   fill_value=FMDI,
                   attributes=dict(standard_name="surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",  # noqa
                                   long_name="surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",  # noqa
                                   units="W m-2",
                                   cell_methods="time: point (local time)",
                                   missing_value=FMDI))

META_SMCL = dict(name="smcl", dtype=np.float32, dims=("time", "land"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="smcl",
                                 long_name="soil_moisture_content",
                                 units="kg m-2",
                                 cell_methods="time: point (local time)",
                                 missing_value=FMDI))

META_HFLS = dict(name="hfls", dtype=np.float32, dims=("time", "land"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="hfls",
                                 long_name="surface_upward_latent_heat_flux",
                                 units="W m-2",
                                 cell_methods="time: point (local time)",
                                 missing_value=FMDI))

META_HFSS = dict(name="hfss", dtype=np.float32, dims=("time", "land"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="hfss",
                                 long_name="surface_upward_sensible_heat_flux",
                                 units="W m-2",
                                 cell_methods="time: point (local time)",
                                 missing_value=FMDI))

META_EF = dict(name="ef", dtype=np.float32, dims=("time", "land"),
               fill_value=FMDI,
               attributes=dict(standard_name="ef",
                               long_name="evaporative_fraction",
                               units="1",
                               cell_methods="time: point (local time)",
                               missing_value=FMDI))

# Metadata for RWR output files.
META_RWR = dict(name="rwr", dtype=np.float32, dims=("lat", "lon"),
                fill_value=FMDI,
                attributes=dict(standard_name="relative_warming_rate",
                                long_name="relative_warming_rate",
                                units="K/day",
                                missing_value=FMDI))

META_INT = dict(name="intercept", dtype=np.float32, dims=("lat", "lon"),
                fill_value=FMDI,
                attributes=dict(standard_name="intercept",
                                long_name="intercept",
                                units="K",
                                missing_value=FMDI))

META_RVAL = dict(name="rval", dtype=np.float32, dims=("lat", "lon"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="correlation_coefficient",
                                 long_name="correlation_coefficient",
                                 units="1",
                                 missing_value=FMDI))

META_PVAL = dict(name="pval", dtype=np.float32, dims=("lat", "lon"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="p_value",
                                 long_name="p_value",
                                 units="1",
                                 missing_value=FMDI))

META_SE = dict(name="stderr", dtype=np.float32, dims=("lat", "lon"),
               fill_value=FMDI,
               attributes=dict(standard_name="standard_error",
                               long_name="standard_error",
                               units="K/day",
                               missing_value=FMDI))

META_TD = dict(name="td", dtype=np.float32, dims=("time", "lat", "lon"),
               fill_value=FMDI,
               attributes=dict(standard_name="td",
                               long_name="tslsi_anomaly_minus_tas_anomaly",
                               units="K",
                               missing_value=FMDI))

META_NEVE = dict(name="neve", dtype=np.float32, dims=("time", "lat", "lon"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="neve",
                                 long_name="number_of_dry_spells",
                                 units="count",
                                 missing_value=FMDI))

META_WSUM = dict(name="wsum", dtype=np.float32, dims=("time", "lat", "lon"),
                 fill_value=FMDI,
                 attributes=dict(standard_name="wsum",
                                 long_name="sum_of_weights_in_td_mean",
                                 units="count",
                                 missing_value=FMDI))

META_DAY = dict(name="time", dtype=np.float32, dims=("time", ),
                fill_value=FMDI,
                attributes=dict(standard_name="time",
                                long_name="time",
                                units="days_since_start_of_dry_spell",
                                missing_value=FMDI))

metadata_rwr = (META_RWR, META_INT, META_RVAL, META_PVAL, META_SE,
                META_TD, META_NEVE, META_WSUM, META_DAY)


def nc_define_var(ncid, metadata):
    """
    Define a new variable in an open netCDF file and add some default
    attributes.

    Parameters
    ----------

    ncid: netCDF4.Dataset
        Handle to an open netCDF file.
    metadata: dict
        Bundle containing the information needed to define the variable.
        The minimum is {name; dtype, dimensions; fill_value; attributes}.
        The contents of the attributes dictionary are all added to the
        variable.

    Returns
    -------

    var_id: netCDF4.Variable
        Handle to a netCDF4 variable.

    """
    var_id = ncid.createVariable(metadata["name"], metadata["dtype"],
                                 metadata["dims"], zlib=True,
                                 fill_value=metadata["fill_value"])
    for item in metadata["attributes"].items():
        var_id.__setattr__(*item)
    return var_id


def nc_write(var, fileout, metadata, ut_var, local_time):
    """
    Write out data to a new netCDF file using a standardised format for
    the data and variable names.

    Parameters
    ----------

    var: ndarray, shape(ntime, nland)
        Variable to be written to the file.  This is typically a daily
        variable at local time.
    fileout: str
        Name of the netCDF output file. Any existing file with this name
        will be clobbered.
    metadata: dict
        Bundle containing the information needed to define the variable.
        See nc_define_var() for details.
    ut_var: <cftime.utime> instance
        Object that defines the time axis of the variable to be output.
        This is very important, as it is used for metadata that will be
        read by subsequent steps of the dry spell code.  This should be
        defined using a basetime that corresponds to the first time index
        of var.  For example, "days since 2000-1-1" would imply that
        var[0,:] is valid on 2000-1-1 at local_time.
    local_time: float, hours.
        The local time of day at which var is valid.  This is used to
        calculate values of the time coordinate variable.

    """

    if ut_var.units != "days":
        raise ValueError("Output time units must be of type 'days since...'.")
    if not 0.0 <= local_time < 24.0:
        raise ValueError("local_time must be in the range [0, 24)")

    # Ensure that the output variables are setup with the same data type as the
    # incoming data.
    if np.dtype(metadata["dtype"]) != var.dtype:
        logger.info("Setting output type to %s", var.dtype)
        metadata = metadata
        metadata["dtype"] = var.dtype

    (ntime, nland) = var.shape

    ncid = nc.Dataset(fileout, "w")
    dim_time = ncid.createDimension("time", ntime)
    dim_land = ncid.createDimension("land", nland)

    var_time = nc_define_var(ncid, META_TIME)
    var_time.units = ut_var.unit_string
    var_time.calendar = ut_var.calendar
    var_time[:] = np.arange(ntime) + local_time/24.0

    var_var = nc_define_var(ncid, metadata)
    var_var[:] = var[:]

    ncid.close()

    return


def nc_read_var(var_file, var_name):
    """
    Read a variable from a netCDF file and throw away length 1 dimensions.

    Parameters
    ----------

    var_file : str
        Input netCDF file name.
    var_name : str
        Input netCDF variable name.

    Returns
    -------
    var : ndarray or masked_array

    """
    logger.info("Reading %s from %s", var_name, var_file)

    with nc.Dataset(var_file, "r") as ncid:
        vid = ncid.variables[var_name]
        var = vid[:].squeeze()
        var_units = vid.units

    logger.info("%s %s (min, med, max): %5.2f, %5.2f, %5.2f",
                var_name,
                var_units,
                *tuple(np.percentile(np.ma.compressed(var), [0, 50, 100])))

    return var


def _unpack(xin, attr):
    """Hack function to make repacking from Composite class cleaner."""
    return np.ma.array([x[0].__getattribute__(attr) for x in xin]).T


def nc_rwr_write(grid, composites, file_out):
    """
    Write out TD and RWR data on a spatial grid to a netCDF file.

    Parameters
    ----------

    grid : <geo_grid.LandGrid> instance.
        Object describing the grid on which the composites list is valid.
    composites : list of lists of <.Composite> instances.
        Composite TD for each land point on the grid.
    file_out : str
        Output netCDF file name.  Any existing file will be clobbered.
    """

    # Use the first land point that has RWR defined to retrieve info about the
    # regression time axis.
    for c in composites:
        if c[0].regress_day_start is not None:
            c0 = c[0]
            break

    days = c0.get_days()

    td_tmp = _unpack(composites, "td")
    neve_tmp = _unpack(composites, "neve")
    wsum_tmp = _unpack(composites, "wsum")
    stats_tmp = _unpack(composites, "stats")

    td_out = np.ma.array([grid.expand(td) for td in td_tmp])
    neve_out = np.ma.array([grid.expand(n) for n in neve_tmp])
    wsum_out = np.ma.array([grid.expand(w) for w in wsum_tmp])
    stats_out = [grid.expand(s) for s in stats_tmp]
    stats_out += [td_out, neve_out, wsum_out, days]

    dims_extra = [("time", neve_out.shape[0]), ]
    gattr = dict(regress_day_start=np.int32(c0.regress_day_start),
                 regress_day_end=np.int32(c0.regress_day_end))

    grid.write_vars(file_out, stats_out, metadata_rwr, dims_extra=dims_extra,
                    gattr=gattr)

    return


def nc_rwr_combine(grid_file, files_in, file_out):
    """
    Merge the data from several netCDF files of TD and RWR.

    These files must be in the format produced by nc_rwr_write().  This
    function merges the composite TD for each grid box from each file
    using the weightings and then calculates the RWRs of these new, merged
    composites.  The input files must all be consistent with the input
    grid file, but this is not checked explicitly.

    Parameters
    ----------

    grid_file : str.
        Name of netCDF file describing the input grid.
    files_in : list of str.
        List of TD/RWR files to be merged.
    file_out : str.
        Output netCDF file name.  Any existing file will be clobbered.
    """

    # Read the grid file.
    grid = read_grid(grid_file)

    # Initialise the output arrays.
    fi = files_in[0]
    td_all = np.zeros_like(nc_read_var(fi, "td"))
    neve_all = np.zeros_like(nc_read_var(fi, "neve"))
    wsum_all = np.zeros_like(nc_read_var(fi, "wsum"))

    td_all.mask = np.ma.nomask
    neve_all.mask = np.ma.nomask
    wsum_all.mask = np.ma.nomask

    some_unmasked = np.ma.nomask

    # Read the time axis for the RWR regression.
    with nc.Dataset(fi, "r") as ncid:
        days = ncid.variables["time"][:].squeeze()
        regress_day_start = ncid.regress_day_start
        regress_day_end = ncid.regress_day_end
        ndays_ante = sum(days < 0)

    # Read and combine TD, Neve, Wsum.
    for fi in files_in:
        td = nc_read_var(fi, "td")
        neve = nc_read_var(fi, "neve")
        wsum = nc_read_var(fi, "wsum")

        some_unmasked += ~neve.mask

        td_all += td.filled(0)*wsum.filled(0)
        neve_all += neve.filled(0)
        wsum_all += wsum.filled(0)

    # Mask out elements that are masked in _all_ input neve arrays.
    td_all.mask += ~some_unmasked
    neve_all.mask += ~some_unmasked
    wsum_all.mask += ~some_unmasked

    # Mask out elements that have events but no data.
    no_pixels = (wsum_all == 0)
    td_all.mask += no_pixels
    wsum_all.mask += no_pixels

    # Finalise the weighted composite mean.
    td_all /= wsum_all

    # Calculate stats on the new combined TD.
    def calc_rwr(td):
        if all(td.mask):
            stats = [FMDI, ] * 5
        else:
            s = slice(ndays_ante + regress_day_start,
                      ndays_ante + regress_day_end + 1)
            stats = linregress(days[s], td[s])
        return stats

    stats_out = [s for s in np.apply_along_axis(calc_rwr, 0, td_all)]
    stats_out += [td_all, neve_all, wsum_all, days]

    # Write out stats.
    dims_extra = [("time", neve_all.shape[0]), ]
    gattr = dict(history="nc_rwr_combine of %s." % ", ".join(files_in),
                 regress_day_start=regress_day_start,
                 regress_day_end=regress_day_end)

    grid.write_vars(file_out, stats_out, metadata_rwr,
                    dims_extra=dims_extra, gattr=gattr)

    return


def event_count(events):
    """Returns the total number of events in multiple events lists."""
    return sum([len(e) for e in events])


def grid_lengths_equal(events):
    """Check whether multiple events lists come from grids with the same
    number of land points."""
    if events:
        nland = len(events[0])
        unequal = [len(e) != nland for e in events[1:]]
        if any(unequal):
            raise ValueError("Grid lengths must be equal " +
                             "for all events lists.")
    else:
        raise ValueError("No events lists were passed to function.")


def asc_events_write(events, events_notes, utime, file_out):
    """
    Write some existing lists of dry spell events to a text file in a
    format that can be read to recreate the lists.  All events must share
    the same base time.

    Parameters
    ----------

    events : list of lists of <event.Event> instances
        List of lists of dry spell events.
    event_notes : list of str
        A comment string for each list of dry spell events, written to the
        header of those events.
    utime : <cftime.utime> instance
        Base time of the dry spells in events.  The start_index of each
        Event is associated with this base time.
    file_out : str
        Output file name.  Any existing file will be clobbered.
    """

    grid_lengths_equal(events)

    out_pattern = "{:d}, {:d}, {:d}, {:g}\n"

    header = "{:s}, {:s}\n".format(utime.unit_string, utime.calendar)
    header += "{:d}\n".format(len(events[0]))
    header += ", ".join([str(event_count(e)) for e in events]) + "\n"

    with open(file_out, "w") as fo:
        fo.write(header)

        for eves, note in zip(events, events_notes):
            fo.write(">>>>> {:s}\n".format(note))
            for land in eves:
                for e in land:
                    fo.write(out_pattern.format(
                        e.land, e.start_index, e.duration, e.antep))


def asc_events_read(file_in):
    """
    Read an existing dry spell events file and return it as a list of
    event.Event objects.

    Parameters
    ----------

    file_in : str
        Name of file containing a catalogue dry spell events in the format
        written by file_io.asc_events_write().

    Returns
    -------
    events : list of lists of <event.Event> instances
        List of dry spell events.
    ut_events : <cftime.utime> instance
        Base time of the dry spells in events.  The start_index of each
        Event is associated with this base time.

    """

    with open(file_in, "r") as fi:
        logger.info("Reading dry spell events from %s", file_in)

        basetime, calendar = fi.readline().split(",")
        ut_events = utime(basetime, calendar=calendar.strip())

        nland = int(fi.readline())

        events = []
        nevents = [int(n.strip()) for n in fi.readline().split(",")]
        for n in nevents:
            logger.info(fi.readline())

            events_sub = [[] for _ in range(nland)]
            for _ in range(n):
                line = fi.readline().split(",")
                args = (int(line[0]), int(line[1]),
                        ut_events,
                        int(line[2]), float(line[3]))
                E = Event(*args)
                events_sub[E.land].append(E)

        events.append(events_sub)

    return events, ut_events


def asc_comp_write(var, neve, wsum, dry_spell_days, file_out,
                   var_name="Composite", header=None):
    """
    Write a single composite to a text file.

    Parameters
    ----------

    var : ndarray, shape(N)
        Composite data for N days.
    neve : ndarray, shape(N)
        Number of values contributing to each composite day.
    dry_spell_days : int
        Number of the N days that are part of the dry spell.  The number
        of antecedent wet days is N-dry_spell_days.  E.g., if N=6 and
        dry_spell_days=4 then var is split between wet and dry days like
        WWDDDD.
    file_out : str
        Output file name.

    var_name : str, optional
        String to be used as a column header for var in the output file.
    header : str, optional
        String to be written as a header at the top of the output file.
        This may contain newlines, and each line in the output file will
        be preceded by a '#'.
    """

    # Column for day of dry spell.
    ncomp = len(var[0])
    days = list(range(dry_spell_days - ncomp, ncomp))

    # Rearrange the incoming data for easier writing with savetxt().
    varx = [days, ]
    for v, n, w in zip(var, neve, wsum):
        varx.append(v)
        varx.append(n)
        varx.append(w)
    varz = [z for z in zip(*varx)]

    # Add column heading to any user-defined header.
    if header is None:
        header = ""
    else:
        header += "\n"

    header += "Day   " + "%10s nevents        wsum" % var_name * len(var)
    fmt = "%5d,  " + "%10.7f, %6d, %10d  " * len(var)

    # Write the composite data out to a text file.
    np.savetxt(file_out, varz, fmt=fmt, header=header)

    return


def asc_comp_read(filename, read_header=False):
    """
    Read a composite text file in the format produced by asc_comp_write().

    Parameters
    ----------
    filename : str
        Input text file name.
    read_header : bool, optional
        If True, the file header will also be returned, but with the '# '
        prefixes removed.

    Returns
    -------
    dayz : ndarray, shape(days)
        Day of dry spell.
    var : ndarray, shape(days)
        Composite mean variable.
    n : ndarray, shape(days)
        Number of values in mean.
    wsum : ndarray, shape(days)
        Sum of meaning weights.
    """

    dayz, var, n, wsum = np.genfromtxt(filename, delimiter=",", skip_header=0,
                                       unpack=True)

    if read_header:
        header = ""
        with open(filename, "r") as fi:
            header += fi.readline().strip("# ")
        ret = (dayz, var, n, wsum, header)
    else:
        ret = (dayz, var, n, wsum)

    return ret


def asc_comp_combine(files_in, file_out):
    """
    Merge the composite means from several text files.  Input files must
    be in the format produced by asc_comp_write().

    Parameters
    ----------
    files_in : list of str
        List of composite mean text files to be merged.
    file_out : str
        Output file name.  Any existing file will be clobbered.

    """
    days = np.loadtxt(files_in[0], usecols=(0,), delimiter=",")
    ndry = sum(days >= 0)

    var_all = np.zeros(len(days), dtype=float)
    n_all = np.zeros(len(days), dtype=int)
    wsum_all = np.zeros(len(days), dtype=int)
    header = ""
    for f in files_in:
        logger.info("Reading %s", f)

        dayz, var, n, wsum, head = asc_comp_read(f, read_header=True)

        var_all += var*wsum
        n_all += n.astype(n_all.dtype)
        wsum_all += wsum.astype(wsum_all.dtype)
        header += head

    var_all /= wsum_all

    asc_comp_write([var_all, ], [n_all, ], [wsum_all, ], ndry, file_out,
                   header=header)

    return


def asc_ptiles_write(composites, file_out, header=None):
    """
    Write composite TD and RWR for several antecedent precipitation
    percentile bins to a single text file.

    Parameters
    ----------
    composites : list of <climatology_composites.Composite> instances.
        Composite TD for each percentile bin.
    file_out : str
        Output file name.  Any existing file will be clobbered.
    header : str, optional
        String to be written as a header to the output file.  Each line
        should begin with '#' else subsequent reads using
        asc_ptiles_read() will fail.
    """

    if header is None:
        header = ""
    else:
        header += "\n"

    header += ("%d antecedent precipitation percentiles.\n" %
               len(composites))

    c = composites[0]
    header += ", ".join("%d" % d for d in c.get_days()) + "\n"
    header += ("%d, %d, regression start/end days.\n" %
               (c.regress_day_start, c.regress_day_end))

    with open(file_out, "w") as fo:
        fo.write(header)
        for c in composites:
            fo.write(", ".join("%g" % b for b in c.bounds) + "\n")
            fo.write(", ".join("%g" % s for s in c.stats) + "\n")
            fo.write(", ".join("%g" % td for td in c.td) + "\n")
            fo.write(", ".join("%d" % n for n in c.neve) + "\n")
            fo.write(", ".join("%d" % n for n in c.wsum) + "\n")
    return


def _readlinev(f, dtype=float):
    return [dtype(x.strip()) for x in f.readline().split(",")]


def asc_ptiles_read(file_in):
    """
    Read a antecedent precipitation percentiles composite text file in the
    format produced by asc_ptiles_write().

    Parameters
    ----------
    file_in : str
        Input text file name.

    Returns
    -------
    header : str
        Input file header including '#' prefixes.
    composites : list of <climatology_composites.Composite> instances.
        Composite TD for each percentile bin.
    """
    with open(file_in, "r") as fi:
        # Skip header lines.
        header = ""
        line = fi.readline()
        while line.startswith("#"):
            header += line
            line = fi.readline()

        nptiles = int(line.split(" ")[0])

        days = _readlinev(fi, int)
        ndays = len(days)
        ndays_dry = days[-1] + 1
        ndays_ante = ndays - ndays_dry

        line = fi.readline().split(",")[:2]
        regress_day_start, regress_day_end = int(line[0]), int(line[1])

        composites = []
        for _ in range(nptiles):
            bounds = _readlinev(fi)
            stats = _readlinev(fi)
            var = np.array(_readlinev(fi))
            n = np.array(_readlinev(fi, int))
            wsum = np.array(_readlinev(fi, int))
            composites.append(Composite(bounds, var*0.0, var, n, wsum,
                                        ndays_ante, ndays_dry))

        for c in composites:
            c.regress_day_start = regress_day_start
            c.regress_day_end = regress_day_end

    return header, composites


def asc_ptiles_compress(file_in, file_out):
    """
    Merge antecedent precipitation percentile composites within a file to
    produce a single composite file.  The input file must be in the format
    produced by asc_ptiles_write().

    Parameters
    ----------
    file_in : str
        Input text file name.
    file_out : str
        Output file name.  Any existing file will be clobbered.
    """

    header, composites = asc_ptiles_read(file_in)

    # Rm any header indicators because the write will add them back in.
    header = header.replace("# ", "")

    c0 = composites[0]
    var_all = np.zeros_like(c0.td)
    n_all = np.zeros_like(c0.neve)
    wsum_all = np.zeros_like(c0.wsum)
    ndays_dry = len(c0.td) - c0.ndays_ante

    for c in composites:
        var_all += c.td*c.wsum
        n_all += c.neve
        wsum_all += c.wsum

    var_all /= wsum_all

    asc_comp_write([var_all, ], [n_all, ], [wsum_all, ], ndays_dry,
                   file_out, header=header)

    return


def asc_ptiles_combine(files_in, file_out):
    """
    Merge antecedent precipitation percentile composite means from several
    text files.  Input files must be in the format produced by
    asc_ptiles_write().

    files_in : list of str
        List of antecedent precipitation percentile composite mean text
        files to be merged.
    file_out : str
        Output file name.  Any existing file will be clobbered.
    """

    # Read all of the input files.
    composites_in = []
    header = ""
    for f in files_in:
        head, comp = asc_ptiles_read(f)
        header += head
        composites_in.append(comp)
    header = header[:-1]

    # Use data from the first file to setup an output structure.
    composites_out = []
    for cin in composites_in[0]:
        cout = Composite(cin.bounds, cin.td*0.0, cin.td*0.0,
                         cin.neve*0, cin.wsum*0,
                         cin.ndays_ante, len(cin.td))
        cout.regress_day_start = cin.regress_day_start
        cout.regress_day_end = cin.regress_day_end
        composites_out.append(cout)

    # Increment the output data.
    for cins in composites_in:
        for cin, cout in zip(cins, composites_out):
            cout.td += cin.td*cin.wsum
            cout.neve += cin.neve
            cout.wsum += cin.wsum

    # Finalise the output data and calculate RWR.
    for cout in composites_out:
        cout.td /= cout.wsum
        days_rwr = slice(cout.ndays_ante + cout.regress_day_start,
                         cout.ndays_ante + cout.regress_day_end + 1)
        cout.calc_rwr(days_rwr)

    # Write to file using the library function.
    asc_ptiles_write(composites_out, file_out, header=header)

    return


def files_exist(files):
    """Check whether all file names in a list are accessible for reading.
    """
    for f in files:
        with open(f, "r") as fi:
            pass
    return True


if __name__ == "__main__":
    pass
