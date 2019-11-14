"""
A module for reading 0.5degree MODIS land surface temperature netCDF
files.
"""

from __future__ import print_function, division

import netCDF4 as nc
from numpy import full, array
from os.path import join

FILE_TEMPLATE = "ES_MOD11_05_%s"
NL = 67209
MDI = -9999.
NPIX_MIN = 500.


def read_file(date, varnames, modis_dir):
    """Read variables from a single MODIS file."""
    fname = join(modis_dir, FILE_TEMPLATE % date.strftime("%Y%m%d"))
    try:
        with nc.Dataset(fname, "r") as ncid:
            var = [ncid.variables[v][:] for v in varnames]
    except (RuntimeError, IOError):
        var = [full((NL), MDI, dtype="f4") for v in varnames]
    return var


def read_files(dates, varnames, modis_dir):
    """Read MODIS data for a list of dates."""
    vgen = (read_file(d, varnames, modis_dir) for d in dates)
    var = [array(v) for v in zip(*vgen)]
    return var


if __name__ == "__main__":
    pass
