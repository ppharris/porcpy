"""
A module that (1) defines the LandGrid class which is used for mapping
between 2D lon/lat grids and vectors of land points, (2) functions for
reading/writing netCDF files of LandGrid instances.
"""

from __future__ import print_function, division

from cartopy.mpl.geoaxes import GeoAxes
import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from itertools import count

from . import logger


class LandGrid(object):
    """
    A mapping between land points and a lon/lat grid derived from a land
    fraction field.

    """

    def __init__(self, filename, lons, lats, land_indexes):
        self.filename = filename

        self.land_indexes = np.int32(land_indexes)
        self.rows, self.cols = self.land_indexes

        self.lons = lons[:]
        self.lats = lats[:]

        self.llon = self.lons[self.cols]
        self.llat = self.lats[self.rows]

        self.nc, self.nr = len(self.lons), len(self.lats)
        self.nl = len(self.rows)

    def __len__(self):
        return self.nl

    def reduce(self, var):
        """Return an input field compressed to a vector of land points."""
        if var.shape != (self.nr, self.nc):
            raise ValueError("Data must be on 2D lon/lat grid to reduce.")
        return var[self.rows, self.cols]

    def expand(self, var, cube=False, var_name="Unknown"):
        """Return an input vector expanded to it's native lon/lat grid."""
        if var.shape != (self.nl,):
            raise ValueError("Data must be on 1D land vector to expand.")
        varo = np.ma.masked_all((self.nr, self.nc), dtype=var.dtype)
        varo[self.rows, self.cols] = var[:]
        if cube:
            varo = self.get_cube(varo, var_name=var_name)
        return varo

    def get_cube(self, var, var_name="Unknown"):
        """Return the input ndarray as an iris cube based on this LandGrid."""
        dim_lon = iris.coords.DimCoord(self.lons, standard_name="longitude",
                                       units="degrees")
        dim_lat = iris.coords.DimCoord(self.lats, standard_name="latitude",
                                       units="degrees")

        dim_lon.guess_bounds()
        dim_lat.guess_bounds()

        cube = iris.cube.Cube(var, var_name=var_name)
        cube.add_dim_coord(dim_lat, 0)
        cube.add_dim_coord(dim_lon, 1)

        return cube

    def plot_var(self, ax, var, kw_plot=None, labelled=True):
        """Quickly plot a map to an existing GeoAxes instance."""

        if not isinstance(ax, GeoAxes):
            raise TypeError("plot_var() requires Cartopy GeoAxes.")

        if isinstance(var, iris.cube.Cube):
            cube = var
        else:
            cube = self.get_cube(var)
        if kw_plot is None:
            kw_plot = dict()

        plt.sca(ax)
        PCM = iplt.pcolormesh(cube, **kw_plot)
        ax.coastlines(resolution="50m", lw=0.5)

        if labelled:
            try:
                ggl = ax.gridlines(color="0.5",
                                   linestyle="--",
                                   draw_labels=True)
                ggl.xlabels_top = False
                ggl.ylabels_right = False
                ggl.xlabel_style = {"size": 10}
                ggl.ylabel_style = {"size": 10}
            except TypeError as e:
                # Catch attempts to gridline unsupported projections.
                logger.warn(str(e))

        return PCM

    def write_vars(self, filename, var, metadata, dims_extra=None,
                   gattr=None):
        """
        Create a new netCDF file containing multiple variables on this
        LandGrid.  Variables are output on the lon/lat grid rather than a
        vector of land points.

        Parameters
        ----------
        filename : str
            Name of netCDF output file to be created.
        var : list of ndarrays or MaskedArrays
            Data to be written to the file.
        metadata : dict
            Bundle containing the information needed to define the
            variable.  The minimum is {name; dtype, dimensions;
            fill_value; attributes}.  The contents of the attributes
            dictionary are all added to the variable.
        dims_extra : list of tuples
            Additional non-lon/lat dimensions that are required by some
            of the output vars. Format is, e.g.,
            dims_extra=[("time", ntim), ].
        gattr : dict
            A dictionary of items to be written as netCDF global
            attributes.

        """

        from .file_io import nc_define_var

        ncid = nc.Dataset(filename, "w")

        dim_lon = ncid.createDimension("lon", self.nc)
        dim_lat = ncid.createDimension("lat", self.nr)
        if dims_extra is not None:
            for dim in dims_extra:
                dim_id = ncid.createDimension(*dim)

        var_lon = ncid.createVariable(dim_lon._name, self.lons.dtype,
                                      (dim_lon._name,))
        var_lon.standard_name = "longitude"
        var_lon.long_name = "longitude"
        var_lon.units = "degrees_east"
        var_lon.axis = "X"
        var_lon[:] = self.lons

        var_lat = ncid.createVariable(dim_lat._name, self.lats.dtype,
                                      (dim_lat._name,))
        var_lat.standard_name = "latitude"
        var_lat.long_name = "latitude"
        var_lat.units = "degrees_north"
        var_lat.axis = "Y"
        var_lat[:] = self.lats

        for v, m in zip(var, metadata):
            vid = nc_define_var(ncid, m)
            vid[:] = v

        if gattr is not None:
            for item in gattr.items():
                ncid.__setattr__(*item)

        ncid.close()

        return


def read_grid(filename, land_name="sftlf", land_thresh=70.0):
    """
    Create a LandGrid instance by reading the definition from a netCDF
    file.

    Parameters
    ----------

    filename : str
        Input netCDF file name.
    land_name : str, optional
        Name of land fraction variable in the netCDF file.
    land_thresh : float, optional
        Minimum land fraction for a grid box to be considered a land
        point.

    Returns
    -------
    land_grid : <dry_spell_rwr.geo_grid.LandGrid> instance

    """

    ncid = nc.Dataset(filename, 'r')
    lons = ncid.variables["lon"]
    lats = ncid.variables["lat"]
    landf = ncid.variables[land_name]

    # Some CMIP models report sftlf as a fraction [0,1] rather than a
    # percentage.
    if landf[:].max() <= 1.0:
        land_thresh = land_thresh * 0.01

    # Some CMIP models have a length-1 time dimension.
    if "time" in landf.dimensions:
        landf = landf[0, ...]

    land_indexes = np.where(landf[:] > land_thresh)

    # If landpoint vector is present in the grid file, then the order of
    # the land points compression needs to be changed to match some
    # existing, external definition.
    if "landpoint" in ncid.variables:
        land = ncid.variables["landpoint"][:]
        if len(land) != len(land_indexes[0]):
            raise ValueError("landpoint mapping inconsistent with the "
                             "number of land points derived from land "
                             "fraction.")

        # Create a mapping from global grid point -> np.where land point.
        lookup = {r*len(lons)+c: k
                  for k, r, c in zip(count(), *land_indexes)}

        # Use that lookup and the np.where land indexes to get a mapping
        # from stored land point index -> global (row, col).
        ll = [lookup[l] for l in land]
        land_indexes = (land_indexes[0][ll], land_indexes[1][ll])

    land_grid = LandGrid(filename, lons, lats, land_indexes)

    ncid.close()

    return land_grid


def write_grid_file(land_frac, lons, lats, filename, land_points=None,
                    gattr=None):
    """
    Create a new land fraction file in a format similar to the fixed
    climate model grid information files in the CMIP5 archive.

    Parameters
    ----------

    land_frac: ndarray, shape(len(lats), len(lons))
        Fraction of grid box that is land (%).  This should be zero rather
        than _FillValue where there is no land, i.e., 100% ocean points.
    lons: ndarray
        Longitudes of each column on the global 2D grid (degrees East).
    lats: ndarray
        Latitudes of each row on the global 2D grid (degrees North).
    filename: str
        Output netCDF file name.  If this file exists it will be clobbered
        by this function.
    land_points: ndarray, dtype=int
        The order that land points should be stored in when compressed
        from (lat, lon) grid to a vector of land points.  The land point
        indexes correspond to the standard compression scheme used by this
        package: i.e., column (longitude) varying fastest starting at the
        lower left (south-west) corner of the domain.  For more details
        see [1].

        This argument is only needed if the desired compression scheme
        differs from the assumed standard, e.g., when using the WFDEI
        input data.

        [1] http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#compression-by-gathering # noqa

    gattr: dict
        A dictionary of items to be written as netCDF global attributes.

    """

    ncol, nrow = len(lons), len(lats)
    out_shape = (nrow, ncol)

    if land_frac.shape != out_shape:
        raise ValueError("Land fraction must already be on lon/lat grid.")

    ncid = nc.Dataset(filename, "w")
    dim_lon = ncid.createDimension("lon", ncol)
    dim_lat = ncid.createDimension("lat", nrow)

    kw = dict(zlib=True, complevel=1, shuffle=False)

    var_lon = ncid.createVariable(dim_lon._name, lons.dtype, (dim_lon._name,),
                                  **kw)
    var_lon.standard_name = "longitude"
    var_lon.long_name = "longitude"
    var_lon.units = "degrees_east"
    var_lon.axis = "X"
    var_lon[:] = lons

    var_lat = ncid.createVariable(dim_lat._name, lats.dtype, (dim_lat._name,),
                                  **kw)
    var_lat.standard_name = "latitude"
    var_lat.long_name = "latitude"
    var_lat.units = "degrees_north"
    var_lat.axis = "Y"
    var_lat[:] = lats

    var_lfrac = ncid.createVariable("sftlf", land_frac.dtype, ("lat", "lon"),
                                    **kw)
    var_lfrac.standard_name = "land_area_fraction"
    var_lfrac.long_name = "land_area_fraction"
    var_lfrac.units = "%"
    var_lfrac[:] = land_frac[:]

    if land_points is not None:
        dim_land = ncid.createDimension("landpoint", len(land_points))
        var_land = ncid.createVariable(dim_land._name, land_points.dtype,
                                       (dim_land._name,), **kw)
        var_land.compress = "%s %s" % (dim_lat._name, dim_lon._name)
        var_land[:] = land_points[:]

    if gattr is not None:
        for item in gattr.items():
            ncid.__setattr__(*item)

    ncid.close()

    return


if __name__ == "__main__":
    pass
