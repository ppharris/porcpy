"""
Module of helper functions for working with the 0.5deg WFDEI land grid.

http://www.eu-watch.org/gfx_content/documents/README-WFDEI.pdf

Weedon et al (2014), The WFDEI meteorological forcing data set: WATCH
Forcing Data methodology applied to ERA-Interim reanalysis data, Water
Resources Res., doi:10.1002/2014WR015638.
"""

from __future__ import print_function, division

from pkg_resources import resource_filename

import numpy as np

from . import IMDI
from .geo_grid import read_grid, LandGrid


file_sftlf_wfdei = resource_filename(__package__, "data/sftlf_fx_WFD-EI.nc")
file_sftlf_wfdei_1d = resource_filename(__package__,
                                        "data/sftlf_fx_WFD-EI_1deg.nc")

NC_WFDEI = 720    # Number of columns in wfdei grid
NR_WFDEI = 360    # Number of rows in wfdei grid
NL_WFDEI = 67209  # Number of land points in WFD-EI global grid
DLON_WFDEI = 0.5
DLAT_WFDEI = 0.5


def read_grid_wfdei():
    """Return a <geo_grid.LandGrid> for the WFDEI grid."""
    return read_grid(file_sftlf_wfdei)


def regrid_wfdei(res):
    """
    Regrid WFDEI from original grid to a new resolution

    Parameters
    ----------
    res : int
        new resolution (e.g. 10 deg)

    Returns
    -------
    coarse_grid : <geo_grid.LandGrid> instance
        Object that describes the coarsened version of the WFDEI grid.
    ListLP: list
        list of WFDEI land points per resxres gridbox

    """

    if not (0 < res < 180):
        raise ValueError("Resolution must be in range [1, 180).")
    elif (180 % res != 0):
        raise ValueError("Resolution must be a factor of 180 (res=%d)." % res)

    wfdei = read_grid_wfdei()

    # get 2d array in which the value is the landpoint number
    WFDEI_2d = wfdei.expand(np.arange(len(wfdei)))
    WFDEI_2d.fill_value = IMDI

    ###########################################################################
    # Define the new, coarser grid based on the 0.5deg WFDEI grid.
    ###########################################################################
    width = int(res/DLON_WFDEI)
    height = int(res/DLAT_WFDEI)

    vv = list(range(0, (wfdei.nr+1), width))
    hh = list(range(0, (wfdei.nc+1), height))
    lp_min = 0.3*width*height

    lons = [0.5*(wfdei.lons[h1] + wfdei.lons[h2-1])
            for h1, h2 in zip(hh[:-1], hh[1:])]

    lats = [0.5*(wfdei.lats[v1] + wfdei.lats[v2-1])
            for v1, v2 in zip(vv[:-1], vv[1:])]

    lons = np.array(lons)
    lats = np.array(lats)

    ###########################################################################
    # List of tile location H,V and landpoints in HV.
    ###########################################################################
    ListH = []
    ListV = []
    ListLP = []

    for H in range(360//res):
        for V in range(180//res):
            cols = slice(hh[H], hh[H+1])
            rows = slice(vv[V], vv[V+1])
            TILE = WFDEI_2d[rows, cols]
            if TILE.count() > lp_min:
                ListH.append(H)
                ListV.append(V)
                ListLP.append(TILE.compressed().tolist())

    coarse_grid = LandGrid(None, lons, lats, [ListV, ListH])

    return coarse_grid, ListLP
