"""
A module that defines (1) a Region class that describes simple lon/lat
regions and (2) some example regions similar to the commonly used SREX
regions (http://www.ipcc-data.org/guidelines/pages/ar5_regions.html).
"""

from __future__ import print_function, division


class Region(object):
    def __init__(self, short_name, lat_bnds, lon_bnds,
                 long_name=None):

        lat_bnds = [float(l) for l in lat_bnds]
        lon_bnds = [float(l) for l in lon_bnds]

        if len(lat_bnds) != 2:
            raise ValueError(
                "Expected 2 latitude bounds, got {}.".format(len(lat_bnds)))
        elif [l for l in lat_bnds if abs(l) > 180.]:
            raise ValueError("Latitudes outside of permitted bounds.")

        if len(lon_bnds) != 2:
            raise ValueError(
                "Expected 2 longitude bounds, got {}.".format(len(lon_bnds)))
        elif [l for l in lon_bnds if l < -180. or l > 360.]:
            raise ValueError("Longitudes outside of permitted bounds.")

        self.short_name = short_name
        self.long_name = long_name
        self.lat_bnds = lat_bnds
        self.lon_bnds = lon_bnds

    def __repr__(self):
        return self.short_name

    def in_bounds(self, lon, lat):
        """Return True if point (lon, lat) lies within this region."""
        if lon > 180.0:
            lon = lon - 360.0
        return (self.lon_bnds[0] <= lon < self.lon_bnds[1] and
                self.lat_bnds[0] <= lat < self.lat_bnds[1])


def get_srex_regions(lon, lat):
    return [r.short_name for r in regions if r.in_bounds(lon, lat)]


regions = [Region("wna", [27.5, 60.0], [-140, -105],
                  long_name="West North America"),
           Region("cna", [27.5, 50.0], [-105, -85],
                  long_name="Central North America"),
           Region("ena", [25.0, 50.0], [-85, -60],
                  long_name="East North America"),

           Region("cam", [5, 27.5], [-120, -70],
                  long_name="Central America/Mexico"),
           Region("amz", [-20, 12.5], [-70, -50],
                  long_name="Amazon"),
           Region("neb", [-20, 0], [-50, -30],
                  long_name="North East Brazil"),
           Region("ssa", [-55.0, -20], [-67.5, -40],
                  long_name="Southeastern South America"),

           Region("waf", [-12.5, 15], [-20, 25],
                  long_name="West Africa"),
           Region("eaf", [-12.5, 15], [25, 60],
                  long_name="East Africa"),
           Region("saf", [-35, -12.5], [0, 60],
                  long_name="Southern Africa"),
           Region("sam", [15, 30], [-20, 40],
                  long_name="Sahara"),

           Region("med", [30, 45], [-10, 40],
                  long_name="Mediterranean"),
           Region("ceu", [45, 60], [-10, 40],
                  long_name="Central/North Europe"),

           Region("nas", [50, 60], [40, 180],
                  long_name="North Asia"),
           Region("was", [15, 50], [40, 60],
                  long_name="West Asia"),
           Region("sas", [5, 30], [60, 100],
                  long_name="South Asia"),
           Region("cas", [30, 50], [60, 75],
                  long_name="Central Asia"),
           Region("tib", [30, 50], [75, 100],
                  long_name="Tibetan Plateau"),
           Region("eas", [20, 50], [100, 140],
                  long_name="East Asia"),

           Region("nau", [-30, -10], [110, 160],
                  long_name="North Australia"),
           Region("sau", [-50, -30], [110, 180],
                  long_name="South Australia"),

           Region("us-all", [15, 45], [-130, -60],
                  long_name="USA"),
           Region("us-west", [15, 45], [-130, -90],
                  long_name="USA West"),
           Region("af-south", [-35, -15], [10, 40],
                  long_name="Africa South"),
           Region("europe", [35, 60], [-20, 40],
                  long_name="Europe")]
