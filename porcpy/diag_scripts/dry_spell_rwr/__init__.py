"""A package for analysing the land surface response to dry spells.

This package is a set of helper routines for running the dry spell land
surface temperature analysis described in,

Folwell, S.S., P.P. Harris, and C.M. Taylor (2015), Large-scale surface
responses during European dry spells diagnosed from land surface
temperature, J. Hydrometeorol., doi:10.1175/JHM-D-15-0064.1

Gallego-Elvira, B., C.M. Taylor, P.P. Harris, D. Ghent, K.L. Veal, and
S.S. Folwell (2016), Global observational diagnosis of soil moisture
control on the land surface energy balance, Geophys. Res. Lett., 43,
2623-2631, doi:10.1002/2016GL068178

"""

import logging
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
import warnings

warnings.filterwarnings("ignore",
                        module="matplotlib",
                        category=MatplotlibDeprecationWarning,
                        message=".*mpl_toolkits.axes_grid1.*")

HEADER = r"""
      ___           ___           ___
     /  /\         /__/\         /  /\
    /  /::\       _\_ \:\       /  /::\
   /  /:/\:\     /__/\ \:\     /  /:/\:\
  /  /:/~/:/    _\_ \:\ \:\   /  /:/~/:/
 /__/:/ /:/___ /__/\ \:\ \:\ /__/:/ /:/___
 \  \:\/:::::/ \  \:\ \:\/:/ \  \:\/:::::/
  \  \::/~~~~   \  \:\ \::/   \  \::/~~~~
   \  \:\        \  \:\/:/     \  \:\
    \  \:\        \  \::/       \  \:\
     \__\/         \__\/         \__\/

""" + __doc__

log_fmt = "%(asctime)s %(levelname)s %(name)s:%(lineno)s\t%(message)s"
logging.basicConfig(format=log_fmt, datefmt='%H:%M:%S', level="INFO")

logger = logging.getLogger(__name__)
logger.info(HEADER)

# Missing data indicators used by this package.
FMDI = -9999.0
IMDI = -9999

# Default values for optional arguments used in the package.  If alternative
# values are required, it is recommended that the user sets them by passing
# optional arguments to module functions rather than by changing these global
# defaults.
_PR_MAX = 0.5  # Default daily precipitation threshold (mm).
_TAS_MIN = 283.15  # Default daily air temperature threshold (K).
_DURATION_MIN = 10  # Default minimum dry spell length (days).
_DURATION_ANTE = 30  # Default dry spell antecedent period (days).

_WIN_LEN = 60  # Default window length for climatology smoother.

_RWR_BEG = 1   # Default first dry spell day for RWR regression (0-based).
_RWR_END = 10  # Default last dry spell day for RWR regression (0-based).
_NDAYS_ANTE = 5  # Default number of antecedent days in composites.
_NDAYS_DRY = 10  # Default number dry spell days in composites.
_PTILES = 1  # Default number of precipitation percentiles.
