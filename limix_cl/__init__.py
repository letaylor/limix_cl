"""
LIMIX command line tool
=======================

Command tool for fitting linear models using csv input dataframes.

Available subpackages
---------------------
utils
    helper methods for using limix_lmm
limix_lmm
    methods that use limix
"""
__version__ = '0.1.1.dev0'
__all__ = ["limix_lmm", "utils"]

from . import utils
from . import limix_lmm

