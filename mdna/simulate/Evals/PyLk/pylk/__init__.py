"""
pyLink
=====

Provides methods to calculate linking number and writhe of polymer configurations

"""

from .writhemap import writhemap
from .writhe import writhe

from .rbp_link import triads2chain, triads2link
from .eval_link import linkingnumber
