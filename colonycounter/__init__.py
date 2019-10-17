# -*- coding: utf-8 -*-
"""
This library was made for the quantification in colony formation assay.
=================================

"""
import sys
import re
import warnings
import logging

from .main_class import Counter
#from . import motif_analysis


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))


__version__ = '0.2.0'

__all__ = ["Counter"]
