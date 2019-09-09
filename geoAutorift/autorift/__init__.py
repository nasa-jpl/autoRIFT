#!/usr/bin/env python

#Should always work - standlone or with ISCE
from .Autorift import Autorift

#Should work if ISCE has been installed with/without Autorift
try:
    from .AutoriftISCE import AutoriftISCE
except ImportError:
    pass #This means ISCE support not available. Don't raise error. Allow stand alone use

