"""
This is a wrapper around electrode package to ensure it's compatibility with 
up-to-date versions of numpy and sion: http://github.com/nist-ionstorage/electrode
Pylion is used in accordance to GPLv3+ license.
"""

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

from .system import System
from .electrode import (PolygonPixelElectrode, PointPixelElectrode,
        CoverElectrode, MeshPixelElectrode, GridElectrode)
from .pattern_constraints import (PotentialObjective, PatternRangeConstraint,
        MultiPotentialObjective)
from .transformations import euler_from_matrix, euler_matrix
from .utils import shaped

#from .constraints import (VoltageConstraint, SymmetryConstraint,
#        PotentialConstraint, ForceConstraint, CurvatureConstraint,
#        OffsetPotentialConstraint)

__version__ = "1.5+dev"
author = "Robert Jordens",
author_email = "jordens@gmail.com",
