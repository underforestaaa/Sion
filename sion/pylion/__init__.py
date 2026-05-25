"""
This is a wrapper around Pylion package to ensure it's compatibility with 
up-to-date versions of numpy and SION: https://doi.org/10.1016/j.cpc.2020.107187
Pylion is used in accordance to MIT license.
"""

from .pylion import Simulation, __version__
from .functions import *

__author__ = """Dimitris Trypogeorgos"""
__email__ = 'dtrypogiorgos@gmail.com'
