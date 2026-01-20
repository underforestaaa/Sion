# -*- coding: utf8 -*-
#
#   electrode: numeric tools for Paul traps
#
#   Copyright (C) 2011-2012 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

import numpy as np

# qualitative color map
# http://colorbrewer2.org/index.php?type=qualitative&scheme=Set3&n=12
set4 = np.array([
    [141, 211, 199],
    [255, 255, 179],
    [190, 186, 218],
    [251, 128, 114],
    [128, 177, 211],
    [253, 180,  98],
    [179, 222, 105],
    [252, 205, 229],
    [217, 217, 217],
    [188, 128, 189],
    [204, 235, 197],
    [255, 237, 111], 
    ])
# set3 = np.array([
#     [255,255,229],
#     [255,247,188],
#     [254,227,145],
#     [254,196,79],
#     [254,153,41],
#     [236,112,20], 
#     [204,76,2],
#     [153,52,4],
#     [102,37,6],
#     ])
# set4 = np.array([
#     [247,252,253],
#     [224,236,244],
#     [191,211,230],
#     [158,188,218],
#     [140,150,198],
#     [140,107,177],
#     [136,65,157],
#     [129,15,124],
#     [77,0,75],
#     ])

