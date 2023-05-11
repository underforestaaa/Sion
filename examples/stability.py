"""
This file contains demonstration of stability analysis of a sion.n_rf_trap. 
The plot of the trap, as well as the plot of its stability diagram and 
stability parameters for Ca ion are shown.
"""

from __future__ import division
import numpy as np, scipy.constants as ct
from electrode import (System)
import sion as sn

# Global definition of trap parameters. Used for all cells in this notebook
L = 1e-6 # µm length scale
Vrf = 100. # RF peak voltage in V
M = 40*ct.atomic_mass # ion mass
Z = 1*ct.elementary_charge # ion charge
Omega = 2*np.pi*30e6 # RF frequency in rad/s
# RF voltage in parametrized form so that the resulting potential equals the RF pseudopotential in eV
Urf = Vrf*np.sqrt(Z/M)/(2*L*Omega)
scale = Z/((L*Omega)**2*M)


#parameters of trap
DCtop = [[1000,300]] #Array of lengths and widths of Top electrodes
DCbottom = [[1000,300]] #Array of lengths and widths of Bottom electrodes 
cwidth = 70  #Width of central dc electrode
rflength = 4000  #length of central dc, rf electrode
rfwidth = [[100, 70], [100, 70], [100, 70]]
patternTop = 3  #number of groups of Top dc electrodes, defined in DCtop. if None then = 1
patternBot = 3 #same for bottom dcs
cheight = 1000   #height of the grounded cover electrode plane 
cmax = 0  # order of the expansion of cover potential. if 0 - coder not considered, if 5 - considered with optimal precision
n_rf = 3
Urf = [[Urf, Urf], [Urf/2, Urf/2], [Urf/4, Urf/2]]

coordinates, s = sn.n_rf_trap(Urf, DCtop, DCbottom, cwidth, rfwidth, rflength, n_rf = 3, L = 1e-6, patternTop=3, patternBot=3, plott = 1)

dc_set = 1*np.array([ 5.05, -15.,     5.05,  15.,   -15.,    15.,    -0.14])
u_set = np.append(np.zeros(2*n_rf), dc_set)

with s.with_voltages(dcs = u_set, rfs = None):
    x1 = s.minimum((0, 30.3, 79.4), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
    print('The minimum is (%.3g, %.3g, %.3g)' % (x1[0], x1[1], x1[2]))
    # Get trap frequencies
    try:
        curv_z, mod_dir=s.modes(x1,sorted=False)
        omega_sec=np.sqrt(Z*curv_z/M)/(L*2*np.pi) * 1e-6
        print("Secular frequencies: (%.4g, %.4g, %.4g) MHz" % (omega_sec[0],omega_sec[1],omega_sec[2]))
        print("In directions\na:", mod_dir[0],"\nb:", mod_dir[1],"\nc:", mod_dir[2] )
        e1 = np.array([0,0,1])
        alp = np.arccos(np.inner(mod_dir[2],e1))
        print('Angle of secular modes rotation alpha:', alp)
    except:
        print("Secular frequencies not found")
        
# calculates stability parameters and gives a plot of the stability diagram and
# This ion's particular a, q parameters
with s.with_voltages(dcs = u_set, rfs = None):
    params = sn.stability(s, M, Omega, Z, dot = x1, plot = 1) 

print("Calculated stability parameters for this ion:\na =", params[0], "q =", params[1])
print("Trap geometry parameters:\n", "\u03B1 =", params[2][0],"\u03B8 =", params[2][1])
print("Range of achievable a-parameters in this trap:", params[3])
print("Maximal achievable q-parameter in this trap:", params[4])

