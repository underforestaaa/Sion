# -*- coding: utf-8 -*-
"""
Example of calculating anharmonic modes in highly anharmonic trap. The results 
are also verified with the article DOI 10.1088/1367-2630/13/7/073026
The numerically calculated modes are more precise, since the condition of 
l2/l4 << 1 does not inforced, and both hexapole and octopole anharmonicities are considered
"""
from __future__ import division
import numpy as np, scipy.constants as ct
import sion as sn
from pathlib import Path
import pylion as pl

# define trap with large anharmonicity 
L = 1e-6 # length scale
Vrf = 100. # RF peak voltage in V
mass = 40*ct.atomic_mass # ion mass
Z = 1*ct.elementary_charge # ion charge
Omega = 2*np.pi*100e6 # RF frequency in rad/s
Urf = Vrf * np.sqrt(Z / mass) / (2 * L * Omega)
scale = Z / ((L * Omega) ** 2 * mass)


# parameters of trap
DCtop = [[760, 20],[1000-240, 10],[760, 20] ]  # Array of lengths and widths of Top electrodes
DCbottom = [[760, 20],[1000-240,10],[760, 20] ]  # Array of lengths and widths of Bottom electrodes
cwidth = 80  # Width of central dc electrode
clength = 60 # length of central dc electrode
boardwidth = 0  # width of gaps between electrodes
rftop = 100  # width of top rf electrode, not including width of central electrode
rflength = 100  # length of rf electrodes
rfbottom = 100  # width of bottom rf electrode
patternTop = 1  # number of groups of Top dc electrodes, defined in DCtop. if None then = 1
patternBot = 1  # same for bottom dcs
getCoordinate = None  # If not None, writes a file with coordinates of vertexes of electrodes
gapped = 0  # gaps between central DC electrode and RF lines
cheight = 1000  # height of the grounded cover electrode plane
cmax = 0  # order of the expansion of cover potential. if 0 - coder not considered, if 5 - considered with optimal precision


elec, Numb, sist = sn.FiveWireTrap(Urf, DCtop ,DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, patternTop, patternBot, getCoordinate, gapped, cheight, cmax)

x0 = np.array(sist.minimum((0., 2, 3), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))
u_set = np.array([0, -15.,   -15. ,  -15.,   -15.,   -15.  , -15.  ,  -0.69])
with sist.with_voltages(dcs = u_set, rfs = None):
    x1 = np.array(sist.minimum( x0, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))
    print('Potential minimum:', x1)
    curv_z, mod_dir=sist.modes(x1,sorted=False)
    omega_sec = np.sqrt(Z*curv_z/mass)/(L*2*np.pi) * 1e-6
    print("Secular frequencies: (%.4g, %.4g, %.4g) MHz" % (omega_sec[0],omega_sec[1],omega_sec[2]))
    print("In directions\na:", mod_dir[0],"\nb:", mod_dir[1],"\nc:", mod_dir[2] )

minimums = [x1]
axis = 0 # this is axial axis in this example 

with sist.with_voltages(dcs = u_set, rfs = None):
    scales = sn.anharmonics(sist, minimums, axis, L)
    
scales = scales[0]
print('l2 =', scales[0], '\nl3 =', scales[1], '\nl4 =', scales[2])

"""
The following lengths suggest a very small hexapole anharmonicity
and high octopole anharmonicity. For verification, results from article:
DOI 10.1088/1367-2630/13/7/073026  are compared. 
The following block containes calculation of anharmonical COM and stretch 
normal modes of an ion pair in this trap in the presence of octopole 
anharmonicity
"""

e = ct.elementary_charge
pi = np.pi
eps = ct.epsilon_0

l2 = scales[0]
l4 = np.abs(scales[2])

omega_c = omega_sec[axis]*(1 + np.sign(scales[2])*3/2**(4/3)*(l2/l4)**2) 
omega_s = np.sqrt(3)*omega_sec[axis]*(1 + np.sign(scales[2])*(5/3)/2**(4/3)*(l2/l4)**2)

print('Theoretical axial anharmonic mode frequencies:','\nCOM:', np.round(omega_c, 2),', Stretch:', np.round(omega_s, 2))


"""
Now we calculate anharmonic frequencies numerically, using final positions,
obtained with simulation. Both octopole and hexapole anharmonicities are accounted for
"""

ion_number = 2

#insert your path to this file here
name = Path(__file__).stem

s = pl.Simulation(name)

#ions' declaration
ions = {'mass': 40, 'charge': 1}

#placing ion in random cloud near minimum
positions = sn.ioncloud_min(x1*L, ion_number, 5e-6)
s.append(pl.placeions(ions, positions))

#declaration of a five wire trap
s.append(sn.five_wire_trap(Omega, Vrf, u_set, elec, Numb, [cmax, cheight]))

#temperature initialization
s.append(pl.thermalvelocities(5, False))

#cooling simulation
s.append(pl.langevinbath(0, 1e-7))

#files with simulation information
s.append(pl.dump('anharmonic_positions.txt', variables=['x', 'y', 'z'], steps=10))
s.append(pl.evolve(1e5))
s.execute()

_, data = pl.readdump('anharmonic_positions.txt')

final_x = data[-1, :, 0]
final_y = data[-1, :, 1]
final_z = data[-1, :, 2]

ion_positions = np.zeros([ion_number, 3])
sort = np.argsort(final_x)

k=0
for i in sort:
    ion_positions[k] = np.array([final_x[i], final_y[i], final_z[i]])
    k+=1

print('Final positions of ions:\n', ion_positions)

#ion_positions = [[0.62996*l2,  4.00e-05 , 3.93e-05],[ -0.62996*l2,  4.00e-05,  3.93e-05]]

masses = [mass, mass]
print(ion_positions[0][0], ion_positions[1][0])
with sist.with_voltages(dcs = u_set, rfs = None):
    freqs, modes = sn.anharmonic_modes(sist, ion_positions, masses, axis)

freqs *= 1e-6

print('Sion anharmonic mode frequencies:\n','COM:', np.round(freqs[0], 2),', Stretch:', np.round(freqs[1], 2))
print('Difference between theoretical and numerical frequencies:', '\nCOM:', omega_c - freqs[0],', Stretch:', omega_s - freqs[1])
print('Anharmonic mode vectors:', modes)
