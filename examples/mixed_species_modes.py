"""
Example, containing simulation of mixed-species ion crystal (consisting of
4 Ca ions and 1 Sr ion) in a five-wire trap, and calculation of the crystal's normal modes.
### IMPORTANT NOTE ###
Due to some Anaconda environment issues, there will be error, if the
simulation of MIXED species is run twice, or if the ion species is switched
between the simulations. This means, that while working with mixed-species 
crystals, you need to restart kernel after each iteration. 
"""

from __future__ import division
import pylion as pl
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, scipy.constants as ct
from electrode import (System, PolygonPixelElectrode, euler_matrix,
                       PointPixelElectrode, PotentialObjective,
                       PatternRangeConstraint, shaped)
import sion as sn


# Global definition of trap parameters.
# This is the same trap as in fivewiretrap.py example
L = 1e-6 # length scale
Vrf = 223. # RF peak voltage in V
mass1 = 40*ct.atomic_mass # Ca mass
mass2 = 88*ct.atomic_mass # Sr mass
Z = 1*ct.elementary_charge # ion charge
Omega = 2*np.pi*35e6 # RF frequency in rad/s
Urf1 = Vrf * np.sqrt(Z / mass1) / (2 * L * Omega)
Urf2 = Vrf * np.sqrt(Z / mass2) / (2 * L * Omega)


# parameters of trap
DCtop = [[760, 600],[1000-240, 400],[760, 600] ]  # Array of lengths and widths of Top electrodes
DCbottom = [[760, 600],[1000-240, 400],[760, 600] ]  # Array of lengths and widths of Bottom electrodes
cwidth = 140  # Width of central dc electrode
clength = 6000 # length of central dc electrode
boardwidth = 0  # width of gaps between electrodes
rftop = 200  # width of top rf electrode, not including width of central electrode
rflength = 6000  # length of rf electrodes
rfbottom = 200  # width of bottom rf electrode
patternTop = 1  # number of groups of Top dc electrodes, defined in DCtop. if None then = 1
patternBot = 1  # same for bottom dcs
getCoordinate = None  # If not None, writes a file with coordinates of vertexes of electrodes
gapped = 0  # gaps between central DC electrode and RF lines
cheight = 1000  # height of the grounded cover electrode plane
cmax = 0  # order of the expansion of cover potential. if 0 - coder not considered, if 5 - considered with optimal precision


elec, Numb, sist = sn.FiveWireTrap(Urf1, DCtop ,DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, patternTop, patternBot, getCoordinate, gapped, cheight, cmax)

x0 = np.array(sist.minimum((0., 2, 3), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))
u_set = 1/5*np.array([0, 15.,   -15. ,   15. ,    8.09, -15.,     8.09,   0.59])

# first find Ca secular frequencies, then Sr freqs.
with sist.with_voltages(dcs=u_set, rfs=None):
    # Check if the minimum was shifted
    x = sist.minimum(x0, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
    # Get trap frequencies
    try:
        curv_z, mod_dir = sist.modes(x, sorted=False)
        omega_sec = np.sqrt(Z * curv_z / mass1) / (L * 2 * np.pi) * 1e-6
        print("Ca secular frequencies: (%.4g, %.4g, %.4g) MHz" % (omega_sec[0], omega_sec[1], omega_sec[2]))
        print("In directions ", mod_dir)
    except:
        print("Secular frequencies not found")

omegas = [omega_sec*1e6 for i in range(4)]
        
elec, Numb, sist = sn.FiveWireTrap(Urf2, DCtop ,DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, patternTop, patternBot, getCoordinate, gapped, cheight, cmax)

# routine to find secular frequencies in minimum point
with sist.with_voltages(dcs=u_set, rfs=None):
    # Check if the minimum was shifted
    x = sist.minimum(x0, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
    # Get trap frequencies
    try:
        curv_z, mod_dir = sist.modes(x, sorted=False)
        omega_sec = np.sqrt(Z * curv_z / mass2) / (L * 2 * np.pi) * 1e-6
        print("Sr secular frequencies: (%.4g, %.4g, %.4g) MHz" % (omega_sec[0], omega_sec[1], omega_sec[2]))
        print("In directions ", mod_dir)
    except:
        print("Secular frequencies not found")

omegas.append(omega_sec*1e6)

ion_number = 5
x0 = x*1e-6

#insert your path to this file here
name = Path(__file__).stem

s = pl.Simulation(name)

#ions' declaration
Caions = {'mass': 40, 'charge': 1}
Srions = {'mass': 88, 'charge': 1}

#placing 4 Ca ions and 1 Sr ion
s.append(pl.placeions(Caions, [[x0[0] - 8e-6 , x0[1], x0[2]],[x0[0] - 4e-6 , x0[1], x0[2]], [x0[0], x0[1], x0[2]], [x0[0] + 4e-6 , x0[1], x0[2]]]))
s.append(sn.five_wire_trap(Omega, Vrf, u_set, elec, Numb, [cmax, cheight]))

s.append(pl.placeions(Srions, [[x0[0]+8e-6, x0[1], x0[2]]]))
#s.append(sn.five_wire_trap(Omega, Vrf, u_set, elec, Numb, [cmax, cheight]))

#temperature initialization
s.append(pl.thermalvelocities(5, False))

#cooling simulation
s.append(pl.langevinbath(0, 1e-7))

#files with simulation information
s.append(pl.dump('positions_mixed.txt', variables=['x', 'y', 'z'], steps=10))
s.append(pl.evolve(1e4))
s.execute()

_, data = pl.readdump('positions_mixed.txt')
data *= 1e6

final_x = data[-1, :, 0]
final_y = data[-1, :, 1]
final_z = data[-1, :, 2]

ion_positions = np.zeros([ion_number, 3])
sort = np.argsort(final_x)

k=0
for i in sort:
    ion_positions[k] = np.array([final_x[i], final_y[i], final_z[i]])
    k+=1

#The last ion is Sr
print('Final positions of ions:\n', ion_positions)

data = data[0:1000, :, :]

plt.figure()
for n in range(ion_number):
    plt.plot(np.arange(data.shape[0]) * 10 + 1, data[:, n, 0])
plt.title('Evolution of ion\'s x coordinates')
plt.xlabel('Time step')
plt.ylabel('Ion\'s x coordinates')
plt.show()

# Plot of the final ion crystal configuration
plt.figure()
plt.plot(data[-1, 0, 0], data[-1, 0, 1], 'bo', label = 'Ca ion')
for i in range(1, ion_number - 1):
    plt.plot(data[-1, i, 0], data[-1, i, 1], 'bo')
plt.plot(data[-1, -1, 0], data[-1, 1, 1], 'go', label = 'Sr ion')
plt.title('Ion\'s equilibrium positions')
plt.xlabel('Ion\'s x coordinates')
plt.ylabel('Ion\'s y coordinates')
plt.ylim([0, max(1, 2 * np.max(np.abs(data[-1, :, 1])))])
plt.legend()
plt.show()

ion_masses = [mass1, mass1, mass1, mass1, mass2]

freqs, modes = sn.normal_modes(ion_positions*L, omegas, ion_masses)

#Normal modes are presented in the order of the increase of their frequency
# which doesn't always coincide with the principle axes of oscillation
axial_modes = []
axial_freqs = [] 
for i in range(3*ion_number):
    k = np.argmax(np.abs(modes[i]))
    if k<ion_number:
        axial_modes.append(modes[i][0:ion_number])
        axial_freqs.append(freqs[i])
    
axial_modes = np.array(axial_modes)

radial_modes_y = []
radial_freqs_y = [] 
for i in range(3*ion_number):
    k = np.argmax(np.abs(modes[i]))
    if k<2*ion_number and k>=ion_number:
        radial_modes_y.append(modes[i][ion_number:2*ion_number])
        radial_freqs_y.append(freqs[i])
    
radial_modes_y = np.array(radial_modes_y)

print('Axial freqs:', axial_freqs)
print('Radial freqs y:', radial_freqs_y)

print('Axial modes:', axial_modes)
print('Radial modes y:', radial_modes_y)



plt.figure(figsize=(4, 6))
plt.plot([], [], color='red', label='radial', linewidth=0.5)
plt.plot([], [], color='blue', label='axial', linewidth=0.5)

for omega in radial_freqs_y:
    plt.plot([-1, 0], [omega / omega_sec[1], omega / omega_sec[1]], color='red', linewidth=0.5)
for omega in axial_freqs:
    plt.plot([-1, 0], [omega / omega_sec[1], omega / omega_sec[1]], color='blue', linewidth=0.5)

plt.ylabel('$\omega/\omega_{\mathrm{com}}^{\mathrm{rad}}$')
plt.xticks([])
plt.xlim(-1, 2)
plt.ylim(ymin=0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(radial_modes_y[::-1, :] / np.max(np.abs(radial_modes_y)), cmap='bwr', vmin = -1, vmax = 1)
plt.colorbar()
plt.xlabel('ion index')
plt.ylabel('Y mode index')
plt.tight_layout()
plt.show()
