"""
Example, containing simulation of mixed-species ion crystal (consisting of
2 Be ions and 1 Ca ion) in a five-wire trap, and calculation of the 
crystal's normal modes. The configuration is similar to the one in:
    https://doi.org/10.3929/ethz-b-000420229
### IMPORTANT NOTE ###
Due to some Anaconda environment issues, there will be error, if the
simulation of MIXED species is run twice, or if the ion species is switched
between the simulations. This means, that while working with mixed-species 
crystals, you need to restart kernel after each simulation. 
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
Vrf = 245*2 # RF peak voltage in V
mass1 = 40*ct.atomic_mass # Yb mass
mass2 = 9*ct.atomic_mass # Ba mass
Z = 1*ct.elementary_charge # ion charge
Omega = 2*np.pi*100e6 # RF frequency in rad/s
Urf1 = Vrf * np.sqrt(Z / mass1) / (2 * L * Omega)
Urf2 = Vrf * np.sqrt(Z / mass2) / (2 * L * Omega)


# parameters of trap
DCtop = [[760, 600],[1000-240, 400],[760, 600] ]  # Array of lengths and widths of Top electrodes
DCbottom = [[760, 600],[1000-240, 400],[760, 600] ]  # Array of lengths and widths of Bottom electrodes
cwidth = 150  # Width of central dc electrode
clength = 6000 # length of central dc electrode
boardwidth = 0  # width of gaps between electrodes
rftop = 200  # width of top rf electrode, not including width of central electrode
rflength = 6000  # length of rf electrodes
rfbottom = 200  # width of bottom rf electrode

sist, RF_electrodes, DC_electrodes = sn.five_wire_trap_design(Urf1, DCtop ,DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, need_coordinates=True, need_plot = True)

x0 = np.array(sist.minimum((0., 2, 3), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))
u_set = 1/1.155*np.array([0, 15.,   -15. ,   15. ,    15, -15.,     15,   1.5 ])
dc_set = u_set[1:]

# first find Yb secular frequencies, then Sr freqs.
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

omegaCa = omega_sec*1e6
        
sist = sn.five_wire_trap_design(Urf2, DCtop ,DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom)

# routine to find Ba secular frequencies in minimum point
with sist.with_voltages(dcs=u_set, rfs=None):
    # Check if the minimum was shifted
    x = sist.minimum(x0, axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
    # Get trap frequencies
    try:
        curv_z, mod_dir = sist.modes(x, sorted=False)
        omega_sec = np.sqrt(Z * curv_z / mass2) / (L * 2 * np.pi) * 1e-6
        print("Be secular frequencies: (%.4g, %.4g, %.4g) MHz" % (omega_sec[0], omega_sec[1], omega_sec[2]))
        print("In directions ", mod_dir)
    except:
        print("Secular frequencies not found")

omegas = []
omegas.append(omega_sec*1e6)
omegas.append(omegaCa)
omegas.append(omega_sec*1e6)


ion_number = 3
x0 = x*1e-6

#insert your path to this file here
name = Path(__file__).stem

s = pl.Simulation(name)

#ions' declaration
Caions = {'mass': 40, 'charge': 1}
Beions = {'mass': 9, 'charge': 1}

#placing 4 Yb ions and 1 Ba ion
s.append(pl.placeions(Caions, [[x0[0], x0[1], x0[2]]]))
s.append(sn.polygon_trap([Omega, Omega], [Vrf, Vrf], dc_set, RF_electrodes, DC_electrodes))

s.append(pl.placeions(Beions, [[x0[0]-4e-6, x0[1], x0[2]], [x0[0]+4e-6, x0[1], x0[2]]]))
#s.append(sn.five_wire_trap(Omega, Vrf, u_set, elec, Numb, [cmax, cheight]))

#cooling simulation
s.append(pl.langevinbath(0, 1e-6))

#files with simulation information
s.append(pl.dump('positions_mixed.txt', variables=['x', 'y', 'z'], steps=10))
s.append(pl.evolve(1e5))
try:
    s.execute()
except:
    pass

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

#The middle ion is Ca
print('Final positions of ions:\n', ion_positions)

#data = data[0:1000, :, :]

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

plt.plot(data[-1, 1, 0], data[-1, 1, 1], 'go', label = 'Be ion')
plt.plot(data[-1, 2, 0], data[-1, 2, 1], 'go')
plt.title('Ion\'s equilibrium positions')
plt.xlabel('Ion\'s x coordinates')
plt.ylabel('Ion\'s y coordinates')
plt.ylim([0, max(1, 2 * np.max(np.abs(data[-1, :, 1])))])
plt.legend()
plt.show()

ion_masses = np.array([mass2, mass1, mass2])

freqs, modes = sn.normal_modes(ion_positions*L, omegas, ion_masses, linear = True)

axial_modes = modes[0]
axial_freqs = freqs[0]

radial_modes_y = modes[1]
radial_freqs_y = freqs[1] 

print('Axial freqs:', axial_freqs)
print('Radial freqs y:', radial_freqs_y)

print('Axial modes:', axial_modes)
print('Radial modes y:', radial_modes_y)

""" Verification """

teor_ax_freqs = np.array([1.531, 4.101, 4.188])*1e6

print("\nVerification\nDifferences in axial normal frequencies:\n", teor_ax_freqs - np.array(axial_freqs), "Hz")

plt.figure(figsize=(4, 6))
plt.plot([], [], color='red', label='radial', linewidth=0.5)
plt.plot([], [], color='blue', label='axial', linewidth=0.5)

for omega in radial_freqs_y:
    plt.plot([-1, 0], [omega / omega_sec[1], omega / omega_sec[1]], color='red', linewidth=0.5)
for omega in axial_freqs:
    plt.plot([-1, 0], [omega / omega_sec[1], omega / omega_sec[1]], color='blue', linewidth=0.5)

#Normal frequency spectrum
plt.ylabel('$\omega/\omega_{\mathrm{com}}^{\mathrm{rad}}$')
plt.xticks([])
plt.xlim(-1, 2)
plt.ylim(ymin=0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Axial interaction matrix
plt.figure()
plt.imshow(axial_modes[::-1, :] / np.max(np.abs(axial_modes)), cmap='bwr', vmin = -1, vmax = 1)
plt.colorbar()
plt.xlabel('ion index')
plt.ylabel('Y mode index')
plt.tight_layout()
plt.show()

#Y interaction matrix
plt.figure()
plt.imshow(radial_modes_y[::-1, :] / np.max(np.abs(radial_modes_y)), cmap='bwr', vmin = -1, vmax = 1)
plt.colorbar()
plt.xlabel('ion index')
plt.ylabel('Y mode index')
plt.tight_layout()
plt.show()


