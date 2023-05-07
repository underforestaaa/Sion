from __future__ import division
import pylion as pl
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, scipy.constants as ct
from electrode import (System, PolygonPixelElectrode, euler_matrix,
                       PointPixelElectrode, PotentialObjective,
                       PatternRangeConstraint, shaped)
import sion as sn

# Global definition of trap parameters.
L = 1e-6 # length scale
Vrf = 223. # RF peak voltage in V
mass = 40*ct.atomic_mass # ion mass
Z = 1*ct.elementary_charge # ion charge
Omega = 2*np.pi*25.8e6 # RF frequency in rad/s
Urf = Vrf * np.sqrt(Z / mass) / (2 * L * Omega)
scale = Z / ((L * Omega) ** 2 * mass)

# parameters of trap
DCtop = [[760, 600],[1000-240, 400],[760, 600] ]  # Array of lengths and widths of Top electrodes
DCbottom = [[760, 600],[1000-240, 400],[760, 600] ]  # Array of lengths and widths of Bottom electrodes
cwidth = 140  # Width of central dc electrode
clength = 6000 # length of central dc electrode
boardwidth = 0  # width of gaps between electrodes
rftop = 100  # width of top rf electrode, not including width of central electrode
rflength = 6000  # length of rf electrodes
rfbottom = 240  # width of bottom rf electrode
patternTop = 1  # number of groups of Top dc electrodes, defined in DCtop. if None then = 1
patternBot = 1  # same for bottom dcs
getCoordinate = None  # If not None, writes a file with coordinates of vertexes of electrodes
gapped = 0  # gaps between central DC electrode and RF lines
cheight = 1000  # height of the grounded cover electrode plane
cmax = 0  # order of the expansion of cover potential. if 0 - coder not considered, if 5 - considered with optimal precision

elec, Numb, s = sn.FiveWireTrap(Urf, DCtop, DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, patternTop,
                             patternBot, getCoordinate, gapped, cheight, cmax)

#declaration of RF and DC electrodes
RF_electrodes=[]
for iterr in range(2):
    RF_electrodes.append(elec[0][1][iterr])
DC_electrodes = []
for ite in range(Numb):
    DC_electrodes.append(elec[ite + 1][1][0])
print(RF_electrodes)
print(DC_electrodes)


x0 = s.minimum((0., 100, 120), axis=(0, 1, 2))

# Constants declaration
ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity

# obtain secular freqs of a trap
# pre-obtained set of voltages on DC electrodes
u_set = np.array([0, 7.804913028626215, -5.623874144063542, 7.804913028626215, 7.804913030569389, -7.804913072245546, 7.80491303056939, -0.8386007732343675])

# routine to find secular frequencies in minimum point
with s.with_voltages(dcs=u_set, rfs=None):
    # Check if the minimum was shifted
    x = s.minimum((0., 90, 120), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
    x_shift = x - x0
    print('The minimum was shifted by: (%.3g, %.3g, %.3g)' % (x_shift[0], x_shift[1], x_shift[2]))

    # Get trap frequencies
    try:
        curv_z, mod_dir = s.modes(x, sorted=False)
        omega_sec = np.sqrt(Z * curv_z / mass) / (L * 2 * np.pi) * 1e-6
        print("secular frequencies: (%.4g, %.4g, %.4g) MHz" % (omega_sec[0], omega_sec[1], omega_sec[2]))
        print("in directions ", mod_dir)
    except:
        print("secular frequencies not found")
    for line in s.analyze_static(x, axis=(0, 1, 2,), m=mass, q=Z, l=L, o=Omega):
        print(line)


omega_sec *=1e6

# simulation of ion crystal
ion_number = 5
x0 = x*1e-6

#insert your path to this file here
name = Path("/Users/a.podlesnyy/Desktop/RQC/Surface Traps/polygon_simulation.py").stem

sim = pl.Simulation(name)

#ion declaration
Caions = {'mass': 40, 'charge': 1}

#placing ions in order
positions = sn.ions_in_order(x0, ion_number, 5e-6)
sim.append(pl.placeions(Caions, positions))

#declarations of voltages for polygon trap
uset = [Vrf]
uset.extend(u_set)
uset[1] = Vrf
uset = np.array(uset)

print(uset)


#polygon trap initialization
sim.append(sn.polygon_trap([Omega,Omega], uset, RF_electrodes, DC_electrodes))

#temperature and cooling
sim.append(pl.thermalvelocities(5, False))
sim.append(pl.langevinbath(0.00001, 1e-7))

#file with simulation information
sim.append(pl.dump('pos1.txt', variables=['x', 'y', 'z'], steps=10))
sim.append(pl.evolve(2e4))
sim.execute()

_, data = pl.readdump('pos1.txt')

final_x = data[-1, :, 0]
final_y = data[-1, :, 1]
final_z = data[-1, :, 2]
sort = np.argsort(final_x)

ion_positions = np.zeros([ion_number, 3])

k=0
for i in sort:
    ion_positions[k] = np.array([final_x[i], final_y[i], final_z[i]])
    k+=1

print('Final ion positions:\n',ion_positions)

# Plot of ion crystal evolution
plt.figure()
for n in range(ion_number):
    plt.plot(np.arange(data.shape[0]) * 10 + 1, data[:, n, 0])
plt.title('Evolution of ion\'s x coordinates')
plt.xlabel('Time step')
plt.ylabel('Ion\'s x coordinates')
plt.show()

# Plot of the final ion crystal configuration
plt.figure()
plt.scatter(data[-1, :, 0], data[-1, :, 1])
plt.title('Ion\'s equilibrium positions')
plt.xlabel('Ion\'s x coordinates')
plt.ylabel('Ion\'s y coordinates')
plt.ylim([-max(1, 1.2 * np.max(np.abs(data[-1, :, 1]))), max(1, 1.2 * np.max(np.abs(data[-1, :, 1])))])
plt.show()

'''Phonon modes of the ion crystal'''





# Declarations of ion masses and modes for normal modes
ion_masses = [mass for el in range(5)]
omegas = [omega_sec for el in range(5)]
M_matrix = np.diag(list(ion_masses)*3)

#obtaining normal modes, for general case
freqs, modes = sn.normal_modes(ion_positions, omegas, ion_masses)
print(freqs)
axial_freqs = freqs[0:5]
radial_freqs = freqs[5:10]
axial_modes = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        axial_modes[i, j] = modes[i][j]
radial_modes = modes[5:10]

"""
Since this is a linear ion chain of 5 ions, just for verification 
I represent these details this way. However, this also works for 
an arbitrary ion crystal of different masses.
"""


print('Axial freqs:', axial_freqs)
print('Radial freqs y:', radial_freqs)

print('Axial modes', axial_modes)

theor_modes = np.array([[0.4472,0.4472,0.4472,0.4472,0.4472],[-0.6395,-0.3017, 0, 0.3017, 0.6395],
               [-0.5377, 0.2805, 0.5143,0.2805, -0.5377], [-0.3017,0.6395,0,-0.6395,0.3017],
               [0.1045,-0.4704,0.7318,-0.4704, 0.1045]])
theor_freqs = np.array([1,3,5.818,9.332,13.47])
freqs = np.sqrt(theor_freqs)*omega_sec[0]
print('Theor freqs', freqs*1e-6)
print('Difference in freqs', np.abs(freqs-axial_freqs)*1e-6)
dif = axial_modes - theor_modes
diff = np.zeros(5)
for i in range(5):
    diff[i] = np.linalg.norm(dif[i])
print('Difference in vectors', diff)
print('If the difference is 2, then the mode just flipped a sign.')

plt.figure(figsize=(4, 6))
plt.plot([], [], color='red', label='radial', linewidth=0.5)
plt.plot([], [], color='blue', label='axial', linewidth=0.5)

for omega in radial_freqs:
    plt.plot([-1, 0], [omega / omega_sec[1], omega / omega_sec[1]], color='red', linewidth=0.5)
for omega in axial_freqs:
    plt.plot([-1, 0], [omega / omega_sec[1], omega / omega_sec[1]], color='blue', linewidth=0.5)

plt.ylabel('$\omega/\omega_{\mathrm{com}}^{\mathrm{rad}}$')
plt.xticks([])
plt.xlim(-1, 2)
plt.ylim(ymin=0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('radial mode frequencies', dpi=300)
plt.show()

plt.figure()
plt.imshow(axial_modes[::-1, :] / np.max(np.abs(axial_modes)), cmap='bwr')
plt.colorbar()
plt.xlabel('ion number')
plt.ylabel('mode number')
plt.tight_layout()
plt.savefig('radial mode matrix', dpi=300)
plt.show()