"""
Example, containing definition and simulation of the five-wire trap with
three electrodes, containing 5 Ca ions. 
Additionally, normal modes of the linear ion chain is calculated with
axial_ and radial_normal_modes(), and verified with theoretical analysis.
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
L = 1e-6 # length scale
Vrf = 223. # RF peak voltage in V
mass = 40*ct.atomic_mass # ion mass
Z = 1*ct.elementary_charge # ion charge
Omega = 2*np.pi*30e6 # RF frequency in rad/s
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

sist, RF_electrodes, DC_electrodes = sn.five_wire_trap_design(Urf, DCtop ,DCbottom, cwidth, clength, boardwidth, rftop, rflength, rfbottom, need_coordinates = True)

x0 = L*np.array(sist.minimum((0., 2, 3), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))
u_set = np.array([0, 7.804913028626215, -5.623874144063542, 7.804913028626215, 7.804913030569389, -7.804913072245546, 7.80491303056939, -0.8386007732343675])
dc_set = u_set[1:]

# routine to find secular frequencies in minimum point
with sist.with_voltages(dcs=u_set, rfs=None):
    # Check if the minimum was shifted
    x = sist.minimum((0., 90, 120), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG")
    curv_z, mod_dir = sist.modes(x, sorted=False)
    omega_sec = np.sqrt(Z * curv_z / mass) / (L * 2 * np.pi) * 1e-6

omega_sec *=1e6

ion_number = 5
x0 = x*1e-6

"""Simulation"""

#insert your path to this file here
name = Path(__file__).stem

s = pl.Simulation(name)

#ions' declaration
ions = {'mass': 40, 'charge': 1}

#placing ion in random cloud near minimum
positions = sn.ioncloud_min(x0, ion_number, 10e-6)
s.append(pl.placeions(ions, positions))

#declaration of a five wire trap
s.append(sn.polygon_trap([Omega, Omega], [Vrf, Vrf], dc_set, RF_electrodes, DC_electrodes))

#cooling simulation
s.append(pl.langevinbath(0, 1e-7))

#files with simulation information
s.append(pl.dump('positions.txt', variables=['x', 'y', 'z'], steps=10))
s.append(pl.evolve(1e5))
try:
    s.execute()
    pass
except:
    pass

_, data = pl.readdump('positions.txt')
data *= 1e6

final_x = data[-1, :, 0]
final_y = data[-1, :, 1]
final_z = data[-1, :, 2]

ion_positions = np.zeros([ion_number, 3])
sort = np.argsort(final_x)
    
equilibrium_positions = []
for i in range(ion_number):
    equilibrium_positions.append(L*np.array([np.mean(data[9000:, i, 0]), np.mean(data[9000:, i, 1]), np.mean(data[9000:, i, 2])]))
equilibrium_positions = np.array(equilibrium_positions)

k=0
for i in sort:
    ion_positions[k] = np.array([equilibrium_positions[i][0], equilibrium_positions[i][1], equilibrium_positions[i][2]])
    k+=1

np.set_printoptions(2)
print('Final positions of ions in um:\n', ion_positions)

"""Normal modes"""

freqs, modes = sn.normal_modes(ion_positions, omega_sec, mass, linear = True)
axial_freqs = freqs[0]
axial_modes = modes[0]
np.set_printoptions(0)

print('Axial mode frequencies:', axial_freqs, 'Hz')

np.set_printoptions(4)
print('Axial modes:', axial_modes)

theor_modes = -1*np.array([[0.4472,0.4472,0.4472,0.4472,0.4472],[0.6395,0.3017, 0, -0.3017, -0.6395],
               [0.5377, -0.2805, -0.5143,-0.2805, 0.5377], [-0.3017,0.6395,0,-0.6395,0.3017],
               [0.1045,-0.4704,0.7318,-0.4704, 0.1045]])
theor_freqs = np.array([1,3,5.818,9.332,13.47])
freqs = np.sqrt(theor_freqs)*omega_sec[0]
print('Verification process.')

l2 = (ct.e**2 / (4 * np.pi * ct.epsilon_0 * 40 * ct.atomic_mass * (omega_sec[0] * 2 * np.pi) ** 2)) ** (1/3)
theor_positions = np.array([-1.7429, -0.8221, 0, 0.8221, 1.7429]) * l2
pos_diff = np.linalg.norm(theor_positions - final_x[sort]*1e-6)

print('Difference from theoretical positions:', np.round(pos_diff * 1e6, 4), 'um')

print('Difference from theoretical frequencies:', np.round(np.abs(freqs-axial_freqs),1), 'Hz')

diff = np.zeros(5)
for i in range(5):
    dif = axial_modes - theor_modes
    diff[i] = np.linalg.norm(dif[i])
    if diff[i] > 1:
        dif = axial_modes + theor_modes
        diff[i] = np.linalg.norm(dif[i])
print('Difference in mode vectors', diff)

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
