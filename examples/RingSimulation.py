from __future__ import division
import pylion as pl
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt, numpy as np, scipy.constants as ct
from mpl_toolkits.mplot3d import Axes3D
from electrode import (System, PolygonPixelElectrode, euler_matrix,
                       PointPixelElectrode, PotentialObjective,
                       PatternRangeConstraint, shaped)
import sion as sn

#trap parameters
L = 1e-6  # µm length scale
Vrf = 760.  # RF peak voltage in V
mass = 40 * ct.atomic_mass  # ion mass
Z = 1 * ct.elementary_charge  # ion charge
Omega = 2 * np.pi * 100e6  # RF frequency in rad/s
# RF voltage in parametrized form so that the resulting potential equals the RF pseudopotential in eV
#Urf = Vrf * np.sqrt(Z / mass) / (2 * L * Omega)
scale = Z / ((L * Omega) ** 2 * mass)

#theoretically obtained radiuses of the ring for ion's height to be 100 mkm
sc = np.sqrt(3/(4*np.sin(np.pi/12)**2)-1)
r1 = 100/np.sqrt(2)*1e-6
r2 = 100*sc*1e-6
res = 20



#placing 1 ion
ion_number = 1
distance = 0
positions = sn.ions_in_order([0,0,100e-6], ion_number, distance)

#insert your path to this file here
name = Path("/Users/a.podlesnyy/Desktop/RQC/Surface Traps/RingSimulation.py").stem

sim = pl.Simulation(name)

#ion's declaration
ions = {'mass': 40, 'charge': 1}

print('Final ion positions:\n',positions)
sim.append(pl.placeions(ions, positions))

#ring trap initialization
sim.append(sn.ringtrap(Omega, r1,r2, Vrf, resolution = res))

#cooling simulation
sim.append(pl.langevinbath(0, 1e-7))

#files with information
sim.append(pl.dump('posring.txt', variables=['x', 'y', 'z'], steps=10))
sim.append(pl.dump('velring.txt', variables=['vx', 'vy', 'vz'], steps=10))
sim.append(pl.evolve(1e4))
sim.execute()

_, data = pl.readdump('posring.txt')

final_x = data[-1, :, 0]
final_y = data[-1, :, 1]
final_z = data[-1, :, 2]

# Plot of the final ion crystal configuration

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[-1, :, 0], data[-1, :, 1], data[-1, :, 2])
ax.set_xlabel('x (um)')
ax.set_ylabel('y (um)')
ax.set_zlabel('z (um)')
plt.show()

