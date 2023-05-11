"""
In this example we model single ion in the RF ring electrode.
The parameters are chosen, so theoretically the position of minimum (and single ion)
will be at the point (0,0, 100) mkm. The final ion position after simulation is
[0.7, 0, 102.7] mkm, which is inaccurate due to the finite resolution of the 
ring electrode, and finite simulation time (ion is not fully cooled at the end
                                            of the simulation)
The increase in accuracy, followed with the increse in the resolution, leads to 
the increase time, required for the simulation.

"""


from __future__ import division
import pylion as pl
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt, numpy as np, scipy.constants as ct
from mpl_toolkits.mplot3d import Axes3D
from electrode import (System, PolygonPixelElectrode, PointPixelElectrode)

import sion as sn

#trap parameters
L = 1e-6  # µm length scale
Vrf = 200.  # RF peak voltage in V
mass = 40 * ct.atomic_mass  # ion mass
Z = 1 * ct.elementary_charge  # ion charge
Omega = 2 * np.pi * 30e6  # RF frequency in rad/s
scale = Z / ((L * Omega) ** 2 * mass)
Urf = Vrf*np.sqrt(Z/mass)/(2*L*Omega)

#theoretically obtained radiuses of the ring for ion's height to be 100 mkm
sc = np.sqrt(3/(4*np.sin(np.pi/12)**2)-1)
r = 100/np.sqrt(2)*1e-6
R = 100*sc*1e-6
res = 40
dif = 0
Nring = 1

#define trap with electrode package.
x, a, [N, j] = sn.linearrings(r*1e6, R*1e6, res, dif, Nring)

ringelectrode = [PointPixelElectrode(points=[xi], areas=[ai]) for
                xi, ai in zip(x,a)] 
s = System(ringelectrode)
rf = np.ones(N)*Urf
s.rfs = rf

#obtain the potential minimum, which will take the resolution into account
x0 = np.array(s.minimum((0., 0, 100), axis=(0, 1, 2), coord=np.identity(3), method="Newton-CG"))

#plot of the ring trap, which will demonstrate the accuracy of chosen resolution
fig, ax = plt.subplots(1,1,figsize=(7, 7))
s.plot_voltages(ax, u=s.rfs)
ax.set_xlim((-R*1.2e6,R*1.2e6))
ax.set_ylim((-R*1.2e6,R*1.2e6))
plt.show()

#placing 1 ion
ion_number = 1
distance = 0
positions = sn.ions_in_order([0,0,110e-6], ion_number, distance)

#insert your path to this file here
name = Path(__file__).stem

sim = pl.Simulation(name)

#ion's declaration
ions = {'mass': 40, 'charge': 1}

sim.append(pl.placeions(ions, positions))

#ring trap initialization
sim.append(sn.ringtrap(Omega, r, R, Vrf, resolution = res))

#cooling simulation
sim.append(pl.langevinbath(0, 1e-7))

#files with information
sim.append(pl.dump('posring.txt', variables=['x', 'y', 'z'], steps=10))
sim.append(pl.dump('velring.txt', variables=['vx', 'vy', 'vz'], steps=10))
sim.append(pl.evolve(1e4))
sim.execute()

_, data = pl.readdump('posring.txt')
data *= 1e6

final_x = data[-1, :, 0]
final_y = data[-1, :, 1]
final_z = data[-1, :, 2]

print('Initial ion positions:\n', [0, 0, 110])
print('Simulated potential minimum:\n', list(x0))
print('Final ion positions:\n', [final_x[0], final_y[0], final_z[0]])

# Plot of the final ion crystal configuration

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[-1, :, 0], data[-1, :, 1], data[-1, :, 2])
ax.set_xlabel('x (um)')
ax.set_ylabel('y (um)')
ax.set_zlabel('z (um)')
plt.show()

