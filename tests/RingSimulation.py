"""
In this example we model single ion in the RF ring electrode.
The parameters are chosen, so theoretically the position of minimum (and single ion)
will be at the point (0,0, 100) mkm. 
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


if __name__ == "__main__":
    #trap parameters
    Vrf = 120.  # RF peak voltage in V
    mass = 40 * ct.atomic_mass  # ion mass
    Z = 1 * ct.elementary_charge  # ion charge
    Omega = 2 * np.pi * 30e6  # RF frequency in rad/s
    Urf = Vrf*np.sqrt(Z/mass)/(2*Omega)
    
    #theoretically obtained radiuses of the ring for ion's height to be 100 mkm
    sc = np.sqrt(3/(4*np.sin(np.pi/12)**2)-1)
    r = 100/np.sqrt(2)*1e-6
    R = 100*sc*1e-6
    res = 50
    dif = 0
    Nring = 1
    scale = 1.5*R
    
    def boundaries(i, x):
        if i == 0:
            if (x[0]**2+x[1]**2 > r**2) and (x[0]**2+x[1]**2 < R**2):
                return True
            else:
                return False
    
    s, trap = sn.point_trap_design(frequencies = [Omega], rf_voltages = [Vrf], dc_voltages = [], boundaries = boundaries, scale = scale, resolution = res, need_plot = True, need_coordinates = True)
    
    
    #placing 1 ion
    ion_number = 1
    distance = 0
    positions = sn.ions_in_order([0,0,95e-6], ion_number, distance)
    
    #insert your path to this file here
    name = Path('RingSimulation.py').stem
    
    sim = pl.Simulation(name)
    
    #ion's declaration
    ions = {'mass': 40, 'charge': 1}
    
    sim.append(pl.placeions(ions, positions))
    
    #ring trap initialization
    sim.append(sn.point_trap(trap))
    
    #cooling simulation
    sim.append(pl.langevinbath(0, 1e-7))
    
    #files with information
    sim.append(pl.dump('posring.txt', variables=['x', 'y', 'z'], steps=10))
    sim.append(pl.evolve(1e4))
    #sim.execute()
    
    
    _, data = pl.readdump('posring.txt')
    data *= 1e6
    
    final_x = data[-1, :, 0]
    final_y = data[-1, :, 1]
    final_z = data[-1, :, 2]
    
    print('Initial ion position:\n', [0, 0, 95])
    print('Final ion position:\n', np.round([final_x[0], final_y[0], final_z[0]]), 'um')
    print('Distance from theoretical position:', np.round(np.linalg.norm(np.array([final_x[0], final_y[0], final_z[0]]) - np.array([0, 0, 100])), 3), 'um')
    
    plt.figure()
    for n in range(ion_number):
        plt.plot(np.arange(data.shape[0]) * 10 + 1, data[:, n, 2])
    plt.title('Evolution of ion\'s z coordinates')
    plt.xlabel('Time step')
    plt.ylabel('z, um')
    plt.show()
