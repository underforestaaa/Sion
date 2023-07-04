# *Sion* package

Python package for simulation and analysis of ion crystals in surface electrode traps.

## Installation

To use *Sion*, the following packages are required to be installed:
- [pylion package](https://bitbucket.org/dtrypogeorgos/pylion/src/master/): LAMMPS wrapper for ion dynamics simulation.
- [electrode package](https://github.com/nist-ionstorage/electrode): Python package for convenient definition and analysis of surface electrode traps.

The package is installed by running <code>python setup.py install</code> in console in the same directory with the setup.py file, which must be downloaded.


## Getting started

All functions, presented in *Sion* are described in examples. The package can be used as a standard Python library in any environment.

To publish the results, obtained with *Sion*, the following articles must be cited:
1. [Surface trap with adjustable ion couplings for scalable and parallel gates](https://doi.org/10.48550/arXiv.2211.07121)


## Features
*   Simulation of ion motion in arbitrary polygon and point surface trap.
*   Optimization and simulation of linear ion suttling in polygon traps.
*   Calculation of normal modes for general case of 1D, 2D, 3D mixed species ion crystals with arbitrary set of ions' secular frequences.
*   Calculation of anharmonic normal modes in 1D ion chains.
*   Stability analysis of assymetric planar traps.
*   Optimization of DC voltage set and geometry of a planar trap to match the desired secular frequency set in given positions. 


## Functionality

The list of *Sion* functions and their brief description. 

### Simulation
Check *examples/FiveWireSimulation.py, polygon_simulation.py, RingSimulation.py*.

* <code>five_wire_trap()</code>: Simulates simple five-wire planar trap, produced by the *FiveWireTrap()* function.
* <code>polygon_trap()</code>: Simulates an arbitrary planar trap formed with polygonal electrodes.
* <code>point_trap()</code>: Simulates an arbitrary point trap. The point trap means a trap of an arbitrary shape, which is approximated by circle-shaped electrodes, called points. The points potential is approximated from the fact, that it has infinitesemal radius, so the smaller are points, the more precise is the simulation (but slower).
* <code>ringtrap()</code>: Simulates a ring-shaped RF electrode.

### Shuttling
Check *examples/linear_shuttling.ipynb*.

* <code>linear_shuttling_voltage()</code>: Optimizes the voltage sequence on all DC electrodes to perform low-heating linear shuttling.
* <code>approx_shuttling()</code>: Fits voltage sequences to the analytic functions, suitable for LAMMPS simulation.
* <code>polygon_shuttling()</code>: Performs simulation of ions in the polygonal trap with DC electrode voltages, arbitrarily changing in time.

### Trap definition

* <code>FiveWireTrap()</code>: A function for a very detailed definition of five-wire traps. It allows to define side DC electrodes only by their width and height, and the number of repeating electrodes of the same shapes. It also allows to obtain the coordinates with the respect to the gap width between electrodes. If specified, a plot of the trap layout may be deomnstrated. 
* <code>linearrings()</code>: Returns an array of RF rings placed in a row, with a DC circle in each center.
* <code>n_rf_trap()</code>: A function for convenient definition of multi-wire traps, with *n* asymmetrical RF lines.
* <code>individual_wells()</code>: Polygonal RF electrode with arbitrary rectangular notch generation in the determined positions for individual potential wells and geometry optimization.

### Useful tools for initializing the simulation

* <code>ioncloud_min()</code>: Generates positions of ions in random ion cloud near the given point.
* <code>ions_in_order()</code>: Positions of ions placed in linear order near the given point.

### Normal modes
Check *examples/2D_crystal_modes.ipynb, mixed_species_modes.py*.

* <code>linear_axial_modes()</code>: Calculates axial normal modes of a linear ion chain.
* <code>linear_radial_modes()</code>: Calculates radial normal modes of a linear ion chain, for y' or z' - rotated radial secular modes.
* <code>normal_modes()</code>: Normal modes of an ion crystal with arbitrary configuration. It may include non-linear ion chains (2D, 3D crystals), mixed-species ions, ions, experiencing different secular frequencies (tweezers, arrays of microtraps).

### Anharmonic parameters and modes
Check *examples/anharmonic_modes.py*.

* <code>anharmonics()</code>: Calculates anharmonic scale lengths at given coordinates. The lengthes are given for harmonic, n = 3 and n = 4 terms. Length are defined as in DOI 10.1088/1367-2630/13/7/073026.
* <code>anharmonic_modes()</code>: Anharmonic normal modes along the chosen pricniple axis of oscillation in a linear ion chain, considering hexapole and octopole anharmonicities. Defined for the mixed species ion crystals in individual potential wells.

### Stability analysis
Check *examples/stability.py*.

* <code>stability()</code>: Returns stability parameters for the linear planar trap (RF confinement only radial). If asked, return plot of the stability a-q diagram for this trap and plots the parameters for this voltage configuration (as a red dot).

### Optimization
Check *examples/individual_wells_voltage_optimization.ipynb, individual_wells_geometry_optimization.ipynb*.

* <code>voltage_optimization()</code>: A function, determining the DC voltage set for the trap, at which the set of desired secular frequencies is achieved at the requested positions, using the ADAM optimization. The final positions of potential minima, however, will slightly change due to the change in the DC potential. The secular frequencies may be determined in all the required principle axes of oscillation simultaneously. The optimization is very slow, and may take hours, so at each number of iterations, determined by the user, it will print the current result.
* <code>geometry_optimization()</code>: Function, optimizating the shape of RF electrode from the *individual_wells()* function, to obtain individual wells with the required secular frequencies. Optmization is different from the reverse geometry optimization, since it optimizes the polygonal geometry for the arbitrary secular frequency set.


## File structure

*  'build/lib/sion.py': contains all the main functions.

*  'examples': examples showing different features of *Sion*.

Free software: MIT license
