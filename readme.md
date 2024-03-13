# *Sion* package

Python package for simulation and analysis of ion crystals in surface electrode traps.

## Installation

To use *Sion*, the following packages are required to be installed:
- [pylion package](https://bitbucket.org/dtrypogeorgos/pylion/src/master/): LAMMPS wrapper for ion dynamics simulation.
- [electrode package](https://github.com/nist-ionstorage/electrode): Python package for convenient definition and analysis of surface electrode traps.

The package is installed by running <code>python setup.py install</code> in console in the same directory with the setup.py file, which must be downloaded.


## Getting started

All functions, presented in *Sion* are described in examples. Main file *sion.py* containes docs for each function.

The package can be used as a standard Python library in any environment.

To publish the results, obtained with *Sion*, the following articles should be cited:
1. [Surface trap with adjustable ion couplings for scalable and parallel gates](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.022605)


## Features
*   Simulation of ion motion in arbitrary polygon and point electrode surface trap.
*   Optimization and simulation of arbitrary ion shuttling in polygon traps.
*   Calculation of normal modes for general case of 1D, 2D, 3D mixed species ion crystals with arbitrary set of ions' secular frequencies.
*   Calculation of anharmonic shift to mode frequencies and mode vectors, induced by the surface trap.
*   Stability analysis of asymmetric planar traps.
*   Optimization of DC voltage set of a planar trap to match the desired secular frequency set in given positions.
*   Convenient trap design. Layout may be imported from GDS file or created by defining the arbitrary electrode shape boundary. 

## File structure

*  'build/lib/sion.py': contains all the main functions.

*  'examples': examples showing different features of *Sion*.

*  'verification': verification of every feature of *Sion*.

Free software: MIT license
