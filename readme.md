# *Sion* package

Python package for simulation and analysis of ion crystals in surface electrode traps.

## Installation

To use *Sion*, the following packages are required to be installed from source and tested:
- [pylion package](https://bitbucket.org/dtrypogeorgos/pylion/src/master/): LAMMPS wrapper for ion dynamics simulation. This package requires installation of LAMMPS software with the specific version. It will work well with the latest LAMMPS version, if in source file *pylion.py* at line 51 you change "lmp_serial" to "lmp".  
- [electrode package](https://github.com/nist-ionstorage/electrode): Python package for convenient definition and analysis of surface electrode traps.  
!Note: for correct execution of these packages, numpy<=1.21.0 is recquired. Alternatively one may install 

The newest version may be installed via pip:  
<code>pip install surface-ion</code>


## Getting started

*Sion* works with surface traps, defined through the *electrode* package. The simulation of ion dynamics is carried through the *pylion* environment.
All functions, presented in *Sion* are described in example notebooks. Main file *sion.py* containes docs for each function.

To publish the results, obtained with *Sion*, the following articles should be cited:
1. [Surface trap with adjustable ion couplings for scalable and parallel gates](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.022605)


## Features
*   Simulation of ion motion in arbitrary polygon and point electrode surface trap.
*   Optimization and simulation of arbitrary ion shuttling in polygon traps.
*   Calculation of normal modes for general case of 1D, 2D, 3D mixed species ion crystals with arbitrary set of ions' secular frequencies.
*   Calculation of anharmonic mathieu modes of ion crystals in surface traps.  
*   Stability analysis of asymmetric planar traps.
*   Optimization of DC voltage set of a planar trap to match the desired secular frequency and radial mode rotation angle in given positions.
*   Convenient trap design. Layout may be imported from GDS file or created by defining the arbitrary electrode shape boundary. 

## File structure

*  'build/lib/sion.py': contains all the main functions.

*  'examples': examples showing different features of *Sion*.

*  'tests': verifications and tests of *Sion* work.

Free software: MIT license
