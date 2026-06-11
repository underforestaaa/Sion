# *SION* package

Python package for simulation and analysis of ion crystals in surface electrode traps.


***
The package combines voltage optimization algorithms with ion-dynamics simulation in LAMMPS through the [electrode](https://github.com/nist-ionstorage/electrode) and [pyLIon](https://bitbucket.org/dtrypogeorgos/pylion/src/master/) packages.
LAMMPS-based molecular dynamics simulations provide a reliable baseline for modeling ion-crystal behavior.
***


## Installation



This package was developed and tested on **Windows only**.

The newest version may be installed via pip:  
<code>pip install surface-ion</code>


The following software must be installed to use *SION*:  
*  Molecular Dynamics simulations use [LAMMPS](https://www.lammps.org/download.html).

*  Parallel runs use [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (MS-MPI). Install the M


## Getting started

*SION* works with surface traps, defined through the *electrode* package. The simulation of ion dynamics is carried through the *pylion* environment.
All functions, presented in *SION* are described in example notebooks. Main file *sion.py* contains docs for each function.

To publish the results, obtained with *SION*, we kindly ask you to cite the following article:
1. [Surface trap with adjustable ion couplings for scalable and parallel gates](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.022605)

Free software: GNU GENERAL PUBLIC LICENSE

## Features

*   Simulation of ion motion in arbitrary polygon and point electrode surface trap.
*   Optimization and simulation of arbitrary ion shuttling in polygon traps.
*   Calculation of normal modes for general case of 1D, 2D, 3D mixed species ion crystals with arbitrary set of ions' secular frequencies.
*   Calculation of anharmonic Mathieu modes of ion crystals in surface traps.  
*   Stability analysis of asymmetric planar traps.
*   Optimization of DC voltage set of a planar trap to match the desired secular frequency and radial mode rotation angle in given positions.
*   Convenient trap design. Layout may be imported from GDS file or created by defining the arbitrary electrode shape boundary. 

## Possible issues

*  The following error may occur from trying to execute simulation two times without restarting the kernel. It is specific to IPython IDEs (Jupyter, spyder). The error will be resolved by restarting the kernel.

<code>SimulationError: There are identical 'uids'. Although this is allowed in some  cases, 'lammps' is probably not going to like it.</code>

