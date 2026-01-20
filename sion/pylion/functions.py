from .lammps import lammps
import numpy as np


@lammps.fix
def efield(uid, ex, ey, ez):
    """Adds a uniform, time-independent e-field to the simulation.
    ex, ey, ez are the magnitudes of the electric field in V/m.

    See Also: http://lammps.sandia.gov/doc/fix_efield.html

    :param ex: x component of electric field
    :param ey: y component of electric field
    :param ez: z component of electric field
    """

    lines = ['\n# Static E-field',
             f'fix {uid} all efield {ex:e} {ey:e} {ez:e}']

    return {'code': lines}


@lammps.ions
def placeions(ions, positions):
    """Places the given ions at the (x, y, z) coordinates specified.

    Example:

    >>> ions = {'mass': 40, 'charge': 1}
    >>> positions = [[1e-4, -0.5e-5, 0], [1e-4, 0, 0], [1e-4, 0.5e-5, 0]]
    >>> placeions(ions, positions)

    :param ions: dict with keys 'charge', 'mass'
    :param positions: list of (x, y , z) coodrinates of each ion
    """

    ions.update({'positions': positions})

    return ions


@lammps.ions
def createioncloud(ions, radius, number):
    """Creates a cloud of ions that can be added to the trap.
    LAMMPS does have a function that can create ions in a cloud-like
    configuration, but it requires a lattice to be declared, and is
    prone to overlapping ions. As a result, we instead calculate individual
    positions and palce them by hand.

    :param ions: dict with keys 'charge', 'mass'
    :param radius: radius of cloud
    :param number: number of atoms
    """

    positions = []

    for ind in range(number):
        d = np.random.random() * radius
        a = np.pi * np.random.random()
        b = 2 * np.pi * np.random.random()

        positions.append([d * np.sin(a) * np.cos(b),
                          d * np.sin(a) * np.sin(b),
                          d * np.cos(a)])

    ions.update({'positions': positions})

    return ions


@lammps.command
def evolve(steps):
    """Evolves the lammps simulation for a certain number of steps.

    See Also: http://lammps.sandia.gov/doc/run.html

    Example:

    >>> evolve(1e6)

    :param steps: number of steps
    """

    lines = ['\n# Run simulation',
             f'run {int(steps):d}\n']

    return {'code': lines}


@lammps.command
def thermalvelocities(temperature, zerototalmomentum=True):
    """Sets the velocities of the ions to those given by a thermal
    distribution of input temperature (Kelvin). Set zeroTotalMom to 'True' if
    the resulting ensemble should have zero total linear momentum.

    Example:

    >>> thermalVelocities(300, zerototalmomentum=False)

    See Also: http://lammps.sandia.gov/doc/velocity.html

    :param temperature: temperature of thermal distribution
    :param zerototalmomentum: boolean
    """

    seed = np.random.randint(1, 1e5)

    if zerototalmomentum:
        tot = 'yes'
    else:
        tot = 'no'

    lines = [f'\nvelocity all create {temperature:e} {seed:d} mom {tot} rot yes dist gaussian\n']

    return {'code': lines}


@lammps.command
def minimise(etol, ftol, maxiter, maxeval, maxdist):
    """Perform an energy minimization of the system, by iteratively
    adjusting atom coordinates.
    The system is minimized by evolving the equations of motion with large
    damping, which saps energy from it. The maximum distance an atom
    can move in an iteration step is determined by maxdist. The minimisation
    is terminated when one of the following criteria is met:

    1. energy difference between steps less than etol.
    2. total force is less than ftol
    3. the number of iterations exceeds maxiter
    4. the number of force evaluations exceeds maxeval

    See Also: http://lammps.sandia.gov/doc/minimize.html

    :param etol: energy difference tolerance
    :param ftol: force tolerance
    :param maxiter: maximum number of iterations
    :param maxeval: maximum number of force evaluations
    :param maxdist: maximum distance atoms can move
    """

    lines = ['\n# minimize',
             'min_style quickmin',
             f'min_modify dmax {maxdist:e}',
             f'minimize {etol:e} {ftol:e} {maxiter:d} {maxeval:d}\n']

    return {'code': lines}


@lammps.fix
def ionneutralheating(uid, ions, rate):
    """Average heating effect due to collision of ions with
    background gas. Applies a small velocity kick to atoms each timestep
    using a normal veolcity distribution.

    :param ions: species to apply the kicks to
    :param rate: heating rate
    """

    rate = abs(rate)  # this is heating after all
    au = 1.66e-27
    iid = ions['uid']
    mass = ions['mass']

    lines = ['\n# Define ion-neutral heating for a species...',
             f'group {uid} type {iid}',
             f'variable k{uid} equal "sqrt(dt * dt * dt * dt * 2 / {mass*au:e} * {rate:e})"',
             f'variable k{uid} equal "sqrt(2 * {rate:e} * {mass*au:e} / 3 / dt)"',
             f'variable f{uid}\t\tatom normal(0:d,v_k{uid},1337)',
             f'fix {uid} {iid} addforce v_f{uid} v_f{uid} v_f{uid}\n']

    return {'code': lines}


@lammps.fix
def langevinbath(uid, temperature, dampingtime):
    """Creates a langevin bath of a given temperature.
    The langevin bath applies a damping force to each atom proportional to
    its velocity plus a stochastic, white noise force of a magnitude such
    that after a time significantly longer than the 'dampingtime' the system
    will thermalise to the specified temperature. The damping time is the
    time taken for velocity to relax to 1/e of its initial value in a zero
    temperature bath.

    See Also: lasercool, http://lammps.sandia.gov/doc/fix_langevin.html

    :param temperature: temperature
    :param dampingtime: effectively defines coupling strength to the bath
    """

    lines = ['\n# Adding a langevin bath...',
             f'fix {uid} all langevin {temperature:e} {temperature:e} {dampingtime:e} 1337\n']

    return {'code': lines}


@lammps.fix
def lasercool(uid, ions, k):
    """Simulates laser cooling of a particular ion species by damping the
    velocity of the ions. kx, ky, kz define the strength of the damping force,
    which is of the form :math:`f_i = - k_i * v_i`.

    See Also: langevinbath

    :param ions: select species of ions
    :param k: (kx, ky, kz) laser wavevector
    """

    kx, ky, kz = k
    iid = ions['uid']

    lines = ['\n# Define laser cooling for a particular atom species.',
             f'group {uid} type {iid}',
             f'variable kx{uid}\t\tequal {kx:e}',
             f'variable ky{uid}\t\tequal {ky:e}',
             f'variable kz{uid}\t\tequal {kz:e}',
             f'variable fX{uid} atom "-v_kx{uid} * vx"',
             f'variable fY{uid} atom "-v_ky{uid} * vy"',
             f'variable fZ{uid} atom "-v_kz{uid} * vz"',
             f'fix {uid} {uid} addforce v_fX{uid} v_fY{uid} v_fZ{uid}\n']

    return {'code': lines}


def _rftrap(uid, trap):
    odict = {}
    ev = trap['endcapvoltage']
    radius = trap['radius']
    length = trap['length']
    kappa = trap['kappa']
    anisotropy = trap.get('anisotropy', 1)
    offset = trap.get('offset', (0, 0))

    odict['timestep'] = 1 / np.max(trap['frequency']) / 20

    lines = [f'\n# Creating a Linear Paul Trap... (fixID={uid})',
             f'variable endCapV{uid}\t\tequal {ev:e}',
             f'variable radius{uid}\t\tequal {radius:e}',
             f'variable zLength{uid}\t\tequal {length:e}',
             f'variable geomC{uid}\t\tequal {kappa:e}',
             '\n# Define frequency components.']

    voltages = []
    freqs = []
    if hasattr(trap['voltage'], '__iter__'):
        voltages.extend(trap['voltage'])
        freqs.extend(trap['frequency'])
    else:
        voltages.append(trap['voltage'])
        freqs.append(trap['frequency'])

    for i, (v, f) in enumerate(zip(voltages, freqs)):
        lines.append(f'variable oscVx{uid}{i:d}\t\tequal {v:e}')
        lines.append(f'variable oscVy{uid}{i:d}\t\tequal {anisotropy*v:e}')
        lines.append(f'variable phase{uid}{i:d}\t\tequal "{2*np.pi*f:e} * step*dt"')
        lines.append(f'variable oscConstx{uid}{i:d}\t\tequal "v_oscVx{uid}{i:d}/(v_radius{uid}*v_radius{uid})"')
        lines.append(f'variable oscConsty{uid}{i:d}\t\tequal "v_oscVy{uid}{i:d}/(v_radius{uid}*v_radius{uid})"')

    lines.append(
        f'variable statConst{uid}\t\tequal "v_geomC{uid} * v_endCapV{uid} / (v_zLength{uid} * v_zLength{uid})"\n')

    xc = []
    yc = []

    xpos = f'(x-{offset[0]:e})'
    ypos = f'(y-{offset[1]:e})'

    # Simplify this case for 0 displacement
    if offset[0] == 0:
        xpos = 'x'

    if offset[1] == 0:
        ypos = 'y'

    for i, _ in enumerate(voltages):
        xc.append(f'v_oscConstx{uid}{i:d} * cos(v_phase{uid}{i:d}) * {xpos}')
        yc.append(f'v_oscConsty{uid}{i:d} * cos(v_phase{uid}{i:d}) * -{ypos}')

    xc = ' + '.join(xc)
    yc = ' + '.join(yc)

    lines.append(f'variable oscEX{uid} atom "{xc} + v_statConst{uid} * {xpos}"')
    lines.append(f'variable oscEY{uid} atom "{yc} + v_statConst{uid} * {ypos}"')
    lines.append(f'variable statEZ{uid} atom "v_statConst{uid} * 2 * -z"')
    lines.append(f'fix {uid} all efield v_oscEX{uid} v_oscEY{uid} v_statEZ{uid}\n')

    odict.update({'code': lines})

    return odict


def _pseudotrap(uid, k, group='all'):

    lines = [f'\n# Pseudopotential approximation for Linear Paul trap... (fixID={uid})']

    # Add a cylindrical SHO for the pseudopotential
    kx, ky, kz = k

    sho = ['\n# SHO',
           f'variable k_x{uid}\t\tequal {kx:e}',
           f'variable k_y{uid}\t\tequal {ky:e}',
           f'variable k_z{uid}\t\tequal {kz:e}',
           f'variable fX{uid} atom "-v_k_x{uid} * x"',
           f'variable fY{uid} atom "-v_k_y{uid} * y"',
           f'variable fZ{uid} atom "-v_k_z{uid} * z"',
           f'variable E{uid} atom "v_k_x{uid} * x * x / 2 + v_k_y{uid} * y * y / 2 + v_k_z{uid} * z * z / 2"',
           f'fix {uid} {group} addforce v_fX{uid} v_fY{uid} v_fZ{uid} energy v_E{uid}\n']

    lines.extend(sho)

    return {'code': lines}


@lammps.fix
def linearpaultrap(uid, trap, ions=None, all=True):
    """Applies an oscillating electric field to atoms. The
    characterisation of the trap follows Berkeland et al. (1998).
    'trap' shoud be a dictionary with the following items:

    - 'radius', of the trap in meters
    - 'length', of the trap in meters
    - 'kappa', is a geometric factor defined in Berkeland et al.
    - 'frequency', should be in Hz, not radians per second.
    - 'voltage', is the voltage of the rf electrodes
    - 'endcapvoltage', the voltage of the endcaps

    The are also three optional parameters:
    - 'anisotropy', is used to imbalance fields in x and y directions,
    such that V_y = anisotropy * V_x. Defaults to 1.
    - 'offset', moves the center of the trap away from the rf-null axis.
    Defaults to (0, 0).
    - 'pseudo', boolean to choose between the full rf trap or the corresponding
    pseudopoential. Defaults to False.

    'frequency' and 'voltage' can be specified as vectors, in which case a
    multi-frequency Paul trap is created.

    As the pseudopotential is dependent on the charge:mass ratio of the ion,
    this fix requires that an 'ions' dict be supplied unless the 'all'
    parameter is True.

    See Also: http://tf.nist.gov/general/pdf/1226.pdf

    :param trap: dictionary containing trap parameters
    :param ions: species to be used for pseudopotential
    :param all: boolean that chooses beteween the pseudopotential applied to
      all the ions in the simulation or just a single species
    """

    if trap.get('pseudo'):
        charge = ions['charge'] * 1.6e-19
        mass = ions['mass'] * 1.66e-27
        ev = trap['endcapvoltage']
        radius = trap['radius']
        length = trap['length']
        kappa = trap['kappa']
        freq = trap['frequency']
        voltage = trap['voltage']

        ar = -4 * charge * kappa * ev / (mass * length**2 * (2*np.pi * freq)**2)
        az = -2*ar

        qr = 2 * charge * voltage / (mass * radius**2 * (2*np.pi * freq)**2)

        wr = 2*np.pi * freq / 2 * np.sqrt(ar + qr**2 / 2)
        wz = 2*np.pi * freq / 2 * np.sqrt(az)

        print(f'Frequency of motion: fr = {wr/2/np.pi:e}, fz = {wz/2/np.pi:e}')

        # Spring constants for force calculation.
        kr = wr**2 * mass
        kz = wz**2 * mass

        odict = {}
        odict['timestep'] = 1 / max(wz, wr) / 10

        if all:
            group = 'all'
        else:
            group = ions['uid']

        sho = _pseudotrap(uid, (kr, kr, kz), group)

        odict.update(sho)
        return odict
    else:
        return _rftrap(uid, trap)


@lammps.variable('fix')
def timeaverage(uid, steps, variables):
    """A variable in LAMMPS representing a time averaged quantity over a
    number of steps.

    :param stes: number of steps to average over
    :param variables: list of variables to be averaged
    """

    variables = ' '.join(variables)

    lines = [f'fix {uid} all ave/atom 1 {steps:d} {steps:d} {variables}\n']

    return {'code': lines}


@lammps.variable('var')
def squaresum(uid, variables):
    """Creates a lammps variable that calculates the square sum of the
    input variables.

    :param variables: list of variables
    """

    vsq = [f'{v}^2' for v in variables]
    sqs = '+'.join(vsq)

    lines = [f'variable {uid} atom "{sqs}"\n']

    return {'code': lines}


@lammps.fix
def dump(uid, filename, variables, steps=10):
    """Dumps variables from lammps into files for analysis.

    :param filename: name of output file
    :param variables: list of variables to be written
    :param steps: variables are written every steps
    """

    lines = []

    try:
        names = variables['output']
        lines.extend(variables['code'])
    except:
        names = ' '.join(variables)

    lines.append(f'dump {uid} all custom {steps:d} {filename} id {names}\n')

    return {'code': lines}


def trapaqtovoltage(ions, trap, a, q):
    """Calculates trap voltages for given a, q parameters.

    :return: tuple of (voltage, endcapvoltage)
    """

    mass = ions['mass'] * 1.66e-27
    charge = ions['charge'] * 1.6e-19
    radius = trap['radius']
    length = trap['length']
    kappa = trap['kappa']
    freq = trap['frequency']

    endcapV = a * mass * length**2 * (2*np.pi * freq)**2 / (-kappa * 4*charge)
    oscV = -q * mass * radius**2 * (2*np.pi * freq)**2 / (2*charge)

    return oscV, endcapV


def readdump(filename):
    """Reads data from the given dump file. The dump should be a file
    with atom quantities in the order `id vargout`, e.g. `id vx vy vz`.

    :param filename: name of input file
    :return: a tuple of (steps, data).
      The shape of data is (steps, ions, (x, y, z)).
    """

    steps = []
    data = []
    import time

    with open(filename, 'r') as f:
        for line in f:
            if line[6:9] == 'TIM':
                steps.append(next(f))
            elif line[6:9] == 'NUM':
                ions = int(next(f))
            elif line[6:9] == 'ATO':
                if line[12:14] != 'id':
                    raise TypeError
                block = [next(f).split()[1:] for _ in range(ions)]
                data.append(block)

    steps = np.array(steps, dtype=float)
    data = np.array(data, dtype=float)  # shape=(steps, ions, (x,y,z))
    return steps, data

