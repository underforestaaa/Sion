# Modified for SION.

from sys import platform
# namespace so code can keep using sys.xxx
class _Sys:
    pass
sys = _Sys()
sys.platform = platform

from .utils import save_atttributes_and_files

if 'win32' in sys.platform:
    import wexpect as pexpect  # type: ignore[import-untyped]
else:
    import pexpect  # type: ignore[import-untyped]

from h5py import File
class _H5:
    pass
h5py = _H5()
h5py.File = File

from signal import signal as set_signal, SIGINT
class _Signal:
    pass
signal = _Signal()
signal.signal, signal.SIGINT = set_signal, SIGINT

from jinja2 import Environment, PackageLoader
class _J2:
    pass
j2 = _J2()
j2.Environment, j2.PackageLoader = Environment, PackageLoader

from json import dumps, loads
class _Json:
    pass
json = _Json()
json.dumps, json.loads = dumps, loads

from datetime import datetime
from collections import defaultdict
import os
import shutil

# SION mod: MPI support for LAMMPS (e.g. MS-MPI on Windows)
# - auto-detects lmp.exe and mpiexec paths
# - sim.set_mpi(processes=N) to enable parallel runs
DEFAULT_LAMMPS_PATHS = [r'C:\LAMMPS\bin\lmp.exe', 'lmp']
DEFAULT_MPIEXEC_PATHS = [r'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe', 'mpiexec']

__version__ = '0.5.3_SION_mod1'


class SimulationError(Exception):
    """Custom error class for Simulation."""
    pass


class Attributes(dict):
    """Light dict wrapper to serve as a container of attributes."""

    def save(self, filename):
        with h5py.File(filename, 'a') as f:
            print(f'Saving attributes to {filename}')
            f.attrs.update({k: json.dumps(v)
                            for k, v in self.items()})

    def load(self, filename):
        with h5py.File(filename, 'r') as f:
            return {k: json.loads(v) for k, v in f.attrs.items()}


class Simulation(list):

    def __init__(self, name='pylion'):
        super().__init__()

        # keep track of uids for list function overrides
        self._uids = []

        # slugify 'name' to use for filename
        name = name.replace(' ', '_').lower()

        self.attrs = Attributes()
        self.attrs['gpu'] = None
        self.attrs['executable'] = self._find_lammps_executable()
        self.attrs['thermo_styles'] = ['step', 'cpu']
        self.attrs['timestep'] = 1e-6
        self.attrs['domain'] = [1e-2, 1e-2, 1e-2]  # length, width, height
        self.attrs['name'] = name
        self.attrs['neighbour'] = {'skin': 1, 'list': 'nsq'}
        self.attrs['coulombcutoff'] = 10
        self.attrs['template'] = 'simulation.j2'
        self.attrs['version'] = __version__
        self.attrs['rigid'] = {'exists': False}

        # mpi/omp: set via set_parallel(mpi_processes=N, omp_threads=M) or execute(..., mpi_processes=..., omp_threads=...)
        self.attrs['mpi'] = {
            'enabled': os.environ.get('USE_MPI', '0') == '1',
            'processes': int(os.environ.get('MPI_NUM_PROCESSES', '4')),
            'executable': self._find_mpiexec(),
        }
        self.attrs['omp_threads'] = int(os.environ.get('OMP_NUM_THREADS', '1'))

    @staticmethod
    def _find_lammps_executable():
        """Try to find lmp.exe in common locations."""
        for path in DEFAULT_LAMMPS_PATHS:
            if os.path.isfile(path):
                return path
            found = shutil.which(path)
            if found:
                return found
        return 'lmp'

    @staticmethod
    def _find_mpiexec():
        """Try to find mpiexec in common locations."""
        for path in DEFAULT_MPIEXEC_PATHS:
            if os.path.isfile(path):
                return path
            found = shutil.which(path)
            if found:
                return found
        return None

    def set_mpi(self, enabled=True, processes=4, executable=None):
        """Turn on MPI. Call like sim.set_mpi(processes=8)."""
        self.attrs['mpi']['enabled'] = enabled
        self.attrs['mpi']['processes'] = processes
        if executable:
            self.attrs['mpi']['executable'] = executable
        elif not self.attrs['mpi']['executable']:
            self.attrs['mpi']['executable'] = self._find_mpiexec()
        return self

    def set_parallel(self, mpi_processes=None, omp_threads=None, mpiexec=None):
        """Set MPI and OpenMP for the run. Call like sim.set_parallel(mpi_processes=4, omp_threads=1)."""
        if mpi_processes is not None:
            self.set_mpi(enabled=True, processes=mpi_processes, executable=mpiexec)
        if omp_threads is not None:
            self.attrs['omp_threads'] = int(omp_threads)
        return self

    def __contains__(self, this):
        """Check if an item exists in the simulation using its ``uid``.
        """

        try:
            return this['uid'] in self._uids
        except KeyError:
            print("Item does not have a 'uid' key.")

    def append(self, this):
        """Appends the items and checks their attributes.
        Their ``uid`` is logged if they have one.
        """

        # only allow for dicts in the list
        if not isinstance(this, dict):
            raise SimulationError("Only 'dicts' are allowed in Simulation().")

        self._uids.append(this.get('uid'))

        # ions will always be included first so to sort you have
        # to give 1-count 'priority' keys to the rest
        if this.get('type') == 'ions':
            this['priority'] = 0
            if this.get('rigid'):
                self.attrs['rigid']['exists'] = True
                self.attrs['rigid'].setdefault('groups',
                                               []).append(this['uid'])

        timestep = this.get('timestep', 1e12)
        if timestep < self.attrs['timestep']:
            print(f'Reducing timestep to {timestep} sec')
            self.attrs['timestep'] = timestep

        super().append(this)

    def extend(self, iterable):
        """Calls ``append`` on an iterable.
        """

        for item in iterable:
            self.append(item)

    def index(self, this):
        """Returns the index of an item using its ``uid``.
        """

        return self._uids.index(this['uid'])

    def remove(self, this):
        """Will not remove anything from the simulation but rather from lammps.
        It adds an ``unfix`` command when it's called.
        Use del if you really want to delete something or better yet don't
        add it to the simulation in the first place.
        """

        code = ['\n# Deleting a fix', f"unfix {this['uid']}\n"]
        self.append({'code': code, 'type': 'command'})

    def sort(self):
        """Sort with 'priority' keys if found otherwise do nothing.
        """

        try:
            super().sort(key=lambda item: item['priority'])
        except KeyError:
            pass
            # Not all elements have 'priority' keys. Cannot sort list

    def _writeinputfile(self):

        self.sort()  # if 'priority' keys exist

        odict = defaultdict(list)
        # deal the items in odict
        for item in self:
            if item.get('type') == 'ions':
                odict['species'].append(item)
            else:
                odict['simulation'].append(item)

        # do a couple of checks
        # check for uids clashing
        uids = list(filter(None.__ne__, self._uids))
        if len(uids) > len(set(uids)):
            raise SimulationError(
                "There are identical 'uids'. Although this is allowed in some "
                " cases, 'lammps' is probably not going to like it.")

        # make sure species will behave
        maxuid = max(odict['species'], key=lambda item: item['uid'])['uid']
        if maxuid > len(odict['species']):
            raise SimulationError(
                f"Max 'uid' of species={maxuid} is larger than the number "
                f"of species={len(odict['species'])}. "
                "Calling '@lammps.ions' decorated functions increments the "
                "'uid' count unless it is for the same ion group.")

        # load jinja2 template
        env = j2.Environment(loader=j2.PackageLoader('sion.pylion', 'templates'),
                             trim_blocks=True)
        template = env.get_template(self.attrs['template'])
        rendered = template.render({**self.attrs, **odict})

        with open(self.attrs['name'] + '.lammps', 'w') as f:
            f.write(rendered)

        # get a few more attrs now that the lammps file is written
        # - simulation time
        self.attrs['time'] = datetime.now().isoformat()

        # - names of the output files
        fixes = filter(lambda item: item.get('type') == 'fix',
                       odict['simulation'])
        self.attrs['output_files'] = [line.split()[5] for fix in fixes
                                      for line in fix['code']
                                      if line.startswith('dump')]

    def _build_command(self):
        """Builds the lammps command, with mpiexec if enabled."""
        lammps_args = [
            self.attrs['executable'],
            '-log', self.attrs['name'] + '.lmp.log',
            '-in', self.attrs['name'] + '.lammps',
        ]
        mpi_cfg = self.attrs.get('mpi', {})
        use_mpi = mpi_cfg.get('enabled', False)
        mpiexec = mpi_cfg.get('executable')
        num_procs = mpi_cfg.get('processes', 4)
        if use_mpi and mpiexec:
            cmd_parts = [
                f'"{mpiexec}"' if ' ' in mpiexec else mpiexec,
                '-n', str(num_procs),
                f'"{self.attrs["executable"]}"' if ' ' in self.attrs['executable'] else self.attrs['executable'],
                '-log', self.attrs['name'] + '.lmp.log',
                '-in', self.attrs['name'] + '.lammps',
            ]
            return ' '.join(cmd_parts)
        if use_mpi and not mpiexec:
            print('[!] mpi enabled but mpiexec not found, running single process')
        return ' '.join(lammps_args)

    @save_atttributes_and_files
    def execute(self, mpi_processes=None, omp_threads=None):
        """Write lammps input file and run the simulation.

        Optional: mpi_processes, omp_threads override parallelization for this run
        (otherwise use set_parallel() or attrs).
        """
        if getattr(self, '_hasexecuted', False):
            raise SimulationError(
                'Simulation has executed already. Do not run it again.')

        if mpi_processes is not None:
            self.set_mpi(enabled=True, processes=mpi_processes)
        if omp_threads is not None:
            self.attrs['omp_threads'] = int(omp_threads)

        omp = self.attrs.get('omp_threads', 1)
        os.environ['OMP_NUM_THREADS'] = str(omp)
        if self.attrs.get('mpi', {}).get('enabled'):
            print(f'[parallel] {self.attrs["mpi"]["processes"]} MPI × {omp} OMP')

        self._writeinputfile()

        def signal_handler(sig, frame):
            print('Simulation terminated by the user.')
            child.terminate()

        signal.signal(signal.SIGINT, signal_handler)

        cmd = self._build_command()
        child = pexpect.spawn(cmd, timeout=None, encoding='utf8')

        self._process_stdout(child)
        child.close()

        self._hasexecuted = True

    def _process_stdout(self, child):
        atoms = 0
        for line in child:
            line = line.rstrip('\r\n')
            if line == 'Created 1 atoms':
                atoms += 1
                continue
            elif line == 'Created 0 atoms':
                raise SimulationError(
                    'lammps created 0 atoms - perhaps you placed ions '
                    'with positions outside the simulation domain?')

            if atoms:
                print(f'Created {atoms} atoms.')
                atoms = False
                continue

            print(line)
