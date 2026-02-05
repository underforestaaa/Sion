import h5py
import signal
import jinja2 as j2
import json
from datetime import datetime
from collections import defaultdict
import sys
import time

from .utils import save_atttributes_and_files

if 'win32' in sys.platform:
    import wexpect as pexpect
else:
    import pexpect

__version__ = '0.5.0'


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
        self.attrs['executable'] = 'lmp'
        self.attrs['timestep'] = 1e-6
        self.attrs['domain'] = [1e-2, 1e-2, 1e-2]  # length, width, height
        self.attrs['name'] = name
        self.attrs['neighbour'] = {'skin': 1, 'list': 'nsq'}
        self.attrs['coulombcutoff'] = 10
        self.attrs['template'] = 'simulation.j2'
        self.attrs['version'] = __version__
        self.attrs['rigid'] = {'exists': False}

        # # initalise the h5 file
        # with h5py.File(self.attrs['name'] + '.h5', 'w') as f:
        #     pass

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
        env = j2.Environment(loader=j2.PackageLoader('pylion', 'templates'),
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

    @save_atttributes_and_files
    def execute(self):
        """Write lammps input file and run the simulation.
        """

        if getattr(self, '_hasexecuted', False):
            raise SimulationError(
                'Simulation has executed already. Do not run it again.')

        self._writeinputfile()

        def signal_handler(sig, frame):
            print('Simulation terminated by the user.')
            child.terminate()
            # sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        child = pexpect.spawn(' '.join([self.attrs['executable'], '-in',
                              self.attrs['name'] + '.lammps']), timeout=None,
                              encoding='utf8')

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
