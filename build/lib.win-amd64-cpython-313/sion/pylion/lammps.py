from .utils import validate_id, _unique_id, pretty_repr
import functools


@pretty_repr
class CfgObject:
    def __init__(self, func, lmp_type, required=None):
        if not required:
            required = []

        self.func = func

        # use default keys 'code', 'type' and update if there is anything else
        # __call__ will overwrite code except for ions
        required = [x if isinstance(x, tuple) else (x, None)
                    for x in required + [('code', []), 'type']]

        self.odict = dict(required)
        self.odict['type'] = lmp_type

        # add dunder attrs from func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):

        func = self.func

        if getattr(self, '_unique_id', False):
            uid = _unique_id(func, *args)
            self.odict['uid'] = uid
            func = functools.partial(self.func, uid)

        self.odict.update(func(*args, **kwargs))
        if not isinstance(self.odict['code'], list):
            raise TypeError("'code' should be a list of strings.")

        return self.odict.copy()


class Ions(CfgObject):
    # need to handle this in the class namespace
    # I'm not sure if this is necessary for all versions of lammps
    # If not I could handle the ions uid just like any other fix
    _ids = set()

    def __call__(self, *args, **kwargs):
        self.odict = super().__call__(*args, **kwargs)

        # if function, charge, mass and rigid are the same it's probably the
        # same ions definition. Don't increment the set count.
        charge, mass = self.odict['charge'], self.odict['mass']
        rigid = self.odict.get('rigid', False)

        uid = _unique_id(self.func, charge, mass, rigid)
        Ions._ids.add(uid)

        self.odict['uid'] = len(Ions._ids)

        return self.odict.copy()


class Variable(CfgObject):

    def __call__(self, *args, **kwargs):
        # vtype can only be 'fix' or 'var'
        # var type variables are easy to add with custom code

        # be nice and only do the check if 'variables' is found in the args
        # otherwise it will pass anyway since the empty set is a subset of
        # any set
        vs = kwargs.get('variables', [])
        allowed = {'id', 'x', 'y', 'z', 'vx', 'vy', 'vz'}
        if not set(vs).issubset(allowed):
            prefix = [item.startswith('v_') for item in vs]
            if not all(prefix):
                raise TypeError(
                    f'Use only {allowed} as variables or previously defined '
                    "variables with the prefix 'v_'.")

        self.odict = super().__call__(*args, **kwargs)

        prefix = {'fix': 'f_', 'var': 'v_'}
        vtype = self.odict['vtype']
        name = self.odict['uid']
        output = ' '.join([f'{prefix[vtype]}{name}[{i+1}]'
                           for i in range(len(vs))])
        self.odict.update({'output': output})

        return self.odict.copy()


class lammps:

    @validate_id
    def fix(func):
        return CfgObject(func, 'fix')

    def command(func):
        return CfgObject(func, 'command')

    # def group(func):
    #     return CfgObject(func, 'group')

    def variable(vtype):
        @validate_id
        # @validate_vars
        def decorator(func):
            return Variable(func, 'variable',
                            required=['output', ('vtype', vtype)])
        return decorator

    def ions(func):
        return Ions(func, 'ions',
                    required=['charge', 'mass', 'positions'])
