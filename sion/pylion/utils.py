import inspect
import os
import sys
import warnings
import functools
import h5py
from termcolor import colored


def pretty_repr(_cls):
    def _repr(self):
        title = colored('signature', 'red')
        lines = f'\n{title}: {self.__name__}{inspect.signature(_cls)}\n'
        title = colored('docstring', 'red')
        lines += f'{title}: {inspect.getdoc(self)}\n'
        title = colored('file', 'red')
        lines += f'{title}:      {os.path.abspath(self.__module__)}\n'
        title = colored('type', 'red')
        lines += f'{title}:      {type(self)}\n'
        return lines
    _cls.__repr__ = _repr
    return _cls


def validate_id(func):
    @functools.wraps(func)
    def wrapper(*args):
        _func = args[0]
        # check that the leftmost argument of the pass function is 'uid'
        for param in inspect.signature(_func).parameters:
            if param == 'uid':
                break
            else:
                raise TypeError("First argument needs to be 'uid'.")

        cfg = func(*args)
        cfg._unique_id = True
        return cfg
    return wrapper


def _unique_id(*args):
    # extract 2 least significant bytes. that should be enough to make sure ids
    # are unique and it's sensitive to small changes in the input arguments
    uid = sum([id(arg) for arg in args])
    # uid &= 0xFFF
    return uid


def save_atttributes_and_files(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

        # this decorator is only for execute() so that first argument is self
        self = args[0]
        attrs = self.attrs

        # save everything at the end so if the simulation fails the h5file is
        # not even created

        # initalise the h5 file
        with h5py.File(attrs['name'] + '.h5', 'w') as f:
            pass

        # save attrs and scripts to h5 file
        attrs.save(attrs['name'] + '.h5')
        _savecallersource(attrs['name'] + '.h5')

        for filename in (attrs['output_files']
                         + ['log.lammps', attrs['name'] + '.lammps']):
            _savescriptsource(attrs['name'] + '.h5', filename)

    return wrapper


def _savescriptsource(h5file, script):
    with h5py.File(h5file, 'a') as f:
        with open(script, 'rb') as pf:
            lines = pf.readlines()
            f.create_dataset(script, data=lines)


def _savecallersource(h5file):
    # inspect the first four frames of the stack to find the correct
    # filename. This covers calling from execute() or _writeinputfile().
    # if the stack is indeed larger than this it's probably the REPL.
    stack = inspect.stack()[:5]
    for frame in stack:
        if sys.argv[0] == frame.filename:
            _savescriptsource(h5file, frame.filename)
            return

    # cannot save on the h5 file if using the repl
    warnings.warn(
        'Caller source not saved. '
        'Are you running the simulation from the REPL?')

# def validate_vars(func):
#     @functools.wraps(func)
#     def wrapper(*args):
#         f = args[0]
#         if 'variables' not in inspect.getfullargspec(f).args:
#             raise TypeError("Could not find 'variables' kwarg.")

#         return func(*args)
#     return wrapper


# def returns(*keys):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args):
#             odict = func(*args)
#             odict_keys = odict.keys()
#             for key in keys:
#                 assert key in odict_keys
#             return odict
#         return wrapper
#     return decorator


# def _unique_id(*args):
#     # this is mainly to deal with negative numbers
#     return hex(hash(args) & (2**64-1))[2:]


# # for ideas for arbitrary base look here
# # https://code.activestate.com/recipes/65212-convert-from-decimal-to-any-base-number/
# def base10toN(num, n):
#     word = "0123456789abcdefghijklmnopqrstuvwxyz"
#     return ((num == 0) and "0") or (base10toN(num // n, n).strip("0") +
#                                     word[:n][num % n])
