import sys
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy


# install wexpect if windows
if 'win32' in sys.platform:
    expect = ['wexpect']
else:
    expect = ['pexpect>=4.2.1']


with open('readme.md') as readme_file:
    readme = readme_file.read()


requirements = [
    'numpy>=1.23',
    'cython>=3.0',
    'jinja2>=2.9.6',
    'scipy>=1.12',
    'h5py>=2.7.0',
    'termcolor>=1.1.0',
    'scikit-optimize>=0.10.2',
    'matplotlib>=3.0',
    'gdspy>=1.6',
    'shapely>=2.0',
    'tqdm>=4.66',
    'nose>=1.0',
    'sphinx>=8.0',
    'numpydoc>=1.0',
    'cvxopt>=1',
    # 'mayavi>=4',
] + expect


short_description = (
    "Python package for simulation and analysis of ion crystals in surface traps.")

setup(
    name='surface-ion',
    version='1.1.0',
    description=short_description,
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Artem Podlesnyy",
    author_email='a.podlesnyy@rqc.ru',
    url='https://github.com/underforestaaa/Sion',
    packages=find_packages(include=['sion', "sion.electrode", "sion.pylion"]),
    package_data={'sion.pylion': ['templates/*.j2']},
    install_requires=requirements,
    license="GPLv3+",
    keywords=['surface trap', 'ion', 'quantum computing', 'ion simulation',
              'normal modes', 'mathieu modes', 'ion shuttling', 'voltage optimization'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv3+',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='tests',
    ext_modules=[
            Extension("electrode._transformations",
                sources=["sion/electrode/transformations.c"],
                include_dirs=[numpy.get_include()]),
            Extension("electrode.cexpressions",
                sources=["sion/electrode/cexpressions.pyx",
                        #"electrode/cexpressions.c",
                        ],
                extra_compile_args=[
                    "-ffast-math", # improves expressions
                    #"-Wa,-adhlns=cexprssions.lst", # for amusement
                    ],
                include_dirs=[numpy.get_include()]),
        ],
    cmdclass = {"build_ext": build_ext},

)
