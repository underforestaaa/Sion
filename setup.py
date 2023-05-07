import sys
from setuptools import setup, find_packages

# install wexpect if windows
if 'win32' in sys.platform:
    expect = ['wexpect']
else:
    expect = ['pexpect>=4.2.1']


with open('readme.md') as readme_file:
    readme = readme_file.read()


requirements = [
    'h5py>=2.7.0',
    'termcolor>=1.1.0',
    'numpy>=1.13.1',
    'jinja2>=2.9.6',
] + expect

setup_requirements = [
    # TODO put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

short_description = (
    "A LAMMPS wrapper for molecular dynamics simulations of trapped ions in planar traps.")

setup(
    name='sion',
    version='0.5.0',
    description=short_description,
    long_description=readme,
    author="Artem Podlesnyy",
    author_email='podlesnyi.as@phystech.edu',
    packages=find_packages(include=['sion']),
    package_data={'sion': ['templates/*j2']},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='sion',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
