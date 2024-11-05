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
    'numpy>=1.13.3',
    'jinja2>=2.9.6',
    'scipy>=1.12.1',
    'scikit-optimize>=0.10.2',
    'electrode>=1.4',
    'matplotlib>=3.9.2',
    'gdspy>=1.6.13',
    'shapely>=2.0.6',
    'tqdm>=4.66.6'
] + expect


short_description = (
    "Python package for simulation and analysis of ion crystals in surface traps.")

setup(
    name='surface-ion',
    version='1.0.3',
    description=short_description,
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Artem Podlesnyy",
    author_email='a.podlesnyy@rqc.ru',
    url='https://github.com/underforestaaa/Sion',
    packages=find_packages(include=['surface-ion']),
    package_data={'surface-ion': ['templates/*j2']},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    keywords=['surface trap', 'ion', 'quantum computing', 'ion simulation',
              'normal modes', 'mathieu modes', 'ion shuttling', 'voltage optimization'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
)
