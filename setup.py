#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = "0.1.1"

long_description = """anaddb_irreps: use phonopy to find the irreducible representations of the phonon modes from anaddb output. It is a simple wrapper of the Phonopy irreps module."""

setup(
    name='anaddb_irreps',
    version=__version__,
    description='anaddb_irreps: use phonopy to find the irreducible representations of the phonon modes from anaddb output.',
    long_description=long_description,
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='BSD-2-clause',
    packages=find_packages(),
    scripts=[],
    install_requires=['numpy>1.16.5', 'ase>=3.19', 'phonopy>=2.43.0', 'abipy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
    ],
    python_requires='>=3.6',
)
