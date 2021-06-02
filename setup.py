#!/bin/env python
#
# RUNGE-KUTTA Integration
#
# Setup and cythonize code.
#export C_INCLUDE_PATH=$C_INCLUDE_PATH:/apps/python/python-3.8.5/lib/python3.8/site-packages/numpy/core/include
#
# Last rev: 2021
from __future__ import print_function, division

import sys, os, numpy as np

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

_USE_CPP      = True
_USE_COMPILED = False
try:
	_USE_CPP      = True if os.environ['USE_CPP']      == 'ON' else False
	_USE_COMPILED = True if os.environ['USE_COMPILED'] == 'ON' else False
except:
	pass
	
with open('README.md') as f:
	readme = f.read()

## Modules
# Code in CPP
RK_cpp = Extension('pyRKIntegrator.RungeKutta',
				   sources      = ['pyRKIntegrator/RungeKutta.pyx','pyRKIntegrator/src/cpp/RK.cpp'],
				   language     = 'c++',
				   include_dirs = ['pyRKIntegrator/src/cpp',np.get_include()]
				  )
# Code in C
RK_c   = Extension('pyRKIntegrator.RungeKutta',
				   sources      = ['pyRKIntegrator/RungeKutta.pyx','pyRKIntegrator/src/c/RK.c'],
				   language     = 'c',
				   include_dirs = ['pyRKIntegrator/src/c',np.get_include()]
				  )

modules_list = [RK_cpp if _USE_CPP else RK_c] if _USE_COMPILED else []

# Main setup
setup(
	name="pyRKIntegrator",
	version="2.0.0",
	ext_modules=cythonize(modules_list,
		language_level = str(sys.version_info[0]), # This is to specify python 3 synthax
		annotate=True         # This is to generate a report on the conversion to C code
	),
    long_description=readme,
    url='https://github.com/ArnauMiro/MEP.git',
    packages=find_packages(exclude=('Examples', 'doc')),
	install_requires=['numpy','matplotlib','cython']
)