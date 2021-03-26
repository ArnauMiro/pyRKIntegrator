#!/bin/env python
#
# RUNGE-KUTTA Integration
#
# Setup and cythonize code.
#export C_INCLUDE_PATH=$C_INCLUDE_PATH:/apps/python/python-3.8.5/lib/python3.8/site-packages/numpy/core/include
#
# Last rev: 2021
from __future__ import print_function, division

import os, numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize


## Modules
# Code in CPP
RK_cpp = Extension('RungeKutta',
				   sources      = ['RungeKutta.pyx','src/cpp/RK.cpp'],
				   language     = 'c++',
				   include_dirs = ['./src/cpp',np.get_include()]
				  )
# Code in C
RK_c   = Extension('RungeKutta',
				   sources      = ['RungeKutta.pyx','src/c/RK.c'],
				   language     = 'c',
				   include_dirs = ['./src/c',np.get_include()]
				  )

# Main setup
setup(
	name="pyRungeKutta",
	ext_modules=cythonize([
			RK_cpp if os.environ['USE_CPP'] == 'ON' else RK_c
		],
		language_level = "3", # This is to specify python 3 synthax
		annotate=True         # This is to generate a report on the conversion to C code
	)
)