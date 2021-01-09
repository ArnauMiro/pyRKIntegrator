#!/bin/env python
#
# RUNGE-KUTTA Integration
#
# Setup and cythonize code.
#export C_INCLUDE_PATH=$C_INCLUDE_PATH:/apps/python/python-3.8.5/lib/python3.8/site-packages/numpy/core/include
#
# Last rev: 2021
from __future__ import print_function, division

import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize

# Main setup
setup(
	name="pyRungeKutta",
	ext_modules=cythonize([
			Extension('RungeKutta' ,sources=['RungeKutta.pyx','src/RK.cpp'], language='c++',include_dirs=['./src',np.get_include()]),
		],
		language_level = "3", # This is to specify python 3 synthax
		annotate=True         # This is to generate a report on the conversion to C code
	)
)