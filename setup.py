#!/usr/bin/env python
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


## Read INIT file
with open('pyRKIntegrator/__init__.py') as f:
	for l in f.readlines():
		if '__version__' in l:
			__version__ = eval(l.split('=')[1].strip())


with open('README.md') as f:
	readme = f.read()


## Read compilation options
options = {}
with open('options.cfg') as f:
	for line in f.readlines():
		if '#' in line or len(line) == 1: continue # Skip comment
		linep = line.split('=')
		options[linep[0].strip()] = linep[1].strip()
		if options[linep[0].strip()] == 'ON':  options[linep[0].strip()] = True
		if options[linep[0].strip()] == 'OFF': options[linep[0].strip()] = False


## Set up compiler options and flags
CC  = 'gcc'      if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'icc'
CXX = 'g++'      if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'icpc'
FC  = 'gfortran' if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'ifort'

CFLAGS   = ''
CXXFLAGS = ' -std=c++11'
FFLAGS   = ''
DFLAGS   = ' -DNPY_NO_DEPRECATED_API'
if CC == 'gcc':
	# Using GCC as a compiler
	CFLAGS   += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	CXXFLAGS += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	FFLAGS   += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	# Vectorization flags
	if options['VECTORIZATION']:
		CFLAGS   += ' -march=native -ftree-vectorize'
		CXXFLAGS += ' -march=native -ftree-vectorize'
		FFLAGS   += ' -march=native -ftree-vectorize'
	# OpenMP flag
	if options['OPENMP_PARALL']:
		CFLAGS   += ' -fopenmp'
		CXXFLAGS += ' -fopenmp'
else:
	# Using GCC as a compiler
	CFLAGS   += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	CXXFLAGS += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	FFLAGS   += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	# Vectorization flags
	if options['VECTORIZATION']:
		CFLAGS   += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
		CXXFLAGS += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
		FFLAGS   += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
	# OpenMP flag
	if options['OPENMP_PARALL']:
		CFLAGS   += ' -qopenmp'
		CXXFLAGS += ' -qopenmp'


## Set up environment variables
os.environ['CC']       = CC
os.environ['CXX']      = CXX
os.environ['CFLAGS']   = CFLAGS + DFLAGS
os.environ['CXXFLAGS'] = CXXFLAGS + DFLAGS
os.environ['LDSHARED'] = CC + ' -shared'


## Libraries and includes
libraries     = ['m']

# OSX needs to also link with python3.8 for reasons...
if sys.platform == 'darwin': libraries += [f'python{sys.version_info[0]}.{sys.version_info[1]}']


## Modules
# Code in CPP
RK_cpp = Extension('pyRKIntegrator.RungeKutta',
				   sources      = ['pyRKIntegrator/RungeKutta.pyx','pyRKIntegrator/src/cpp/RK.cpp'],
				   language     = 'c++',
				   include_dirs = ['pyRKIntegrator/src/cpp',np.get_include()],
				   libraries    = libraries
				  )
# Code in C
RK_c   = Extension('pyRKIntegrator.RungeKutta',
				   sources      = ['pyRKIntegrator/RungeKutta.pyx','pyRKIntegrator/src/c/RK.c'],
				   language     = 'c',
				   include_dirs = ['pyRKIntegrator/src/c',np.get_include()],
				   libraries    = libraries
				  )

modules_list = [RK_cpp if options['USE_CPP'] else RK_c] if options['USE_COMPILED'] else []


## Main setup
setup(
	name               = 'pyRKIntegrator',
	version            = __version__,
	author             = ['Arnau Miro','Manel Soria'],
	author_email       = ['arnau.miro@upc.edu','manel.soria@upc.edu'],
	maintainer         = 'Arnau Miro',
	maintainer_email   = 'arnau.miro@upc.edu',
	ext_modules=cythonize(modules_list,
		language_level = str(sys.version_info[0]), # This is to specify python 3 synthax
		annotate       = False                     # This is to generate a report on the conversion to C code
	),
    long_description   = readme,
    url                = 'https://github.com/ArnauMiro/pyRKIntegrator.git',
    packages           = find_packages(exclude=('Examples','doc')),
	install_requires   = ['numpy','matplotlib','cython']
)