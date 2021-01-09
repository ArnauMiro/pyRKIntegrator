#!/bin/env python
'''
	RUNGE-KUTTA Integration

	Library to perform numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERK, a generic Runge-Kutta variable step integrator.

	The inputs of this function are:
		> scheme: Runge-Kutta scheme to use. Options are:
			* Euler-Heun 1(2) (eulerheun12)
			* Bogacki-Shampine 2(3) (bogackishampine23)
			* Fehlberg 4(5) (fehlberg45)
			* Cash-Carp 4(5) (cashcarp45)
			* Dormand-Prince 4-5 (dormandprince45)
			* Calvo 5(6) (calvo56)
			* Dormand-Prince 7(8) (dormandprince78)
			* Curtis 8(10) (curtis810)
			* Hiroshi 9(12) (hiroshi912)
		> odefun: a function so that
			dydx = odefun(x,y,n)
		where n is the number of y variables to integrate.
		> xspan: integration start and end point.
		> y0: initial conditions (must be size n).
		> n: number of initial conditions.

		An additional parameter can be passed to the integrator, which contains
		parameters for the integrator:
			> h0:       Initial step for the interval
			> hmin:      Minimum interval allowed
			> eps:       Tolerance to meet.
			> eventfcn:  Event function.
			> outputfcn: Output function.

	If the inputted tolerance is not met, the intergrator will use hmin and will
	produce a successful step. Warning! Accuracy might be compromised.

	The function will return a structure containing the following information:
		> retval: Return value. Will be negative in case of errors or positive if successful.
			* 1: indicates a successful run.
			* 2: indicates that at a certain point hmin was used for the step.
		> n: Number of steps taken.
		> err: Maximum error achieved.
		> x: solution x values of size n.
		> y: solution y values of size n per each variable.

	Arnau Miro, 2018
	Last rev: 2020
'''
from __future__ import print_function, division

import os, platform
import numpy as np, ctypes as ct

## LIBRARY IMPORT PER PLATFORM ##
if platform.system() == "Linux":   __LIBPATH__ = 'libRK.so'
if platform.system() == "Darwin":  __LIBPATH__ = 'libRK.dylib'
if platform.system() == "Windows": __LIBPATH__ = 'libRK.dll'

__LIBPATH__ = os.path.join(os.path.dirname(os.path.abspath(__file__)),__LIBPATH__)

## DEFINITIONS ##

libRK      = ct.cdll.LoadLibrary(__LIBPATH__)

c_char     = ct.c_char
c_char_p   = ct.POINTER(ct.c_char)
c_int      = ct.c_int
c_int_p    = ct.POINTER(ct.c_int)
c_double   = ct.c_double
c_double_p = ct.POINTER(ct.c_double)

odefun_f   = ct.CFUNCTYPE(None, c_double, c_double_p, c_int, c_double_p)
event_f    = ct.CFUNCTYPE(c_int, c_double, c_double_p, c_int, c_double_p, c_int_p)
output_f   = ct.CFUNCTYPE(c_int, c_double, c_double_p, c_int)

## RUNGE-KUTTA SCHEMES ##

RK_SCHEMES = ['eulerheun12',
			  'bogackishampine23',
			  'dormandprince34a',
			  'fehlberg45',
			  'cashkarp45',
			  'dormandprince45',
			  'dormandprince45a',
			  'calvo56',
			  'dormandprince78',
			  'curtis810',
			  'hiroshi912'
			 ]

## RUNGE-KUTTA-NYSTROM SCHEMES ##
RKN_SCHEMES = ['rkn34',
			   'rkn46',
			   'rkn68',
			   'rkn1012'
			  ]

## RUNGE-KUTTA PARAMETERS ##

class RK_PARAM(ct.Structure):
	'''
	This data structure contains the parameters for the
	integrator.
	'''
	_fields_ = [
		('h0',          c_double),
		('eps',         c_double),
		('epsevf',      c_double),
		('minstep',     c_double),
		('secfact',     c_double),
		('secfact_max', c_double),
		('secfact_min', c_double),
		('eventfcn',    event_f),
		('outputfcn',   output_f)
	]
	@staticmethod
	def eventf(x,y,n,value,direction):
		'''
		This is just a dummy event function meant not to interfere
		with the code.
		'''
		return -10 # Dummy flag
	@staticmethod
	def outputf(x,y,n):
		'''
		This is just a dummy output function meant not to interfere
		with the code.
		'''
		return 1
	def set_h(self,xspan,div=10.):
		'''
		Initial and max step based on xspan
		'''
		self.h0 = (xspan[1] - xspan[0])/div

class RK_OUT(ct.Structure):
	'''
	This data structure contains the outputs of the integrator.
	'''
	_fields_ = [
		('retval', c_int),
		('n',      c_int),
		('err',    c_double),
		('x',      c_double_p),
		('y',      c_double_p),
		('dy',     c_double_p)
	]
	def __del__(self):
		'''
		Deallocate memory from C/C++
		'''
		# Prepare function
		freerkout = libRK.freerkout
		freerkout.argtypes = [ct.POINTER(RK_OUT)]
		freerkout.restype  = None
		# Launch function
		freerkout(self)

class odeset(RK_PARAM):
	'''
	ODESET class

	Sets the parameters for odeRK. They are:
		> h0:     Initial step for the interval
		> eps:    Tolerance to meet (Relative).
		> eventfcn: Event function.
		> outputfcn: Output function.
	'''
	def __init__(self,h0=.01,eps=1.e-8,epsevf=1.e-4,minstep=1.e-12,
		secfact=0.9,secfact_max=5.,secfact_min=0.2,eventfcn=None,outputfcn=None):
		'''
		Class constructor
		'''
		# odeRK variables
		self.h0          = h0
		self.eps         = eps
		self.epsevf      = epsevf
		self.minstep     = minstep
		self.secfact     = secfact
		self.secfact_max = secfact_max
		self.secfact_min = secfact_min
		self.eventfcn    = event_f(self.eventf  if eventfcn  == None else eventfcn)
		self.outputfcn   = output_f(self.outputf if outputfcn == None else outputfcn)
	def __del__(self):
		'''
		Class destructor
		'''
		pass

## RUNGE-KUTTA ODE ##

def odeRK(scheme,odefun,xspan,y0,odeset=odeset()):
	'''
	RUNGE-KUTTA Integration

	Numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERK, a generic Runge-Kutta variable step integrator.

	The inputs of this function are:
		> scheme: Runge-Kutta scheme to use. Options are:
			* Euler-Heun 1(2) (eulerheun12)
			* Bogacki-Shampine 2(3) (bogackishampine23)
			* Fehlberg 4(5) (fehlberg45)
			* Cash-Karp 4(5) (cashkarp45)
			* Dormand-Prince 4-5 (dormandprince45)
			* Calvo 5(6) (calvo56)
			* Dormand-Prince 7(8) (dormandprince78)
			* Curtis 8(10) (curtis810)
			* Hiroshi 9(12) (hiroshi912)
		> odefun: a function so that
			dydx = odefun(x,y,n)
		where n is the number of y variables to integrate.
		> xspan: integration start and end point.
		> y0: initial conditions (must be size n).

		An additional parameter can be passed to the integrator, which contains
		parameters for the integrator:
			> h0:     Initial step for the interval
			> eps:    Tolerance to meet.
			> eventfcn: Event function.
			> outputfcn: Output function.

	The function will return:
		> x: solution x values of size n.
		> y: solution y values of size n per each variable.
		> err: Maximum error achieved.
	'''
	if not scheme.lower() in RK_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)
	n = len(y0)
	# Prepare function
	odeRK = libRK.odeRK
	odeRK.argtypes = [c_char_p,odefun_f,c_double_p,c_double_p,c_int,ct.POINTER(RK_PARAM)]
	odeRK.restype  = RK_OUT
	# Run function
	rko = odeRK(scheme.encode('utf-8'),
				odefun_f(odefun),
				xspan.ctypes.data_as(c_double_p),
				y0.ctypes.data_as(c_double_p),
				ct.c_int(n),
				odeset
			   )
	# Check error
	if (rko.retval == 0): 
		raise ValueError('ERROR! Something went wrong during the integration.')
	if (rko.retval == -1): 
		raise ValueError('ERROR! Integration step required under minimum.')
	# Set output
	x   = np.fromiter(rko.x,dtype=np.double,count=rko.n+1)
	y   = np.fromiter(rko.y,dtype=np.double,count=n*(rko.n+1)).reshape((rko.n+1,n))
	err = rko.err
	# Return
	return x,y,err

def ode23(odefun,xspan,y0,odeset=odeset()):
	'''
	ODE23

	Numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode23.

	The inputs of this function are:
		> odefun: a function so that
			dydx = odefun(x,y,n)
		where n is the number of y variables to integrate.
		> xspan: integration start and end point.
		> y0: initial conditions (must be size n).

		An additional parameter can be passed to the integrator, which contains
		parameters for the integrator:
			> h0:     Initial step for the interval
			> eps:    Tolerance to meet.
			> eventfcn: Event function.
			> outputfcn: Output function.

	The function will return:
		> x: solution x values of size n.
		> y: solution y values of size n per each variable.
		> err: Maximum error achieved.
	'''
	return odeRK("bogackishampine23",odefun,xspan,y0,odeset)

def ode45(odefun,xspan,y0,odeset=odeset()):
	'''
	ODE45

	Numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode45.

	The inputs of this function are:
		> odefun: a function so that
			dydx = odefun(x,y,n)
		where n is the number of y variables to integrate.
		> xspan: integration start and end point.
		> y0: initial conditions (must be size n).

		An additional parameter can be passed to the integrator, which contains
		parameters for the integrator:
			> h0:     Initial step for the interval
			> eps:    Tolerance to meet.
			> eventfcn: Event function.
			> outputfcn: Output function.

	The function will return:
		> x: solution x values of size n.
		> y: solution y values of size n per each variable.
		> err: Maximum error achieved.
	'''
	return odeRK("dormandprince45",odefun,xspan,y0,odeset)


## RUNGE-KUTTA-NYSTROM ODE ##

def odeRKN(scheme,odefun,xspan,y0,dy0,odeset=odeset()):
	'''
	RUNGE-KUTTA-NYSTROM Integration

	Numerical integration using Runge-Kutta-Nystrom methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERKN, a generic Runge-Kutta-Nystrom variable step integrator.

	The inputs of this function are:
		> scheme: Runge-Kutta-Nystrom scheme to use. Options are:
			* RKN 3(4)   (rkn34)
			* RKN 6(8)   (rkn68)
			* RKN 10(12) (rkn1012)
		> odefun: a function so that
			dy2dx = odefun(x,y,n)
		where n is the number of y variables to integrate.
		> xspan: integration start and end point.
		> y0: initial conditions (must be size n).
		> dy0: initial conditions for the fist derivative (must be size n).

		An additional parameter can be passed to the integrator, which contains
		parameters for the integrator:
			> h0:     Initial step for the interval
			> eps:    Tolerance to meet.
			> eventfcn: Event function.
			> outputfcn: Output function.

	The function will return:
		> x:  solution x values of size n.
		> y:  solution y values of size n per each variable.
		> dy: solution dy values of size n per each variable.
		> err: Maximum error achieved.
	'''
	if not scheme.lower() in RKN_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)
	n = len(y0)
	# Prepare function
	odeRKN = libRK.odeRKN
	odeRKN.argtypes = [c_char_p,odefun_f,c_double_p,c_double_p,c_double_p,c_int,ct.POINTER(RK_PARAM)]
	odeRKN.restype  = RK_OUT
	# Run function
	rko = odeRKN(scheme.encode('utf-8'),
				 odefun_f(odefun),
				 xspan.ctypes.data_as(c_double_p),
				 y0.ctypes.data_as(c_double_p),
				 dy0.ctypes.data_as(c_double_p),
				 ct.c_int(n),
				 odeset)
	# Check error
	if (rko.retval == 0): 
		raise ValueError('ERROR! Something went wrong during the integration.')
	if (rko.retval == -1): 
		raise ValueError('ERROR! Integration step required under minimum.')
	# Set output
	x   = np.fromiter(rko.x,dtype=np.double,count=rko.n+1)
	y   = np.fromiter(rko.y,dtype=np.double,count=n*(rko.n+1)).reshape((rko.n+1,n))
	dy  = np.fromiter(rko.dy,dtype=np.double,count=n*(rko.n+1)).reshape((rko.n+1,n))
	err = rko.err
	# Return
	return x,y,dy,err

def CheckTableau(scheme):
	'''
	Check the sanity of the Butcher's Tableau 
	of a specified Runge-Kutta scheme.
	'''
	if not scheme.lower() in RK_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)
	# Prepare function
	check_tableau = libRK.check_tableau
	check_tableau.argtypes = [c_char_p]
	check_tableau.restype  = c_int
	# Run function
	retval = check_tableau(scheme.encode('utf-8'))
	# Return
	return True if retval else False
