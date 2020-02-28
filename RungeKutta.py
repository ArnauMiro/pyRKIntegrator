#!/bin/env python
#
#
#

import os,platform
import numpy as np, ctypes as ct

## LIBRARY IMPORT PER PLATFORM ##

if platform.system() == "Linux":   lpath = './RK.so'
if platform.system() == "Darwin":  lpath = './RK.dylib'
if platform.system() == "Windows": lpath = './RK.dll'

rklib = ct.cdll.LoadLibrary(lpath)

## DEFINITIONS ##

c_char     = ct.c_char
c_int      = ct.c_int
c_int_p    = ct.POINTER(ct.c_int)
c_double   = ct.c_double
c_double_p = ct.POINTER(ct.c_double)

odefun_f   = ct.CFUNCTYPE(None, c_double, c_double_p, c_int, c_double_p)
event_f    = ct.CFUNCTYPE(c_int, c_double, c_double_p, c_int, c_double_p, c_int_p)
output_f   = ct.CFUNCTYPE(c_int, c_double, c_double_p, c_int)

## DATA STRUCTURES ##

class RK_PARAM(ct.Structure):
	'''
	This data structure contains the parameters for the
	integrator.
	'''
	_fields_ = [
		('scheme',c_char*256),
		('h0',c_double),
		('eps',c_double),
		('epsevf',c_double),
		('minstep',c_double),
		('eventfcn',event_f),
		('outputfcn',output_f)
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
	def set_h(self,xspan):
		'''
		Initial and max step based on xspan
		'''
		self.h0 = (xspan[1] - xspan[0])/10.;
	def set_scheme(self,scheme):
		'''
		Scheme aware function that will set the proper parameters for
		each parameter.
		'''
		# Check if the scheme is valid
		if not scheme in self.schemes: 
			raise ValueError('Scheme %s not valid!' % scheme)
		# If the scheme is valid, set it
		self.scheme = scheme.lower()

class RK_OUT(ct.Structure):
	'''
	This data structure contains the outputs of the integrator.
	'''
	_fields_ = [
		('retval',c_int),
		('n',c_int),
		('err',c_double),
		('x',c_double_p),
		('y',c_double_p),
		('dy',c_double_p)
	]

class odeset(RK_PARAM):
	'''
	ODESET class

	Sets the parameters for odeRK. They are:
		> h0:     Initial step for the interval
		> eps:    Tolerance to meet (Relative).
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
		> n: Number of OMP threads to run in parallel
		> eventfcn: Event function.
		> outputfcn: Output function.
	'''
	def __init__(self,h0=.01,eps=1.e-8,scheme="DormandPrince45",
		epsevf=1.e-4,minstep=1.e-12,eventfcn=None,outputfcn=None):
		'''
		Class constructor
		'''
		# Available schemes
		self.schemes = ['EulerHeun12',
				        'BogackiShampine23',
				        'DormandPrince34A',
				        'Fehlberg45',
				        'CashKarp45',
				        'DormandPrince45',
				        'DormandPrince45A',
				        'Calvo56',
				        'DormandPrince78',
				        'Curtis810',
				        'Hiroshi912'
				       ]
		# odeRK variables
		self.h0        = h0
		self.eps       = eps
		self.epsevf    = epsevf
		self.minstep   = minstep
		self.eventfcn  = event_f(self.eventf  if eventfcn  == None else eventfcn)
		self.outputfcn = output_f(self.outputf if outputfcn == None else outputfcn)
		self.set_scheme(scheme)
	def __del__(self):
		'''
		Class destructor
		'''
		pass


class odesetN(RK_PARAM):
	'''
	ODESET class

	Sets the parameters for odeRKN. They are:
		> h0:     Initial step for the interval
		> eps:    Tolerance to meet (Relative).
		> scheme: Runge-Kutta scheme to use. Options are:
			* RKN 6(8)   (rkn68)
			* RKN 10(12) (rkn1012)
		> n: Number of OMP threads to run in parallel
		> eventfcn: Event function.
		> outputfcn: Output function.
	'''
	def __init__(self,h0=.01,eps=1.e-8,scheme="RKN68",
		epsevf=1.e-4,minstep=1.e-12,eventfcn=None,outputfcn=None):
		'''
		Class constructor
		'''
		# Available schemes
		self.schemes = ['RKN34',
						'RKN46',
						'RKN68',
				        'RKN1012'
				       ]
		# odeRK variables
		self.h0        = h0
		self.eps       = eps
		self.epsevf    = epsevf
		self.minstep   = minstep
		self.eventfcn  = event_f(self.eventf  if eventfcn  == None else eventfcn)
		self.outputfcn = output_f(self.outputf if outputfcn == None else outputfcn)
		self.set_scheme(scheme)
	def __del__(self):
		'''
		Class destructor
		'''
		pass

## FUNCTIONS ##

def odeRK(odefun,xspan,y0,odeset=odeset()):
	'''
	RUNGE-KUTTA Integration

	Numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERK, a generic Runge-Kutta variable step integrator.

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
			> eventfcn: Event function.
			> outputfcn: Output function.

	The function will return:
		> x: solution x values of size n.
		> y: solution y values of size n per each variable.
		> err: Maximum error achieved.
	'''
	n = len(y0)
	# Prepare function
	odeRK = rklib.odeRK
	odeRK.argtypes = [odefun_f,c_double_p,c_double_p,c_int,RK_PARAM]
	odeRK.restype  = RK_OUT
	# Run function
	rko = odeRK(odefun_f(odefun),
				xspan.ctypes.data_as(c_double_p),
				y0.ctypes.data_as(c_double_p),
				ct.c_int(n),
				odeset)
	# Check error
	if (rko.retval == 0): 
		raise ValueError('ERROR! Something went wrong during the integration.')
	if (rko.retval == -1): 
		raise ValueError('ERROR! Integration step required under minimum.')
	# Set output
	x = np.zeros((rko.n+1,),dtype=np.double)
	y = np.zeros((rko.n+1,n),dtype=np.double)
	for ii in xrange(rko.n+1):
		x[ii] = rko.x[ii]
		for jj in xrange(n):
			y[ii,jj] = rko.y[n*ii+jj]
	# Free memory allocated in C
	rklib.freerkout(rko)
	# Return
	return x,y,rko.err

def odeRKN(odefun,xspan,y0,dy0,odeset=odesetN()):
	'''
	RUNGE-KUTTA-NYSTROM Integration

	Numerical integration using Runge-Kutta-Nystrom methods, implemented
	based on MATLAB's ode45.

	The function to call is ODERKN, a generic Runge-Kutta-Nystrom variable step integrator.

	The inputs of this function are:
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
			> scheme: Runge-Kutta-Nystrom scheme to use. Options are:
				* RKN 6(8)   (rkn68)
				* RKN 10(12) (rkn1012)
			> eventfcn: Event function.
			> outputfcn: Output function.

	The function will return:
		> x:  solution x values of size n.
		> y:  solution y values of size n per each variable.
		> dy: solution dy values of size n per each variable.
		> err: Maximum error achieved.
	'''
	n = len(y0)
	# Prepare function
	odeRKN = rklib.odeRKN
	odeRKN.argtypes = [odefun_f,c_double_p,c_double_p,c_double_p,c_int,RK_PARAM]
	odeRKN.restype  = RK_OUT
	# Run function
	rko = odeRKN(odefun_f(odefun),
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
	x  = np.zeros((rko.n+1,),dtype=np.double)
	y  = np.zeros((rko.n+1,n),dtype=np.double)
	dy = np.zeros((rko.n+1,n),dtype=np.double)
	for ii in xrange(rko.n+1):
		x[ii] = rko.x[ii]
		for jj in xrange(n):
			y[ii,jj]  = rko.y[n*ii+jj]
			dy[ii,jj] = rko.dy[n*ii+jj]
	# Free memory allocated in C
	rklib.freerkout(rko)
	# Return
	return x,y,dy,rko.err

def CheckTableau(odeset):
	'''
	Check the sanity of the Butcher's Tableau of a specified
	Runge-Kutta scheme in ODESET
	'''
	# Run function
	retval = rklib.check_tableau(odeset)
	# Return
	return True if retval else False


## MAIN ##

if __name__ == '__main__':
	'''
	Small example how to use the code.
	'''
	import matplotlib
	matplotlib.use("TkAgg")
	import matplotlib.pyplot as plt

	def testfun(x,y,n,dydx):
		dydx[0] = np.cos(x) + np.sin(y[0])

	# Set span and initial solution
	xspan = np.array([0., 10.],np.double)
	y0    = np.array([0.],np.double)

	# Launch integrator
	x,y,err = odeRK(testfun,xspan,y0)

	# Plot
	plt.figure(num=1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')
	plt.plot(x,y[:,0])
	plt.show()