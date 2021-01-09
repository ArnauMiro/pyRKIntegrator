#!/bin/env cython
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

	Arnau Miro, Elena Terzic 2021
	Last rev: 2021
'''
#from __future__ import print_function, division

import numpy as np, ctypes as ct

cimport numpy as np
cimport cython
from libc.string cimport memcpy

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

# Expose functions from the RK cpp header
cdef extern from "RK.h":
	# Typedef for RK parameters
	ctypedef struct RK_PARAM:
		double h0
		double eps
		double epsevf
		double minstep
		double secfact
		double secfact_max
		double secfact_min
		int    (*eventfcn)(double,double*,int,double*,int*) # Event function must return continue or stop
		int    (*outputfcn)(double,double*,int)             # Output function must return continue or stop
	# Typedef for RK output
	ctypedef struct RK_OUT:
		int    retval
		int    n
		double err
		double *x
		double *y
		double *dy
	# Expose the Runge-Kutta basic integrator
	cdef RK_OUT c_odeRK "odeRK" (const char *scheme, void (*odefun)(double,double*,int,double*),
		double xspan[2], double y0[], const int n, const RK_PARAM *rkp)
	# Expose the free rkout
	cdef void freerkout(const RK_OUT *rko)
 	# Expose the check tableau
	cdef int check_tableau(const char *scheme)


ctypedef void (*odefun)(double,double*,int,double*)
odefun_f = ct.CFUNCTYPE(None, ct.c_double, ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ct.c_double))
ctypedef int  (*eventfcn)(double,double*,int,double*,int*)
ctypedef int  (*outputfcn)(double,double*,int) 


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef double[:] double1D_to_numpy(double *cdouble,int n):
	'''
	Convert a 1D C double pointer into a numpy array
	'''
	cdef int ii
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),np.double)
	memcpy(&out[0],cdouble,n*sizeof(double))
	return out


cdef class rkout:
	'''
	RKOUT class

	Wrapper to manage the output of ODERK.
	'''
	cdef RK_OUT rko

	def __cinit__(self):
		pass

	def __dealloc__(self):
		freerkout(&self.rko)

	# We can expose every variable here so it can be accessed by python.
	# These operations are slow and costly to do so it is wise just 
	# to perform them once.
	@property
	def err(self):
		return self.rko.err
	@property
	def x(self):
		return double1D_to_numpy(rko.x,rko.n)


cdef class odeset:
	'''
	ODESET class

	Sets the parameters for odeRK. They are:
		> h0:        Initial step for the interval
		> eps:       Tolerance to meet (Relative).
		> eventfcn:  Event function.
		> outputfcn: Output function.
	'''
	cdef RK_PARAM c_rkp 

	def __cinit__(self,double h0=.01,double eps=1.e-8, double epsevf=1.e-4, double minstep=1.e-12,\
				  double secfact=0.9,double secfact_max=5.,double secfact_min=0.2,\
				  object eventfcn=None,object outputfcn=None):
		'''
		Class constructor
		'''
		# odeRK variables
		self.c_rkp.h0          = h0
		self.c_rkp.eps         = eps
		self.c_rkp.epsevf      = epsevf
		self.c_rkp.minstep     = minstep
		self.c_rkp.secfact     = secfact
		self.c_rkp.secfact_max = secfact_max
		self.c_rkp.secfact_min = secfact_min
		self.c_rkp.eventfcn    = NULL #if eventfcn  == None else eventfcn
		self.c_rkp.outputfcn   = NULL #if outputfcn == None else outputfcn

	# We can expose every variable here so it can be accessed by python.
	# These operations are slow and costly to do so it is wise just 
	# to perform them once.
	@property
	def h0(self):
		return self.c_rkp.h0
	@h0.setter
	def h0(self, double h0):
		self.c_rkp.h0 = h0
	@property
	def eps(self):
		return self.c_rkp.eps
	@eps.setter
	def eps(self, double eps):
		self.c_rkp.eps = eps
	@property
	def epsevf(self):
		return self.c_rkp.epsevf
	@epsevf.setter
	def epsevf(self, double epsevf):
		self.c_rkp.epsevf = epsevf
	@property
	def minstep(self):
		return self.c_rkp.minstep
	@minstep.setter
	def minstep(self, double minstep):
		self.c_rkp.minstep = minstep
	@property
	def secfact(self):
		return self.c_rkp.secfact
	@secfact.setter
	def secfact(self, double secfact):
		self.c_rkp.secfact = secfact
	@property
	def secfact_max(self):
		return self.c_rkp.secfact_max
	@secfact_max.setter
	def secfact_max(self, double secfact_max):
		self.c_rkp.secfact_max = secfact_max
	@property
	def secfact_min(self):
		return self.c_rkp.secfact_min
	@secfact_min.setter
	def secfact_min(self, double secfact_min):
		self.c_rkp.secfact_min = secfact_min		


def odeRK(str scheme,object fun,double[:] xspan,double[:] y0,odeset params=odeset()):
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
	
	cdef int n        = len(y0)
	cdef rkout out    = rkout()
	cdef odefun c_fun = (<odefun *><size_t>ct.addressof(odefun_f(fun)))[0]

	# Run C function
	out.rko = c_odeRK(scheme.encode('utf-8'),c_fun,&xspan[0],&y0[0],n,&params.c_rkp)

	## PARSE the output of RKO ##

#	# Check error
#	if (rko.retval == 0): 
#		raise ValueError('ERROR! Something went wrong during the integration.')
#	if (rko.retval == -1): 
#		raise ValueError('ERROR! Integration step required under minimum.')
#	# Set output
#	x   = np.fromiter(rko.x,dtype=np.double,count=rko.n+1)
#	y   = np.fromiter(rko.y,dtype=np.double,count=n*(rko.n+1)).reshape((rko.n+1,n))
#	err = rko.err
#	# Return
#	return x,y,err



# Convert to a function that python can see
def CheckTableau(str scheme):
	'''
	Check the sanity of the Butcher's Tableau 
	of a specified Runge-Kutta scheme.
	'''
	if not scheme.lower() in RK_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)
	return True if check_tableau(scheme.encode('utf-8')) else False
