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

import numpy as np
from . import RK_SCHEMES, RKN_SCHEMES

cimport numpy as np
cimport cython
from libc.string cimport memcpy


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
	# Expose the Runge-Kutta-Nystrom integrator
	cdef RK_OUT c_odeRKN "odeRKN"(const char *scheme, void (*odefun)(double,double*,int,double*),
		double xspan[2], double y0[], double dy0[], const int n, const RK_PARAM *rkp);
	# Expose the free rkout
	cdef void freerkout(const RK_OUT *rko)
 	# Expose the check tableau
	cdef int check_tableau(const char *scheme)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=1] double1D_to_numpy(double *cdouble,int n):
	'''
	Convert a 1D C double pointer into a numpy array
	'''
	cdef np.ndarray[np.double_t,ndim=1] out = np.zeros((n,),np.double)
	memcpy(&out[0],cdouble,n*sizeof(double))
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.int_t,ndim=1] int1D_to_numpy(int *cint,int n):
	'''
	Convert a 1D C integer pointer into a numpy array
	'''
	cdef np.ndarray[np.int_t,ndim=1] out = np.zeros((n,),np.int)
	memcpy(&out[0],cint,n*sizeof(int))
	return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef np.ndarray[np.double_t,ndim=2] double2D_to_numpy(double *cdouble, int n, int m):
	'''
	Convert a 2D C double pointer into a numpy array
	'''
	cdef np.ndarray[np.double_t,ndim=2] out = np.zeros((n,m),np.double)
	memcpy(&out[0,0],cdouble,n*m*sizeof(double))
	return out

cdef void odefun_wrapper(double t,double* y,int n,double* dydx):
	global wrap_odefun
	cdef np.ndarray[np.double_t,ndim=1] _y    = double1D_to_numpy(y,n)
	cdef np.ndarray[np.double_t,ndim=1] _dydx = double1D_to_numpy(dydx,n)
	wrap_odefun(t,_y,n,_dydx)
	memcpy(dydx,&_dydx[0],n*sizeof(double))

cdef int eventfun_wrapper(double t,double* y,int n,double* value,int* direction):
	global wrap_eventfun
	cdef np.ndarray[np.double_t,ndim=1] _y      = double1D_to_numpy(y,n)
	cdef np.ndarray[np.double_t,ndim=1] _value  = double1D_to_numpy(value,1)
	cdef np.ndarray[np.int_t,ndim=1] _direction = int1D_to_numpy(direction,1)
	cdef int retval = wrap_eventfun(t,_y,n,_value,_direction)
	value[0]     = _value[0]
	direction[0] = _direction[0]
	return retval

cdef int outputfun_wrapper(double t,double *y,int n):
	global wrap_outputfun
	cdef np.ndarray[np.double_t,ndim=1] _y = double1D_to_numpy(y,n)
	return wrap_outputfun(t,_y,n)


cdef class rkout:
	'''
	RKOUT class

	Wrapper to manage the output of ODERK.
	'''
	cdef RK_OUT rko
	cdef int nvariables 

	def __cinit__(self,nvariables):
		self.nvariables = nvariables

	def __dealloc__(self):
		freerkout(&self.rko)

	# We can expose every variable here so it can be accessed by python.
	# These operations are slow and costly to do so it is wise just 
	# to perform them once.
	@property
	def retval(self):
		return self.rko.retval
	@property
	def err(self):
		return self.rko.err
	@property
	def x(self):
		return double1D_to_numpy(self.rko.x,self.rko.n+1)
	@property
	def y(self):
		return double2D_to_numpy(self.rko.y,self.rko.n+1,self.nvariables)
	@property
	def dy(self):
		return double2D_to_numpy(self.rko.dy,self.rko.n+1,self.nvariables)


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
				  object eventfun=None,object outputfun=None):
		'''
		Class constructor
		'''
		global wrap_outputfun, wrap_eventfun
		# odeRK variables
		self.c_rkp.h0          = h0
		self.c_rkp.eps         = eps
		self.c_rkp.epsevf      = epsevf
		self.c_rkp.minstep     = minstep
		self.c_rkp.secfact     = secfact
		self.c_rkp.secfact_max = secfact_max
		self.c_rkp.secfact_min = secfact_min
		self.c_rkp.eventfcn     = NULL
		self.c_rkp.outputfcn    = NULL

		if not eventfun == None:
			wrap_eventfun       = eventfun
			self.c_rkp.eventfcn = &eventfun_wrapper
		if not outputfun == None:
			wrap_outputfun       = outputfun
			self.c_rkp.outputfcn = &outputfun_wrapper 

	@cython.boundscheck(False) # turn off bounds-checking for entire function
	@cython.wraparound(False)  # turn off negative index wrapping for entire function
	@cython.nonecheck(False)
	def set_h(self,double[:] xspan,double div=10.):
		'''
		Initial and max step based on xspan
		'''
		self.c_rkp.h0 = (xspan[1] - xspan[0])/div

	def __str__(self):
		'''
		Print parameters in a readable format
		'''
		return 'Odeset structure:\n' + \
			   'h0:                  %6.4f\n' % self.c_rkp.h0             + \
			   'epsilon:             %.2e\n'  % self.c_rkp.eps            + \
			   'eps eventfun:        %.2e\n'  % self.c_rkp.epsevf         + \
			   'minimum step:        %6.4f\n' % self.c_rkp.minstep        + \
			   'security factor:     %6.4f\n' % self.c_rkp.secfact        + \
			   'max security factor: %6.4f\n' % self.c_rkp.secfact_max    + \
			   'min security factor: %6.4f\n' % self.c_rkp.secfact_min

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


def odeRK(object scheme,object fun,double[:] xspan,double[:] y0,odeset params=odeset()):
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
		> fun: a function so that
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
	global wrap_odefun
	if not scheme.lower() in RK_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)

	cdef int n     = len(y0)
	cdef rkout out = rkout(nvariables=n)
	wrap_odefun    = fun

	# Run C function
	out.rko = c_odeRK(scheme.encode('utf-8'),&odefun_wrapper,&xspan[0],&y0[0],n,&params.c_rkp)
	cdef int retval = out.retval
	if retval == 0:
		raise ValueError('ERROR! Something went wrong during the integration.')
	if retval == -1:
		raise ValueError('ERROR! Integration step required under minimum.')

	return out.x,out.y,out.err

def ode23(object fun,double[:] xspan,double[:] y0,odeset params=odeset()):
	'''
	ODE23

	Numerical integration using Runge-Kutta methods, implemented
	based on MATLAB's ode23.

	The inputs of this function are:
		> fun: a function so that
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
	return odeRK("bogackishampine23",fun,xspan,y0,params)

def ode45(object fun,double[:] xspan,double[:] y0,odeset params=odeset()):
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
	return odeRK("dormandprince45",fun,xspan,y0,params)


def odeRKN(object scheme,object fun,double[:] xspan,double[:] y0,double[:] dy0,odeset params=odeset()):
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
	global wrap_odefun
	if not scheme.lower() in RKN_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)
	
	cdef int n     = len(y0)
	cdef rkout out = rkout(nvariables=n)
	wrap_odefun    = fun 

	# Run C function
	out.rko = c_odeRKN(scheme.encode('utf-8'),&odefun_wrapper,&xspan[0],&y0[0],&dy0[0],n,&params.c_rkp)
	cdef int retval = out.retval
	if retval == 0:
		raise ValueError('ERROR! Something went wrong during the integration.')
	if retval == -1:
		raise ValueError('ERROR! Integration step required under minimum.')

	# Set output
	cdef np.ndarray[np.double_t,ndim=1] x  = out.x
	cdef np.ndarray[np.double_t,ndim=2] y  = out.y
	cdef np.ndarray[np.double_t,ndim=2] dy = out.dy
	cdef double                       err = out.err

	# Return
	return x,y,dy,err


# Convert to a function that python can see
def CheckTableau(object scheme):
	'''
	Check the sanity of the Butcher's Tableau 
	of a specified Runge-Kutta scheme.
	'''
	if not scheme.lower() in RK_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)
	return True if check_tableau(scheme.encode('utf-8')) else False