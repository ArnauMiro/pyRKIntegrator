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

	Arnau Miro, Elena Terzic 2021
	Last rev: 2021
'''
from __future__ import print_function, division

import numpy as np
from . import RK_SCHEMES, RKN_SCHEMES


class odeset:
	'''
	ODESET class

	Sets the parameters for odeRK. They are:
		> h0:        Initial step for the interval
		> eps:       Tolerance to meet (Relative).
		> eventfcn:  Event function.
		> outputfcn: Output function.
	'''
	def __init__(self,h0=.01,eps=1.e-8,epsevf=1.e-4,minstep=1.e-12,\
				 secfact=0.9,secfact_max=5.,secfact_min=0.2,\
				 eventfun=None,outputfun=None):
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
		self.eventfcn    = eventfun
		self.outputfcn   = outputfun 

	def set_h(self,xspan,div=10.):
		'''
		Initial and max step based on xspan
		'''
		self.h0 = (xspan[1] - xspan[0])/div

	# We can expose every variable here so it can be accessed by python.
	# These operations are slow and costly to do so it is wise just 
	# to perform them once.
	@property
	def h0(self):
		return self.h0
	@h0.setter
	def h0(self, double h0):
		self.h0 = h0
	@property
	def eps(self):
		return self.eps
	@eps.setter
	def eps(self, double eps):
		self.eps = eps
	@property
	def epsevf(self):
		return self.epsevf
	@epsevf.setter
	def epsevf(self, double epsevf):
		self.epsevf = epsevf
	@property
	def minstep(self):
		return self.minstep
	@minstep.setter
	def minstep(self, double minstep):
		self.minstep = minstep
	@property
	def secfact(self):
		return self.secfact
	@secfact.setter
	def secfact(self, double secfact):
		self.secfact = secfact
	@property
	def secfact_max(self):
		return self.secfact_max
	@secfact_max.setter
	def secfact_max(self, double secfact_max):
		self.secfact_max = secfact_max
	@property
	def secfact_min(self):
		return self.secfact_min
	@secfact_min.setter
	def secfact_min(self, double secfact_min):
		self.secfact_min = secfact_min		


def odeRK(scheme,fun,xspan,y0,params=odeset()):
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
	if not scheme.lower() in RK_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)


	# Implement here the RungeKutta algorithm from RK.cpp
	raise NotImplementedError('To implement!')


	# Return
	if retval == 0:
		raise ValueError('ERROR! Something went wrong during the integration.')
	if retval == -1:
		raise ValueError('ERROR! Integration step required under minimum.')

	return x,y,err

def ode23(fun,xspan,y0,params=odeset()):
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

def ode45(fun,xspan,y0,params=odeset()):
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


def odeRKN(scheme,fun,xspan,y0,dy0,params=odeset()):
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

	
	# Implement RKN function from RK.cpp here
	raise NotImplementedError('To implement!')

	
	# Return
	if retval == 0:
		raise ValueError('ERROR! Something went wrong during the integration.')
	if retval == -1:
		raise ValueError('ERROR! Integration step required under minimum.')

	return x,y,dy,err


# Convert to a function that python can see
def CheckTableau(object scheme):
	'''
	Check the sanity of the Butcher's Tableau 
	of a specified Runge-Kutta scheme.
	'''
	if not scheme.lower() in RK_SCHEMES:
		raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)


	# Implement check_tableau function from RK.cpp here
	raise NotImplementedError('To implement!')


#	return 