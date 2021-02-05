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

NNSTEP = 1000

class odeset():
	'''
	ODESET class

	Sets the parameters for odeRK. They are:
		> h0:        Initial step for the interval
		> eps:       Tolerance to meet (Relative).
		> eventfcn:  Event function.
		> outputfcn: Output function.
	'''
	def __init__(self,h0=.01,eps=1.e-8,epsevf=1.e-4,minstep=1.e-12,
				 secfact=0.9,secfact_max=5.,secfact_min=0.2,
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

	hmin = abs(xspan[1]-xspan[0]) * params.minstep

	# Select Runge-Kutta method
	rkm = RKMethod(scheme)
	
	# Initialization
	cont, last = True, False
	h = params.h0
	retval = 0
	dim = len(y0)
	
	# Create temporary arrays to store the output
	x = np.zeros((NNSTEP,), dtype=np.double)
	y = np.zeros((NNSTEP,dim), dtype=np.double)
	
	n      = 0 # Start at iteration zero
	err    = 0.
	x[n]   = xspan[0]
	y[n,:] = y0
		
	# Vector containing the derivatives
	f = np.zeros((rkm.nstep, dim), dtype=np.double)
	
	# Definitions
	# int dir[1]
#	ylow  = np.zeros((dim,), dtype=np.double)
#	yhigh = np.zeros((dim,), dtype=np.double)
	dydx  = np.zeros((dim,), dtype=np.double)
	# val = 0
	# g_ant = 0.
	
	# Runge-Kutta loop
	while cont:
		# Exit criteria
		if x[n] + h > xspan[1]:
			cont, last = False, True
			# Arrange the step so it finishes at xspan[1]
			h = abs(x[n] - xspan[1])
			
		# Initialize
		ylow  = y[n,:].copy()
		yhigh = y[n,:].copy()
		
		# Calculus loop
		for ii in range(rkm.nstep):
			xint = x[n] + h * rkm.C[ii]
			yint = y[n,:].copy()
			
			for kk in range(dim):
				for jj in range(ii):
					yint[kk] += rkm.A[ii,jj] * f[jj,kk]
					
			# Call function
			fun(xint, yint, dim, dydx)

			# Update dydx and compute solutions
			f[ii,:]   = h * dydx
			ylow[:]  += rkm.Bhat[ii] * f[ii,:]
			yhigh[:] += rkm.B[ii] * f[ii,:]

		# Compute the total error
		# Work with both relative and absolute errors
		rel_err = np.max(np.abs(1.-ylow/yhigh))
		abs_err = np.max(np.abs(yhigh-ylow))
			
		error = max(1e-20,min(rel_err,abs_err)) # Avoid division by zero

		# Step size control
		# Source: Ketcheson, David, and Umair bin Waheed. 
		#         "A comparison of high-order explicit Runge-Kutta, extrapolation, and deferred correction methods in serial and parallel." 
		#         Communications in Applied Mathematics and Computational Science 9.2 (2014): 175-200.
		hest = params.secfact * h * (params.eps/error)**(0.7/rkm.alpha)
		
		# Event function
		if params.eventfcn:
			# do stuff
			#ACABAR!!
			pass
		
		if error < params.eps or last:
			# This is a successful step
			retval = 1 if retval <= 0 else retval
			err    = max(error,err)
			n     += 1
			
			# Reallocate
			if n % NNSTEP == 0:
				x = np.hstack([x, np.zeros((NNSTEP,), dtype=np.double)])
				y = np.vstack([y, np.zeros((NNSTEP, dim), dtype=np.double)])
				
			# Set output values
			x[n]   = x[n-1] + h
			y[n,:] = yhigh
			
			# Output function
			if params.outputfcn:
				cont = bool(params.outputfcn(x[-1], yhigh, dim)) if cont else False
			
			# Set the new step
			# Source: Ketcheson, David, and Umair bin Waheed. 
			#         "A comparison of high-order explicit Runge-Kutta, extrapolation, and deferred correction methods in serial and parallel." 
			#         Communications in Applied Mathematics and Computational Science 9.2 (2014): 175-200.
	
			h = min(params.secfact_max * h, max(params.secfact_min * h, hest))
		
		else:
			# This is a failed step
			if h < hmin:
				# Check if our step has decreased too much
				cont, retval = False, -1
			else:
				h = hest

	# Return
	if retval == 0:
		raise ValueError('ERROR! Something went wrong during the integration.')
	if retval == -1:
		raise ValueError('ERROR! Integration step required under minimum.')

	return x[:n], y[:n,:], err

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

	retval = 1
	# Return
	if retval == 0:
		raise ValueError('ERROR! Something went wrong during the integration.')
	if retval == -1:
		raise ValueError('ERROR! Integration step required under minimum.')

	# return x,y,dy,err
	return

class RKMethod():
	def __init__(self, scheme):
		if scheme == 'eulerheun12':
			self.EulerHeun12()
		elif scheme == 'bogackishampine23':
			self.Bogackishampine23()
		elif scheme == 'dormandprince34a':
			self.DormandPrince34A()
		elif scheme == 'fehlberg45':
			self.Fehlberg45()
		elif scheme == 'cashkarp45':
			self.CashKarp45()
		elif scheme == 'dormandprince45':
			self.Dormandprince45()
		elif scheme == 'dormandprince45a':
			self.DormandPrince45A()
		elif scheme == 'calvo56':
			self.Calvo56()
		elif scheme == 'dormandprince78':
			self.DormandPrince78()
		elif scheme == 'curtis810':
			self.Curtis810()
		elif scheme == 'hiroshi912':
			self.Hiroshi912()
		else:
			raise ValueError('ERROR! Scheme <%s> not implemented.' % scheme)
		return
				
	def EulerHeun12(self):
		self.nstep = 2
		self.alpha = 1.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = 1.
	
		# A matrix
		self.A[1,0] = 1.
	
		# 2nd order solution
		self.B[0] = .5
		self.B[1] = .5
	
		# 1st order solution
		self.Bhat[0] = 1.
		self.Bhat[1] = 0.
		return
			
	def Bogackishampine23(self):
		self.nstep = 4
		self.alpha = 2.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = .25
		self.C[2] = 3./4.
		self.C[3] = 1.
	
		# A matrix
		self.A[1,0] = .25
		self.A[2,0] = 0.
		self.A[2,1] = 3./4.
		self.A[3,0] = 2./9.
		self.A[3,1] = 1./3.
		self.A[3,2] = 4./9.
	
		# 3rd order solution
		self.B[0] = 2./9.
		self.B[1] = 1./3.
		self.B[2] = 4./9.
		self.B[3] = 0.
	
		# 2nd order solution
		self.Bhat[0] = 7./24.
		self.Bhat[1] = 1./4.
		self.Bhat[2] = 1./3.
		self.Bhat[3] = 1./8.
		return
		
	def DormandPrince34A(self):
		pLambda = 0.1
	
		self.nstep = 5
		self.alpha = 3.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = 0.5
		self.C[2] = 0.5
		self.C[3] = 1.
		self.C[4] = 1.
	
		# A matrix
		self.A[1,0] = 0.5
		self.A[2,0] = 0.
		self.A[2,1] = 0.5
		self.A[3,0] = 0.
		self.A[3,1] = 0.
		self.A[3,2] = 1.
		self.A[4,0] = 1./6.
		self.A[4,1] = 1./3.
		self.A[4,2] = 1./3.
		self.A[4,3] = 1./6.
	
		# 5th order solution
		self.B[0] = 1./6.
		self.B[1] = 1./3.
		self.B[2] = 1./3.
		self.B[3] = 1./6.
		self.B[4] = 0.
	
		# 4th order solution
		self.Bhat[0] = 1./6.
		self.Bhat[1] = 1./3.
		self.Bhat[2] = 1./3.
		self.Bhat[3] = 1./6.-pLambda
		self.Bhat[4] = pLambda
		return
		
	def Fehlberg45(self):
		self.nstep = 6
		self.alpha = 4.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = .25
		self.C[2] = 3./8.
		self.C[3] = 12./13.
		self.C[4] = 1.
		self.C[5] = 1./2.
	
		# A matrix
		self.A[1,0] = .25
		self.A[2,0] = 3./32.
		self.A[2,1] = 9./32.
		self.A[3,0] = 1932./2197.
		self.A[3,1] = -7200./2197.
		self.A[3,2] = 7296./ 2197.
		self.A[4,0] = 439./216.
		self.A[4,1] = -8.
		self.A[4,2] = 3680./513.
		self.A[4,3] = -845./4104.
		self.A[5,0] = -8./27.
		self.A[5,1] = 2.
		self.A[5,2] = -3544./2565.
		self.A[5,3] = 1859./4104.
		self.A[5,4] = -11./40.
	
		# 5th order solution
		self.B[0] = 16./135.
		self.B[1] = 0.
		self.B[2] = 6656./12825.
		self.B[3] = 28561./56430
		self.B[4] = -9./50.
		self.B[5] = 2./55.
	
		# 4th order solution
		self.Bhat[0] = 25./216.
		self.Bhat[1] = 0.
		self.Bhat[2] = 1408./2565. 
		self.Bhat[3] = 2197./4104.
		self.Bhat[4] = -1./5.
		self.Bhat[5] = 0.
		return
	
	def CashKarp45(self):
		self.nstep = 6
		self.alpha = 4.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = 1./5.
		self.C[2] = 3./10.
		self.C[3] = 3./5.
		self.C[4] = 1.
		self.C[5] = 7./8.
	
		# A matrix
		self.A[1,0] = 1./5.
		self.A[2,0] = 3./40.
		self.A[2,1] = 9./40.
		self.A[3,0] = 3./10.
		self.A[3,1] = -9./10.
		self.A[3,2] = 6./5.
		self.A[4,0] = -11./54.
		self.A[4,1] = 5./2.
		self.A[4,2] = -70./27.
		self.A[4,3] = 35./27.
		self.A[5,0] = 1631./55296.
		self.A[5,1] = 175./512. 
		self.A[5,2] = 575./13824.
		self.A[5,3] = 44275./110592.
		self.A[5,4] = 253./4096.
	
		# 5th order solution
		self.B[0] = 37./378.
		self.B[1] = 0.
		self.B[2] = 250./621.
		self.B[3] = 125./594.
		self.B[4] = 0.
		self.B[5] = 512./1771.
	
		# 4th order solution
		self.Bhat[0] = 2825./27648.
		self.Bhat[1] = 0.
		self.Bhat[2] = 18575./48384.
		self.Bhat[3] = 13525./55296.
		self.Bhat[4] = 277./14336.
		self.Bhat[5] = 1./4.
		return
	
	def Dormandprince45(self):
		self.nstep = 7
		self.alpha = 4.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = .2
		self.C[2] = .3
		self.C[3] = 4./5.
		self.C[4] = 8./9.
		self.C[5] = 1.
		self.C[6] = 1.
	
		# A matrix
		self.A[1,0] = .2
		self.A[2,0] = .3/4.
		self.A[2,1] = .9/4.
		self.A[3,0] = 44./45.
		self.A[3,1] = -56./15.
		self.A[3,2] = 32./ 9.
		self.A[4,0] = 19372./6561.
		self.A[4,1] = -25360./2187.
		self.A[4,2] = 64448./6561.
		self.A[4,3] = -212./729.
		self.A[5,0] = 9017./3168.
		self.A[5,1] = -355./33.
		self.A[5,2] = 46732./5247.
		self.A[5,3] = 49./176.
		self.A[5,4] = -5103./18656.
		self.A[6,0] = 35./384.
		self.A[6,1] = 0.
		self.A[6,2] = 500./1113.
		self.A[6,3] = 125./192.
		self.A[6,4] = -2187./6784.
		self.A[6,5] = 11./84.
	
		# 5th order solution
		self.B[0] = 35./384.
		self.B[1] = 0.
		self.B[2] = 500./1113.
		self.B[3] = 125./192.
		self.B[4] = -2187./6784.
		self.B[5] = 11./84.
		self.B[6] = 0.
	
		# 4th order solution
		self.Bhat[0] = 5179./57600.
		self.Bhat[1] = 0.
		self.Bhat[2] = 7571./16695.
		self.Bhat[3] = 393./640.
		self.Bhat[4] = -92097./339200.
		self.Bhat[5] = 187./2100.
		self.Bhat[6] = 1./40.
		return
	
	def DormandPrince45A(self):
		pLambda = 1./60.
	
		self.nstep = 7
		self.alpha = 4.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = .125
		self.C[2] = .25
		self.C[3] = 4./9.
		self.C[4] = 4./5.
		self.C[5] = 1.
		self.C[6] = 1.
	
		# A matrix
		self.A[1,0] = .125
		self.A[2,0] = 0.
		self.A[2,1] = 1./4.
		self.A[3,0] = 196./729.
		self.A[3,1] = -320./729.
		self.A[3,2] = 448./729.
		self.A[4,0] = 836./2875.
		self.A[4,1] = 64./575.
		self.A[4,2] = -13376./20125.
		self.A[4,3] = 21384./20125.
		self.A[5,0] = -73./48.
		self.A[5,1] = 0.
		self.A[5,2] = 1312./231.
		self.A[5,3] = -2025./448.
		self.A[5,4] = 2875./2112.
		self.A[6,0] = 17./192.
		self.A[6,1] = 0.
		self.A[6,2] = 64./231.
		self.A[6,3] = 2187./8960.
		self.A[6,4] = 2875./8448.
		self.A[6,5] = 1./20.
	
		# 5th order solution
		self.B[0] = 17./192.
		self.B[1] = 0.
		self.B[2] = 64./231.
		self.B[3] = 2187./8960.
		self.B[4] = 2875./8448.
		self.B[5] = 1./20.
		self.B[6] = 0.
	
		# 4th order solution
		self.Bhat[0] = 17./192.
		self.Bhat[1] = 0.
		self.Bhat[2] = 64./231.
		self.Bhat[3] = 2187./8960.
		self.Bhat[4] = 2875./8448.
		self.Bhat[5] = 1./20.-pLambda
		self.Bhat[6] = pLambda
		return
	
	def Calvo56(self):
		self.nstep = 9
		self.alpha = 5.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0] = 0.
		self.C[1] = 2./15.
		self.C[2] = 1./5.
		self.C[3] = 3./10.
		self.C[4] = 14./25.
		self.C[5] = 19./25.
		self.C[6] = 35226607./35688279.
		self.C[7] = 1.
		self.C[8] = 1.
	
		# A matrix
		self.A[1,0] = 2./15.
		self.A[2,0] = 1./20.
		self.A[2,1] = 3./20.
		self.A[3,0] = 3./40.
		self.A[3,1] = 0.
		self.A[3,2] = 9./40.
		self.A[4,0] = 86727015./196851553.
		self.A[4,1] = -60129073./52624712.
		self.A[4,2] = 957436434./1378352377.
		self.A[4,3] = 83886832./147842441.
		self.A[5,0] = -86860849./45628967.
		self.A[5,1] = 111022885./25716487.
		self.A[5,2] = 108046682./101167669.
		self.A[5,3] = -141756746./36005461.
		self.A[5,4] = 73139862./60170633.
		self.A[6,0] = 77759591./16096467.
		self.A[6,1] = -49252809./6452555.
		self.A[6,2] = -381680111./51572984.
		self.A[6,3] = 879269579./66788831.
		self.A[6,4] = -90453121./33722162.
		self.A[6,5] = 111179552./157155827.
		self.A[7,0] = 237564263./39280295.
		self.A[7,1] = -100523239./10677940.
		self.A[7,2] = -265574846./27330247.
		self.A[7,3] = 317978411./18988713.
		self.A[7,4] = -124494385./35453627.
		self.A[7,5] = 86822444./100138635.
		self.A[7,6] = -12873523./724232625.
		self.A[8,0] = 17572349./289262523.
		self.A[8,1] = 0.
		self.A[8,2] = 57513011./201864250.
		self.A[8,3] = 15587306./354501571.
		self.A[8,4] = 71783021./234982865.
		self.A[8,5] = 29672000./180480167.
		self.A[8,6] = 65567621./127060952.
		self.A[8,7] = -79074570./210557597.
	
		# 8th order solution
		self.B[0] = 17572349./289262523.
		self.B[1] = 0.
		self.B[2] = 57513011./201864250.
		self.B[3] = 15587306./354501571.
		self.B[4] = 71783021./234982865.
		self.B[5] = 29672000./180480167.
		self.B[6] = 65567621./127060952.
		self.B[7] = -79074570./210557597.
		self.B[8] = 0.
	
		# 5th order solution
		self.Bhat[0] = 15231665./510830334.
		self.Bhat[1] = 0.
		self.Bhat[2] = 59452991./116050448.
		self.Bhat[3] = -28398517./122437738.
		self.Bhat[4] = 56673824./137010559.
		self.Bhat[5] = 68003849./426673583.
		self.Bhat[6] = 7097631./37564021.
		self.Bhat[7] = -71226429./583093742.
		self.Bhat[8] = 1./20.
		return
	
	def DormandPrince78(self):
		self.nstep = 13
		self.alpha = 7.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0]  = 0.
		self.C[1]  = .5555555555555555555555555555555555555555555555555555555555555555555555555555555555556e-1
		self.C[2]  = .8333333333333333333333333333333333333333333333333333333333333333333333333333333333333e-1
		self.C[3]  = .125
		self.C[4]  = .3125
		self.C[5]  = .375
		self.C[6]  = .1475
		self.C[7]  = .465
		self.C[8]  = .5648654513822595753983585014261682587385670087264133215107858541861772043626893634869
		self.C[9]  = .65
		self.C[10] = .9246562776405044467450135743183695426492034467027398177693057974193090932038137681734
		self.C[11] = 1.
		self.C[12] = 1.
	
		# A matrix
		self.A[1,0]   = .5555555555555555555555555555555555555555555555555555555555555555555555555555555555556e-1
		self.A[2,0]   = .2083333333333333333333333333333333333333333333333333333333333333333333333333333333333e-1
		self.A[2,1]   = .625e-1
		self.A[3,0]   = .3125e-1
		self.A[3,1]   = 0.
		self.A[3,2]   = .9375e-1
		self.A[4,0]   = .3125
		self.A[4,1]   = 0.
		self.A[4,2]   = -1.171875
		self.A[4,3]   = 1.171875
		self.A[5,0]   = .375e-1
		self.A[5,1]   = 0.
		self.A[5,2]   = 0.
		self.A[5,3]   = .1875
		self.A[5,4]   = .15
		self.A[6,0]   = .4791013711111111111111111111111111111111111111111111111111111111111111111111111111111e-1
		self.A[6,1]   = 0.
		self.A[6,2]   = 0.
		self.A[6,3]   = .1122487127777777777777777777777777777777777777777777777777777777777777777777777777778
		self.A[6,4]   = -.2550567377777777777777777777777777777777777777777777777777777777777777777777777777778e-1
		self.A[6,5]   = .1284682388888888888888888888888888888888888888888888888888888888888888888888888888889e-1
		self.A[7,0]   = .1691798978729228118143110713603823606551492879543441068765183916901985069878116840258e-1
		self.A[7,1]   = 0.
		self.A[7,2]   = 0.
		self.A[7,3]   = .3878482784860431695265457441593733533707275558526027137301504136188413357699752956243
		self.A[7,4]   = .3597736985150032789670088963477236800815873945874968299566918115320174598617642813621e-1
		self.A[7,5]   = .1969702142156660601567152560721498881281698021684239074332818272245101975612112344987
		self.A[7,6]   = -.1727138523405018387613929970023338455725710262752107148467532611655731300161441266618
		self.A[8,0]   = .6909575335919230064856454898454767856104137244685570409618590205329379141801779261271e-1
		self.A[8,1]   = 0.
		self.A[8,2]   = 0.
		self.A[8,3]   = -.6342479767288541518828078749717385465426336031120747443029278238678259156558727343870
		self.A[8,4]   = -.1611975752246040803668769239818171234422237864808493434053355718725564004275765826294
		self.A[8,5]   = .1386503094588252554198669501330158019276654889495806914244800957316809692178564181463
		self.A[8,6]   = .9409286140357562697242396841302583434711811483145028697758469500622977999086075976730
		self.A[8,7]   = .2116363264819439818553721171319021047635363886083981439225363020792869599016568720713
		self.A[9,0]   = .1835569968390453854898060235368803084497642516277329033567822939550366893470341427061
		self.A[9,1]   = 0.
		self.A[9,2]   = 0.
		self.A[9,3]   = -2.468768084315592452744315759974107457777649753547732495587510208118948846864075671103
		self.A[9,4]   = -.2912868878163004563880025728039519800543380294081727678722180378521228988619909525814
		self.A[9,5]   = -.2647302023311737568843979946594614325963055282676866851254669985184857900430792761417e-1
		self.A[9,6]   = 2.847838764192800449164518254216773770231582185011953158945518598604677368771006006947
		self.A[9,7]   = .2813873314698497925394036418267117820980705455360140173438806414700945017562775967023
		self.A[9,8]   = .1237448998633146576270302126636397203122013536069738523260934117931117648560568049432
		self.A[10,0]  = -1.215424817395888059160510525029662994880024988044112261492933435562347094075572196306
		self.A[10,1]  = 0.
		self.A[10,2]  = 0.
		self.A[10,3]  = 16.67260866594577243228041328856410774858907460078758981907659463146817850515953150318
		self.A[10,4]  = .9157418284168179605957186504507426331593736334298114556675054711691705693180672015549
		self.A[10,5]  = -6.056605804357470947554505543091634004081967083324413691952297768849489422976536037188
		self.A[10,6]  = -16.00357359415617811184170641007882303068079304063907122528682690677051264091851070681
		self.A[10,7]  = 14.84930308629766255754539189802663208272299893302745636963476380528935538206408294404
		self.A[10,8]  = -13.37157573528984931829304139618159579089195289469740214314819172008509426917249413996
		self.A[10,9]  = 5.134182648179637933173253611658602898712494286162881495270691720760048063805245199662
		self.A[11,0]  = .2588609164382642838157309322317577667296307766301063163257807024364127266506000943372
		self.A[11,1]  = 0.
		self.A[11,2]  = 0.
		self.A[11,3]  = -4.774485785489205112310117509706042746829391853746564335432052648850273207666153414325
		self.A[11,4]  = -.4350930137770325094407004118103177819323551661617974116300885410959889587870529265443
		self.A[11,5]  = -3.049483332072241509560512866312031613982854911220735245095857885009959537455491093975
		self.A[11,6]  = 5.577920039936099117423676634464941858623588944531498679305832747426169795721583100328
		self.A[11,7]  = 6.155831589861040097338689126688954481197754937462937466615262374558053507207577588692
		self.A[11,8]  = -5.062104586736938370077406433910391644990220712141673880668482097753119682588189892664
		self.A[11,9]  = 2.193926173180679061274914290465806019788262707389033759251111926114017568440058142981
		self.A[11,10] = .1346279986593349415357262378873236613955852772571946513284934221746877884770684011715
		self.A[12,0]  = .8224275996265074779631682047726665909572303617765850630130165537026064642659285212794
		self.A[12,1]  = 0.
		self.A[12,2]  = 0.
		self.A[12,3]  = -11.65867325727766428397655303545841477547369082638864247596731917545919361630644756876
		self.A[12,4]  = -.7576221166909361958811161540882449653663757591954118634754438929149012424414834787423
		self.A[12,5]  = .7139735881595815279782692827650546753142248878566301594716125352788068216039494192152
		self.A[12,6]  = 12.07577498689005673956617044860067967095705800972197897084832370633043082304810801069
		self.A[12,7]  = -2.127659113920402656390820858969398635427927973275119750336406473203376323696576615278
		self.A[12,8]  = 1.990166207048955418328071698344314152176173017359794982793519250552799420955298804232
		self.A[12,9]  = -.2342864715440402926602946918568015314512425817004394700255034276765120670064339827205
		self.A[12,10] = .1758985777079422650731051058901448183145508638446243836782009233893397195776568900819
		self.A[12,11] = 0.
	
		# 8th order solution
		self.B[0]  = 4.17474911415302462220859284685e-2
		self.B[1]  = 0.
		self.B[2]  = 0.
		self.B[3]  = 0.
		self.B[4]  = 0.
		self.B[5]  = -5.54523286112393089615218946547e-2
		self.B[6]  = 2.39312807201180097046747354249e-1
		self.B[7]  = 7.0351066940344302305804641089e-1
		self.B[8]  = -7.59759613814460929884487677085e-1
		self.B[9]  = 6.60563030922286341461378594838e-1
		self.B[10] = 1.58187482510123335529614838601e-1
		self.B[11] = -2.38109538752862804471863555306e-1
		self.B[12] = 2.5e-1
	
		# 7th order solution
		self.Bhat[0]  = 2.9553213676353496981964883112e-2
		self.Bhat[1]  = 0.
		self.Bhat[2]  = 0.
		self.Bhat[3]  = 0.
		self.Bhat[4]  = 0.
		self.Bhat[5]  = -8.28606276487797039766805612689e-1
		self.Bhat[6]  = 3.11240900051118327929913751627e-1
		self.Bhat[7]  = 2.46734519059988698196468570407
		self.Bhat[8]  = -2.54694165184190873912738007542
		self.Bhat[9]  = 1.44354858367677524030187495069
		self.Bhat[10] = 7.94155958811272872713019541622e-2
		self.Bhat[11] = 4.44444444444444444444444444445e-2
		self.Bhat[12] = 0.
		return
	
	def Curtis810(self):
		self.nstep = 21
		self.alpha = 8.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0]  = 0.
		self.C[1]  = .1452518960316150517617548528770033320314511251329947060838468741983976455607179673401
		self.C[2]  = .1452518960316150517617548528770033320314511251329947060838468741983976455607179673401
		self.C[3]  = .2178778440474225776426322793155049980471766876994920591257703112975964683410769510101
		self.C[4]  = .5446946101185564441065806982887624951179417192487301478144257782439911708526923775252
		self.C[5]  = .6536335321422677329278968379465149941415300630984761773773109338927894050232308530303
		self.C[6]  = .2746594919905254008808021630247618520892150865127407293922085868737635475402543533498
		self.C[7]  = .7735775201106609448405825008093973718589542913426807556412662673054607938029043386501
		self.C[8]  = .5801831400829957086304368756070480288942157185070105667309497004790955953521782539876
		self.C[9]  = .1174723380352676535744985130203309248171321557319478803362088220814723414805867429383
		self.C[10] = .3573842417596774518429245029795604640404982636367873040901247917361510345429002009092
		self.C[11] = .6426157582403225481570754970204395359595017363632126959098752082638489654570997990908
		self.C[12] = .1174723380352676535744985130203309248171321557319478803362088220814723414805867429383
		self.C[13] = .8825276619647323464255014869796690751828678442680521196637911779185276585194132570617
		self.C[14] = .3573842417596774518429245029795604640404982636367873040901247917361510345429002009092
		self.C[15] = .6426157582403225481570754970204395359595017363632126959098752082638489654570997990908
		self.C[16] = .8825276619647323464255014869796690751828678442680521196637911779185276585194132570617
		self.C[17] = 1.
		self.C[18] = .3510848126232741617357001972386587771203155818540433925049309664694280078895463510848
		self.C[19] = .6157407407407407407407407407407407407407407407407407407407407407407407407407407407407
		self.C[20] = 1.
	
		# A matrix
		self.A[1,0]   = .1452518960316150517617548528770033320314511251329947060838468741983976455607179673401
		self.A[2,0]   = .7262594801580752588087742643850166601572556256649735304192343709919882278035898367003e-1
		self.A[2,1]   = .7262594801580752588087742643850166601572556256649735304192343709919882278035898367003e-1
		self.A[3,0]   = .5446946101185564441065806982887624951179417192487301478144257782439911708526923775252e-1
		self.A[3,1]   = 0.
		self.A[3,2]   = .1634083830355669332319742094866287485353825157746190443443277334731973512558077132576
		self.A[4,0]   = .5446946101185564441065806982887624951179417192487301478144257782439911708526923775252
		self.A[4,1]   = 0.
		self.A[4,2]   = -2.042604787944586665399677618582859356692281447182738054304096668414966890697596415720
		self.A[4,3]   = 2.042604787944586665399677618582859356692281447182738054304096668414966890697596415720
		self.A[5,0]   = .6536335321422677329278968379465149941415300630984761773773109338927894050232308530303e-1
		self.A[5,1]   = 0.
		self.A[5,2]   = 0.
		self.A[5,3]   = .3268167660711338664639484189732574970707650315492380886886554669463947025116154265151
		self.A[5,4]   = .2614534128569070931711587351786059976566120252393904709509243735571157620092923412121
		self.A[6,0]   = .8233707757482716585173454344310125296066814318521742241762319051772963627695955263034e-1
		self.A[6,1]   = 0.
		self.A[6,2]   = 0.
		self.A[6,3]   = .2119171963202803561687843468555305553175658807629274312902985594840086570224567152664
		self.A[6,4]   = -.3997343508054218311577932550061320162379840049816347807630118786107674477850206579628e-1
		self.A[6,5]   = .2037865317596006197606259822674324543477946306275935376058802473310199901934015124941e-1
		self.A[7,0]   = .8595305779007343831562027786771081909543936570474230618236291858949564375587825985001e-1
		self.A[7,1]   = 0.
		self.A[7,2]   = 0.
		self.A[7,3]   = 0.
		self.A[7,4]   = 0.
		self.A[7,5]   = .2911769478058850960337179621761553399856026049598393013981874594942289837064329700000
		self.A[7,6]   = .3964475145147024104912442607655312127779123206780991480607158892217361663405931088001
		self.A[8,0]   = .8612093485606967549983047372292119178898514571588438099912534616486575243508895957628e-1
		self.A[8,1]   = 0.
		self.A[8,2]   = 0.
		self.A[8,3]   = 0.
		self.A[8,4]   = 0.
		self.A[8,5]   = .1397464826824442089036313891001189801074425314582326737716288563521183595455090268480
		self.A[8,6]   = .3951098495815674599900526056001284215294125840404176924334653987770478924197803010468
		self.A[8,7]   = -.4079412703708563576307759281612056453162454270752418047326990081493640904820003348350e-1
		self.A[9,0]   = .7233144422337948077616348229119326315582930871089020733092900891206129381937795204778e-1
		self.A[9,1]   = 0.
		self.A[9,2]   = 0.
		self.A[9,3]   = 0.
		self.A[9,4]   = 0.
		self.A[9,5]   = .2200276284689998102140972735735070061373242800181187459951219347361114857342828430157
		self.A[9,6]   = .8789533425436734013369780264792573637952226487753296416823846876217040795688489371334e-1
		self.A[9,7]   = -.4445383996260350863990674880611108986832860648196030000580004690002268108984238641730e-1
		self.A[9,8]   = -.2183282289488754689095532966861839909872150913926337371522805434288481649401165594213
		self.A[10,0]  = .8947100936731114228785441966773836169071038390882857211057269158522704971585365845223e-1
		self.A[10,1]  = 0.
		self.A[10,2]  = 0.
		self.A[10,3]  = 0.
		self.A[10,4]  = 0.
		self.A[10,5]  = .3946008170285561860741397654755022300929434262701385530048127140223687993778661654316
		self.A[10,6]  = .3443011367963333487713764986067104675654371857504670290688086760696354596195596354011
		self.A[10,7]  = -.7946682664292661290694938113119430997053815140863772328764150866582492425892231395780e-1
		self.A[10,8]  = -.3915218947895966123834967996391962853380545808840091268064277812752553499114569444180
		self.A[10,9]  = 0.
		self.A[11,0]  = .3210006877963209212945282736072241886741425314298532400216927262619488479186214523312e-1
		self.A[11,1]  = 0.
		self.A[11,2]  = 0.
		self.A[11,3]  = 0.
		self.A[11,4]  = 0.
		self.A[11,5]  = 0.
		self.A[11,6]  = 0.
		self.A[11,7]  = -.1846375997512050141835163881753227910996323204749769226655464078048769505209525299752e-3
		self.A[11,8]  = .1560894025313219860759149162557283383430181475726228517203663063649626288079337909898
		self.A[11,9]  = .1934496857654560252749984220385188727138526287670744309970093278715606577140084022992
		self.A[11,10] = .2611612387636636496908928477536452288263163392010050661129958478089356710938164130987
		self.A[12,0]  = .4423749328524996327035388417792688154433173133294892285295756457561276315648477233732e-1
		self.A[12,1]  = 0.
		self.A[12,2]  = 0.
		self.A[12,3]  = 0.
		self.A[12,4]  = 0.
		self.A[12,5]  = 0.
		self.A[12,6]  = 0.
		self.A[12,7]  = .4640774434539039636406222168781981616534115643208114455689698789119941732444857047798e-2
		self.A[12,8]  = .4704660282615136532130927218172390570903230981414159347904277946537920001824903276586e-1
		self.A[12,9]  = .8620749948011488160369445167416002799205317397013619044391270706339561700281526529703e-1
		self.A[12,10] = -.2607983024682138093233254079066687623148682426317395111719299641390118652802949600035e-1
		self.A[12,11] = -.3858020174396621532493277639159499581333235076531298977820093139813399390137768850940e-1
		self.A[13,0]  = .2318046717429411567006043539613275607940758021709332569729352990777336390158311630529e-1
		self.A[13,1]  = 0.
		self.A[13,2]  = 0.
		self.A[13,3]  = 0.
		self.A[13,4]  = 0.
		self.A[13,5]  = 0.
		self.A[13,6]  = 0.
		self.A[13,7]  = .3197856784116367067302124322582100058864027838197120089129330601737324659881765852593
		self.A[13,8]  = .5933233331841898686063939886797828376866051205773280426848164018120869674204443797948
		self.A[13,9]  = -1.937519548878479314706815782408229952008442222624773168771865465659822020582450444783
		self.A[13,10] = .1803950557030502357344063195737827904476240180662764468232042537858892203518134072359
		self.A[13,11] = -.4554014298857220726863505256926549022316460712353658688873150702827663762861750674926
		self.A[13,12] = 2.158764106255762807077594619172645539322916635447781333204724468181634037726021280742
		self.A[14,0]  = .2624364325798105891527733985858552391723553030719144065844544880498188553839263944447e-1
		self.A[14,1]  = 0.
		self.A[14,2]  = 0.
		self.A[14,3]  = 0.
		self.A[14,4]  = 0.
		self.A[14,5]  = 0.
		self.A[14,6]  = 0.
		self.A[14,7]  = .4863139423867266106526843913609225996253073727381961544415263239431571586043622332760e-1
		self.A[14,8]  = .4274382538346478867636942429421724367591866585774144180215122660980822123988151132213e-1
		self.A[14,9]  = -.4862259869465547771298976981868643277396586803130813159599600102115609499827986711663
		self.A[14,10] = .1326047194917652331781527125743684254490968718259563958293167893998110899691451568372
		self.A[14,11] = -.9402962152946515651634831658142934852383791641671387741034606371378082209616938685225e-1
		self.A[14,12] = .6993864679941022534190304512277131176659196396138275832136258135631963192299339871223
		self.A[14,13] = -.1197020013028860976492784934312243036670658451195397948726104511062042521592125912599e-1
		self.A[15,0]  = .5568066641536216461090823068917803436066365804361903532125349474551476120813558125830e-1
		self.A[15,1]  = 0.
		self.A[15,2]  = 0.
		self.A[15,3]  = 0.
		self.A[15,4]  = 0.
		self.A[15,5]  = 0.
		self.A[15,6]  = 0.
		self.A[15,7]  = -.4324853319508358432896036654421685136736530810118924113940744870078036705505610668088
		self.A[15,8]  = -.9979726994172038714656907882931844552238093285811791155499130927685987422432191170216
		self.A[15,9]  = 2.707893755718926115778725270396739994070337972517006747100005607751792006959604868323
		self.A[15,10] = -1.024823023512132929313567156576969954855232272749038347671818195935585095295127839150
		self.A[15,11] = 1.334565206642246959252239602313589265188981560552694580059808406200559397799055652161
		self.A[15,12] = -2.587748998830690939658228913150922979184368065866213469477796089200252812362701917187
		self.A[15,13] = .8992773696348355846430438306111181223414632598285854300924423251352733205187087732678e-1
		self.A[15,14] = 1.497578446211167333777988534023066333042434967475357134513165331964695787890042760189
		self.A[16,0]  = -.8434891199686377639125188391985671318383858641413517143104162188088468627447515172982e-3
		self.A[16,1]  = 0.
		self.A[16,2]  = 0.
		self.A[16,3]  = 0.
		self.A[16,4]  = 0.
		self.A[16,5]  = 0.
		self.A[16,6]  = 0.
		self.A[16,7]  = .7602144218856081893754106886111596435015500427480120290148318740899211421773423234728
		self.A[16,8]  = 1.769083927820959377467464871522349066447068428702073590698445112684989184432409492025
		self.A[16,9]  = -4.499239797622297101452915424261016593995695456495268863455643396071539024609271033574
		self.A[16,10] = 1.490558190212043468817221563278239942209691100326719140478588601720867838040211450448
		self.A[16,11] = -2.552203480132132516997563217309689292804518121743365818482497611667126218719069737195
		self.A[16,12] = 4.795167551528575994217413424533259845001657006088189480440731104737960266616292993321
		self.A[16,13] = -.9161854401769482236671414092387917470686251714192236693920061138984202381209109248553e-1
		self.A[16,14] = -1.525735678746850818217653470352135651821164556169070505816135230784807058389577753184
		self.A[16,15] = .7371445601564892133467497107205798584829803038168267854389817508169123996459113657504
		self.A[17,0]  = .1017366974111576638766809656369828971944080018220332809259398740674738807023371082700
		self.A[17,1]  = 0.
		self.A[17,2]  = 0.
		self.A[17,3]  = 0.
		self.A[17,4]  = 0.
		self.A[17,5]  = 0.
		self.A[17,6]  = 0.
		self.A[17,7]  = -1.696217553209432810711666838709742166182992092906177246174096517233561845662947862824
		self.A[17,8]  = -3.825235846211624254528740857512255693551264719132875740261231165548583482101116676418
		self.A[17,9]  = 9.754768979885866648856431516333641627109105703674164986615824197909762854575668793816
		self.A[17,10] = -2.520767789227152291196336314591227486393143379933686189126240710041836742414125694941
		self.A[17,11] = 5.472417145227780046950992000565734793413395536531652419585004300790370984185945495978
		self.A[17,12] = -9.781098113458736121002383874108051372067873053264954833376114258940736444388841687929
		self.A[17,13] = .3189152692455334369024560213486753019540464785641163242047782111839399471147176681561
		self.A[17,14] = 3.447227036527756718156475010324322155277035924051392880570525223655410460762027138915
		self.A[17,15] = -.6051983612219277832241707671295607127814820499715293613761402732652780120810041653591
		self.A[17,16] = .3334525350307787459202631378414806560287636505658634784117511174230383993073398823363
		self.A[18,0]  = -.1012987737478284424676828232882617689682012456457322189102956361570156443805900941944
		self.A[18,1]  = 0.
		self.A[18,2]  = 0.
		self.A[18,3]  = 0.
		self.A[18,4]  = 0.
		self.A[18,5]  = -.2409389328948775401304659380663043147167897928467308244359962659633933617326533285822e-1
		self.A[18,6]  = -.6679880790275182076676283582867036095782150170801495251932447614617249253864579543857
		self.A[18,7]  = 1.600262798493100648047998296908183265688507618079976446601985464092263571149154964705
		self.A[18,8]  = 3.706958893826695766827011000213884379914407774639901049574259778345288538246990591819
		self.A[18,9]  = -8.581755560147929325446798534254342948628755672447282004336563881429983605741487870996
		self.A[18,10] = .5607314974300953986559644699099897253584501767603091982484141468619493221310582281877e-1
		self.A[18,11] = -4.547761497422899514520768375507009011918601407646237921467449197008085790456674001879
		self.A[18,12] = 9.255775439941294621826928846245618922061242300726600002589630404152665447900428712156
		self.A[18,13] = -.3450876657451631707159097079770789925142348071643902737346329921538351794816584861003
		self.A[18,14] = 0.
		self.A[18,15] = 0.
		self.A[18,16] = 0.
		self.A[18,17] = 0.
		self.A[19,0]  = .3826909723812638609001259641818040193828105314579492422836388985468479567237561247336e-1
		self.A[19,1]  = 0.
		self.A[19,2]  = 0.
		self.A[19,3]  = 0.
		self.A[19,4]  = 0.
		self.A[19,5]  = .7786978965202527814624406274393101840018332461648638653990700950184871893714491273096
		self.A[19,6]  = .4859454140913448249612202172501868752761599132465501266008866131088163955018926230543
		self.A[19,7]  = 1.814925350154666364151014269029611427420766367555858499108920245656959783343309816408
		self.A[19,8]  = 4.551165245704657956889158854062833952834232753889932986749613143631480116805870313264
		self.A[19,9]  = -7.173770670344544101351160462586215092596352548535380880420409450623251883641801862305
		self.A[19,10] = -.3943009017000923237232456850787591816773705728833192412204243696911216045268772747196
		self.A[19,11] = -6.036544185898100312430357626685382432626027303329497026597513524312479466987506315664
		self.A[19,12] = 7.338904299721887701527380004651998686389416058019429466200740313593568240326087171554
		self.A[19,13] = -.4143158595971836110248598960027762194900538872022960061452263646470675916118824501965
		self.A[19,14] = 0.
		self.A[19,15] = 0.
		self.A[19,16] = 0.
		self.A[19,17] = 0.
		self.A[19,18] = -.3732349451502749258108621577582478607301443393311959731632798508493352335121760204375
		self.A[20,0]  = .2162339046022045866878628785550588026780578552494608097931198882276791962912244674840e-1
		self.A[20,1]  = 0.
		self.A[20,2]  = 0.
		self.A[20,3]  = 0.
		self.A[20,4]  = 0.
		self.A[20,5]  = .4611834700744369218866370212060318930941322187829670117414118166503940620998117275429
		self.A[20,6]  = .1940797759547798743610542713744618433967649025379792966207862125676964319674160574624
		self.A[20,7]  = .7041001229739959807963554405302474570280838416767002383409508232534658577705201658489
		self.A[20,8]  = 2.877431096792763528910415905652149398266490601780194388811216042455337979365709745445
		self.A[20,9]  = 0.
		self.A[20,10] = -.4332742088749107411735902392606181444105337491234912425673655059805456011404518143074
		self.A[20,11] = -2.234178753588834452567105459024473991729105867012210449973082203886376638514123583334
		self.A[20,12] = .2235678086885984010238782832657956960650576194069632574873732156942360146780276407657
		self.A[20,13] = .1293532338308457711442786069651741293532338308457711442786069651741293532338308457711
		self.A[20,14] = 0.
		self.A[20,15] = 0.
		self.A[20,16] = 0.
		self.A[20,17] = 0.
		self.A[20,18] = .1418136968194278394808045812385429206355105705182818920178205766092934777719870449624
		self.A[20,19] = -1.085699633131323582531514699802817081967439754938101617737029931360398856861850276906
	
		# 10th order solution
		self.B[0]  = .3333333333333333333333333333333333333333333333333333333333333333333333333333333333333e-1
		self.B[1]  = 0.
		self.B[2]  = 0.
		self.B[3]  = 0.
		self.B[4]  = 0.
		self.B[5]  = 0.
		self.B[6]  = 0.
		self.B[7]  = 0.
		self.B[8]  = 0.
		self.B[9]  = 0.
		self.B[10] = 0.
		self.B[11] = .1387145942588715882541801312803271702142521598590204181697361204933422401935856968980
		self.B[12] = .1892374781489234901583064041060123262381623469486258303271944256799821862794952728707
		self.B[13] = .9461873907446174507915320205300616311908117347431291516359721283999109313974763643533e-1
		self.B[14] = .2774291885177431765083602625606543404285043197180408363394722409866844803871713937960
		self.B[15] = .1387145942588715882541801312803271702142521598590204181697361204933422401935856968980
		self.B[16] = .9461873907446174507915320205300616311908117347431291516359721283999109313974763643533e-1
		self.B[17] = .3333333333333333333333333333333333333333333333333333333333333333333333333333333333333e-1
		self.B[18] = 0.
		self.B[19] = 0.
		self.B[20] = 0.
	
		# 8th order solution
		self.Bhat[0]  = .3339829895931337572271945815422988633728883413227543303554098429202731077409488318421e-1
		self.Bhat[1]  = 0.
		self.Bhat[2]  = 0.
		self.Bhat[3]  = 0.
		self.Bhat[4]  = 0.
		self.Bhat[5]  = 0.
		self.Bhat[6]  = 0.
		self.Bhat[7]  = 0.
		self.Bhat[8]  = .5024509803921568627450980392156862745098039215686274509803921568627450980392156862745e-1
		self.Bhat[9]  = -.1423859191318858946753152353981644782061337055184060977838998119673893661279423564924
		self.Bhat[10] = .2126013199429258434998789109063801828540550730541648287733608913970804891935883227446
		self.Bhat[11] = .3254854965632843133622967470840062095221514741629108993207015882688341071771214986692
		self.Bhat[12] = .3312629399585921325051759834368530020703933747412008281573498964803312629399585921325
		self.Bhat[13] = .1887845809230650005639203350759631573314744764356665687950807917985316096487827551997
		self.Bhat[14] = 0.
		self.Bhat[15] = 0.
		self.Bhat[16] = 0.
		self.Bhat[17] = 0.
		self.Bhat[18] = .6159811094287144604404508847679200761569154839698701533406779753675999506388462962070e-1
		self.Bhat[19] = -.9440109660594088037957791636147830082275023098021053120999763935859315478744850552959e-1
		self.Bhat[20] = .3341117040855897708234682470384970584684876341854831047975628586614323631403861184369e-1
		return
	
	def Hiroshi912(self):
		self.nstep = 29
		self.alpha = 9.
	
		# Allocate Butcher's tableau
		self.C     = np.zeros((self.nstep,), dtype=np.double)
		self.A     = np.zeros((self.nstep,self.nstep), dtype=np.double)
		self.B     = np.zeros((self.nstep,), dtype=np.double)
		self.Bhat  = np.zeros((self.nstep,), dtype=np.double)
		self.Bp    = 1
		self.Bphat = 1
	
		self.alloc = True
	
		# C coefficient
		self.C[0]  = 0.
		self.C[1]  = .4351851851851851851851851851851851851851851851851851851851851851851851851851851851852
		self.C[2]  = .4429824561403508771929824561403508771929824561403508771929824561403508771929824561404
		self.C[3]  = .6644736842105263157894736842105263157894736842105263157894736842105263157894736842105
		self.C[4]  = .1069403994175161223216143124609943831911795298522987310172664863740378614520490950697
		self.C[5]  = .1644736842105263157894736842105263157894736842105263157894736842105263157894736842105
		self.C[6]  = .5843251088534107402031930333817126269956458635703918722786647314949201741654571843251
		self.C[7]  = .6382358235823582358235823582358235823582358235823582358235823582358235823582358235824e-1
		self.C[8]  = .2
		self.C[9]  = .3333333333333333333333333333333333333333333333333333333333333333333333333333333333333
		self.C[10] = .9446116054065563496847375720237340591364440847545279548637407620203329280139199114088
		self.C[11] = .5179584680428461035835615460185207449756996828425551866437683588600807071775316523299e-1
		self.C[12] = .8488805186071653506398389301626743020641481756400195420459339398355773991365476236893e-1
		self.C[13] = .2655756032646428930981140590456168352972012641640776214486652703185222349414361456016
		self.C[14] = .5
		self.C[15] = .7344243967353571069018859409543831647027987358359223785513347296814777650585638543984
		self.C[16] = .9151119481392834649360161069837325697935851824359980457954066060164422600863452376311
		self.C[17] = .9446116054065563496847375720237340591364440847545279548637407620203329280139199114088
		self.C[18] = .3333333333333333333333333333333333333333333333333333333333333333333333333333333333333
		self.C[19] = .2
		self.C[20] = .5843251088534107402031930333817126269956458635703918722786647314949201741654571843251
		self.C[21] = .1644736842105263157894736842105263157894736842105263157894736842105263157894736842105
		self.C[22] = .4429824561403508771929824561403508771929824561403508771929824561403508771929824561404
		self.C[23] = .4351851851851851851851851851851851851851851851851851851851851851851851851851851851852
		self.C[24] = 1.
		self.C[25] = .4970267001007476028032930885363848318550815967070143979104342007017543082576391973337
		self.C[26] = .8043478260869565217391304347826086956521739130434782608695652173913043478260869565217
		self.C[27] = .8717948717948717948717948717948717948717948717948717948717948717948717948717948717949
		self.C[28] = 1.
	
		# A matrix
		self.A[1,0]   = .4351851851851851851851851851851851851851851851851851851851851851851851851851851851852
		self.A[2,0]   = .2175227402212137286104398734798923400326123258875071216675507357419304139407870179368
		self.A[2,1]   = .2254597159191371485825425826604585371603701302528437555254317203984204632521954382036
		self.A[3,0]   = .1661184210526315789473684210526315789473684210526315789473684210526315789473684210526
		self.A[3,1]   = 0.
		self.A[3,2]   = .4983552631578947368421052631578947368421052631578947368421052631578947368421052631579
		self.A[4,0]   = .8681163193918508865080629713483674348617570159685880673878476405659784672427434203840e-1
		self.A[4,1]   = 0.
		self.A[4,2]   = .3456981948164897000739691452644149026866903085576649936607708344667115529230058847943e-1
		self.A[4,3]   = -.1444105200331793633658889920028385056366520260032657508759536112923114056452583544813e-1
		self.A[5,0]   = .3850951504524952575244726520324404860898823844950484490138343541601000605579674908780e-1
		self.A[5,1]   = 0.
		self.A[5,2]   = 0.
		self.A[5,3]   = .9889604363651382462812798900870548282288802143553903392828423536490053424426345577709e-4
		self.A[5,4]   = .1258652731216402762123982910182735616976625577395859318541619645591514091994326716669
		self.A[6,0]   = .5247404461891304721708365639844356354386624470737997647397329116437616959059854701778
		self.A[6,1]   = 0.
		self.A[6,2]   = 0.
		self.A[6,3]   = .7610651429965941990931946190296373660533922130208900643092560661920652146268815424885e-1
		self.A[6,4]   = -2.135538596825204678401302463372339475781988628449683891486081018353295131152366999963
		self.A[6,5]   = 2.119016745189825526524339470866652730733632823644186992594087231585247087949150559862
		self.A[7,0]   = .3572122856624484555412656871410309454657254402954350529356994466234212462699695879099e-1
		self.A[7,1]   = 0.
		self.A[7,2]   = 0.
		self.A[7,3]   = 0.
		self.A[7,4]   = .4596205641509305161863483311675360886293026632692961086374314175410528945129931429031e-1
		self.A[7,5]   = -.1800016957713219812376781513215261066354472582005125759344515744644655947135707070493e-1
		self.A[7,6]   = .1404669540301245333646491248782654898654978218139650184903068535815036288843799818693e-3
		self.A[8,0]   = .1888176809184108413680419706237505704281864015973418413153299364663524364341955734913e-1
		self.A[8,1]   = 0.
		self.A[8,2]   = 0.
		self.A[8,3]   = 0.
		self.A[8,4]   = 0.
		self.A[8,5]   = .8381198329740939383758467268684128106638290469173528975872944409244863365387981657651e-1
		self.A[8,6]   = .9031585241436450659339626877786556727134244569192456020771509670971435444573825121328e-5
		self.A[8,7]   = .9729721702550808557495179062390587533407132090396133365371679075124515126725605224924e-1
		self.A[9,0]   = -.4080067703469846387001599007585094416645285179656696882006645696781274941696191338570e-1
		self.A[9,1]   = 0.
		self.A[9,2]   = 0.
		self.A[9,3]   = 0.
		self.A[9,4]   = 0.
		self.A[9,5]   = -.6005539308646711394173617417282635784603029781694346062080009050806164919590318269156
		self.A[9,6]   = .1585222658367901823092810289995660365558277114283363946284113044047141269675809741481e-2
		self.A[9,7]   = .3026658851734428583799288358366834830670493200050903757742452562250879940584583879948
		self.A[9,8]   = .6704368334008921764176894190107687125274815661799611686408713261126274393811928758984
		self.A[10,0]  = 6.344326927733666873696263058980778771328558370914152084380122688263116113189631874466
		self.A[10,1]  = 0.
		self.A[10,2]  = 0.
		self.A[10,3]  = 0.
		self.A[10,4]  = 0.
		self.A[10,5]  = 0.
		self.A[10,6]  = 1.975263319684766813850955500728188106557880392559127720135666889111189136253813975729
		self.A[10,7]  = -13.82822337504897849061527131653164721900473695395858210726164571154383357718661916508
		self.A[10,8]  = 14.82423926991066347292103944504014293055447712090877566105421769783705551827529470930
		self.A[10,9]  = -8.370994536873562320168249116193728530299734845668945403444620801647194262518201483006
		self.A[11,0]  = -.9910783781470375735018424706990342484478259963427121190616343432259493286903501070750e-1
		self.A[11,1]  = 0.
		self.A[11,2]  = 0.
		self.A[11,3]  = 0.
		self.A[11,4]  = 0.
		self.A[11,5]  = -1.046319958132641377892211373672215159710875883372049858922517534871447317958500644882
		self.A[11,6]  = -.4662578801256029967826301701912208605727776822301916928051722653634090636623902615221e-3
		self.A[11,7]  = .4243518408197806737520879584209906999975132950474577957967837636354467847566731274758
		self.A[11,8]  = .8350767448298191909277817680995907515365477233544915021061336714983604939879298121048
		self.A[11,9]  = -.6204379562043795620437956204379562043795620437956204379562043795620437956204379562044e-1
		self.A[11,10] = .3051106025934401220442410373760488176964149504195270785659801678108314263920671243326e-3
		self.A[12,0]  = .1731635805893703684448829089734942070445562382116745420114094326722705203174775140332e-1
		self.A[12,1]  = 0.
		self.A[12,2]  = 0.
		self.A[12,3]  = 0.
		self.A[12,4]  = 0.
		self.A[12,5]  = 0.
		self.A[12,6]  = 0.
		self.A[12,7]  = 0.
		self.A[12,8]  = .8690027159880925811795093256330430220184719404902985010762237927304116502630885425916e-3
		self.A[12,9]  = -.9198044746158460357290146202828770811642343120551843217420771155968903404182588708338e-4
		self.A[12,10] = .1833594777349928706130754092439075395240645268074916590065144720493485758190652981058e-6
		self.A[12,11] = .6679448817377525524901838117990401028051762116902291244289142812068791591710992924480e-1
		self.A[13,0]  = .1497702285787817250603134142375551527881374815295532946701659971640303704469333256722e-1
		self.A[13,1]  = 0.
		self.A[13,2]  = 0.
		self.A[13,3]  = 0.
		self.A[13,4]  = 0.
		self.A[13,5]  = 0.
		self.A[13,6]  = 0.
		self.A[13,7]  = 0.
		self.A[13,8]  = .1299053198125687130806056561360945267563487713700848629960379786258472215118648828622
		self.A[13,9]  = .4444252090193421415880477613428889941728587016143899156135432416265141679149665438354e-2
		self.A[13,10] = -.2185380330434085597487818850115824768660155325792369439283645364044817815712239904765e-5
		self.A[13,11] = .6235770138025566019351747531432668514647398310437277172705875380249580436468314885283e-1
		self.A[13,12] = .5389349250407735998767659637686133399860483467584655047185578940287507515886082812093e-1
		self.A[14,0]  = .1956886038861791180930007029621719896471785620210633245797118818081004119873004962774
		self.A[14,1]  = 0.
		self.A[14,2]  = 0.
		self.A[14,3]  = 0.
		self.A[14,4]  = 0.
		self.A[14,5]  = 0.
		self.A[14,6]  = 0.
		self.A[14,7]  = 0.
		self.A[14,8]  = 1.132878041352190960246053936692850204846641933665191609652843227265803479941175715645
		self.A[14,9]  = .9754686357770750130309903534180324613492470735016258589778436907188769670738143543989
		self.A[14,10] = .4607633555393391875625052489818876130584589587566695260680043303699764560744838206512e-3
		self.A[14,11] = -.4833419929869440350491944569648386990066349467457051943375195864364996227787596872297
		self.A[14,12] = .2837255261146423361717978265203569959020643630988946562818139270625464117810614651744
		self.A[14,13] = -1.604879577498682731680210867877554840351555444499826924680761144749197624460666828087
		self.A[15,0]  = -.6628581660952109109950216641584960475848453023977418371554567406302394052049642993363
		self.A[15,1]  = 0.
		self.A[15,2]  = 0.
		self.A[15,3]  = 0.
		self.A[15,4]  = 0.
		self.A[15,5]  = 0.
		self.A[15,6]  = 0.
		self.A[15,7]  = 0.
		self.A[15,8]  = -5.301219753823165631138506283750981613687493380976143446419698251739201243786272433269
		self.A[15,9]  = -5.493744530005151771950832780438489668352801158207831962667055860430416893664761589330
		self.A[15,10] = .6448107716343851659685115053410322161204197332247306929859781244843876115320017653542e-2
		self.A[15,11] = 2.226911096857986220827171118161838780361655511952909244621862410327793805385281866476
		self.A[15,12] = -.8260094546883369994121948528168152647220748853112540011094140493170779782006783468229
		self.A[15,13] = 9.736973734199539440165209948098457305642565726332220268712929325968631361580234056276
		self.A[15,14] = 1.047923362573352907746375340805459350884588027111516805638308114257144242834404582751
		self.A[16,0]  = 9.451896878619703179615895147719645120861585484250220485534769741441232402797149004402
		self.A[16,1]  = 0.
		self.A[16,2]  = 0.
		self.A[16,3]  = 0.
		self.A[16,4]  = 0.
		self.A[16,5]  = 0.
		self.A[16,6]  = 0.
		self.A[16,7]  = 0.
		self.A[16,8]  = 74.07837676951819392708315469048815926453257791083353986411194358002277837032529121349
		self.A[16,9]  = 80.08971633421252928663933464653279322909492698803384552922422585968931770071180510457
		self.A[16,10] = -.1241702484260160323471778040612055855620194337087768444994146945534716737633295008233
		self.A[16,11] = -32.04108125365225923900659816438329678215226279318337501510105142207810131441066638158
		self.A[16,12] = 15.51919421000708125838831543558938623465692911684926717234376165413330242668122537342
		self.A[16,13] = -136.4444237346563024309541869705527862321862115141467999073204884062451779712319267938
		self.A[16,14] = -11.36109896858298168444633695622136642878717316645309025294277498926997344146696251530
		self.A[16,15] = 1.746701961099335199963616081872403749335232589961167014444435282876535760443759733215
		self.A[17,0]  = 1.059086740089530572084686602455620391446019980075022020682291957643879144548290466243
		self.A[17,1]  = 0.
		self.A[17,2]  = 0.
		self.A[17,3]  = 0.
		self.A[17,4]  = 0.
		self.A[17,5]  = 0.
		self.A[17,6]  = 1.975263319684766813850955500728188106557880392559127720135666889111189136253813975729
		self.A[17,7]  = -13.82822337504897849061527131653164721900473695395858210726164571154383357718661916508
		self.A[17,8]  = -26.72676722061492102175215840759265382960786498230207636418025793583833107088763635551
		self.A[17,9]  = -53.49798986553787152824955918158085193982743760760492627590775850823540448031315562431
		self.A[17,10] = .7179812487812968675250181644918743516316099586081748670146732996043783143974128607587e-1
		self.A[17,11] = 18.05559723664965139501142471462605880255358278669785143935434349294122071206578011157
		self.A[17,12] = -8.765163819232982113792324392383185125858432915348234208512916467037776994687248464210
		self.A[17,13] = 76.87522358555575651988028075659565616302805855797883225625477685501643635288672453023
		self.A[17,14] = 6.541007260506904941709851688460080123863085658071943624909334991972860537373864701434
		self.A[17,15] = -.8461837296739539683436767973109554382501241857205608248520628152174372445350689400749
		self.A[17,16] = .3096334815052354314802658810823658907325235844531318754050068324709258105543338930304e-1
		self.A[18,0]  = -.9776673260040047976710026469624134014469714141641816253185153958371390334954366206102e-1
		self.A[18,1]  = 0.
		self.A[18,2]  = 0.
		self.A[18,3]  = 0.
		self.A[18,4]  = 0.
		self.A[18,5]  = -.6005539308646711394173617417282635784603029781694346062080009050806164919590318269156
		self.A[18,6]  = .1585222658367901823092810289995660365558277114283363946284113044047141269675809741481e-2
		self.A[18,7]  = .3026658851734428583799288358366834830670493200050903757742452562250879940584583879948
		self.A[18,8]  = .4285248952296136656501504776128834912424841890282487604085652387338305228046948965999
		self.A[18,9]  = -.2494936049956423479331713062821820760626388763661853215936426940474748269791918051828e-1
		self.A[18,10] = .2011342149874705950638155217740310555133571768677965217932407885434178872356469888792e-1
		self.A[18,11] = .1265809640150575117740069284741749647793461966567090378701137188708013785394651655531
		self.A[18,12] = -.7625505511021937468044360160133448561674800889340550320120549846220783051800508863794e-2
		self.A[18,13] = .2257683215130023640661938534278959810874704491725768321513002364066193853427895981087
		self.A[18,14] = -.2745492489167708483879226183906024865124767290578864277433486220883760763766017543321e-1
		self.A[18,15] = .7783761260627937338705544952370637303437265442198231246348540202794422441986844495010e-2
		self.A[18,16] = .1680830476266175444138000910160343423421707378563384687234091879582812751321589095322e-4
		self.A[18,17] = -.2135549195295375067495229039525877007339581782873806009604806379884885927885993014667e-1
		self.A[19,0]  = .4918722777774214635454694806095471334564471116346374258920200467475606642975663633169e-1
		self.A[19,1]  = 0.
		self.A[19,2]  = 0.
		self.A[19,3]  = 0.
		self.A[19,4]  = 0.
		self.A[19,5]  = .8381198329740939383758467268684128106638290469173528975872944409244863365387981657651e-1
		self.A[19,6]  = .9031585241436450659339626877786556727134244569192456020771509670971435444573825121328e-5
		self.A[19,7]  = .9729721702550808557495179062390587533407132090396133365371679075124515126725605224924e-1
		self.A[19,8]  = -.7780700010332922911083090560520956762627768671838912755501179322945892692289272010345e-1
		self.A[19,9]  = .2322816011687352870985186646759446719321113455516507860765563208248738749278233611786
		self.A[19,10] = -.1069713522971685698540328329852038445947544770807964964519088069861375575524890552230e-1
		self.A[19,11] = -.1273357855008010164642613326804825893357151796749893640146110897003842493364412495521
		self.A[19,12] = .1195596306045770335798408097988199315656849089154404333183111529360783290389646594900
		self.A[19,13] = .1268882175226586102719033232628398791540785498489425981873111782477341389728096676737
		self.A[19,14] = .2044989775051124744376278118609406952965235173824130879345603271983640081799591002045e-1
		self.A[19,15] = -.3909856506984242164691157929655038995858933888334921275260056794557284704385445582377e-2
		self.A[19,16] = -.9117410119332842109125953032188840693633119317454401715027062207348542639139803659651e-5
		self.A[19,17] = .1126093151077456154585623418197514536662349793737574398626301058923382946852045056475e-1
		self.A[19,18] = -.3209868434922071245903287586373535845929558438862699119277778588606558307508436673461
		self.A[20,0]  = .5247404461891304721708365639844356354386624470737997647397329116437616959059854701778
		self.A[20,1]  = 0.
		self.A[20,2]  = 0.
		self.A[20,3]  = .7610651429965941990931946190296373660533922130208900643092560661920652146268815424885e-1
		self.A[20,4]  = -2.135538596825204678401302463372339475781988628449683891486081018353295131152366999963
		self.A[20,5]  = 2.119016745189825526524339470866652730733632823644186992594087231585247087949150559862
		self.A[20,6]  = 0.
		self.A[20,7]  = 0.
		self.A[20,8]  = -.6068669292751351984511141135476525247489731841233590933089036540462466843434512014723
		self.A[20,9]  = -.6975023816048118989159659127206041759560640169551283418432821453966450135301718130674
		self.A[20,10] = -.2539552135383387231958801508125376392283214913791972115937570999107334296436460657368e-1
		self.A[20,11] = 0.
		self.A[20,12] = 0.
		self.A[20,13] = 0.
		self.A[20,14] = 0.
		self.A[20,15] = 0.
		self.A[20,16] = 0.
		self.A[20,17] = .2539552135383387231958801508125376392283214913791972115937570999107334296436460657368e-1
		self.A[20,18] = .6975023816048118989159659127206041759560640169551283418432821453966450135301718130674
		self.A[20,19] = .6068669292751351984511141135476525247489731841233590933089036540462466843434512014723
		self.A[21,0]  = .3850951504524952575244726520324404860898823844950484490138343541601000605579674908780e-1
		self.A[21,1]  = 0.
		self.A[21,2]  = 0.
		self.A[21,3]  = .9889604363651382462812798900870548282288802143553903392828423536490053424426345577709e-4
		self.A[21,4]  = .1258652731216402762123982910182735616976625577395859318541619645591514091994326716669
		self.A[21,5]  = 0.
		self.A[21,6]  = .1148546657824708136802241649776446744025406100582655444423341878411637297467351936831
		self.A[21,7]  = 0.
		self.A[21,8]  = -.1299246462925958527077352114079744607750386324042347411360003531859939834460301561195
		self.A[21,9]  = -.3664591598580916638621456622089859363144583004356560608229075984337239720284817960193
		self.A[21,10] = 0.
		self.A[21,11] = 0.
		self.A[21,12] = 0.
		self.A[21,13] = 0.
		self.A[21,14] = 0.
		self.A[21,15] = 0.
		self.A[21,16] = 0.
		self.A[21,17] = 0.
		self.A[21,18] = .3664591598580916638621456622089859363144583004356560608229075984337239720284817960193
		self.A[21,19] = .1299246462925958527077352114079744607750386324042347411360003531859939834460301561195
		self.A[21,20] = -.1148546657824708136802241649776446744025406100582655444423341878411637297467351936831
		self.A[22,0]  = .2175227402212137286104398734798923400326123258875071216675507357419304139407870179368
		self.A[22,1]  = .2254597159191371485825425826604585371603701302528437555254317203984204632521954382036
		self.A[22,2]  = 0.
		self.A[22,3]  = 0.
		self.A[22,4]  = 0.
		self.A[22,5]  = -.7003676470588235294117647058823529411764705882352941176470588235294117647058823529412
		self.A[22,6]  = -.3841432262252079340469853296205751191707035527677198880717241749265840367609120672261
		self.A[22,7]  = 0.
		self.A[22,8]  = 0.
		self.A[22,9]  = 0.
		self.A[22,10] = 0.
		self.A[22,11] = 0.
		self.A[22,12] = 0.
		self.A[22,13] = 0.
		self.A[22,14] = 0.
		self.A[22,15] = 0.
		self.A[22,16] = 0.
		self.A[22,17] = 0.
		self.A[22,18] = 0.
		self.A[22,19] = 0.
		self.A[22,20] = .3841432262252079340469853296205751191707035527677198880717241749265840367609120672261
		self.A[22,21] = .7003676470588235294117647058823529411764705882352941176470588235294117647058823529412
		self.A[23,0]  = .4351851851851851851851851851851851851851851851851851851851851851851851851851851851852
		self.A[23,1]  = 0.
		self.A[23,2]  = -.4244806610219170956648818689597945762515945523075081224120777274585003570971510665439
		self.A[23,3]  = 0.
		self.A[23,4]  = 0.
		self.A[23,5]  = 0.
		self.A[23,6]  = 0.
		self.A[23,7]  = 0.
		self.A[23,8]  = 0.
		self.A[23,9]  = 0.
		self.A[23,10] = 0.
		self.A[23,11] = 0.
		self.A[23,12] = 0.
		self.A[23,13] = 0.
		self.A[23,14] = 0.
		self.A[23,15] = 0.
		self.A[23,16] = 0.
		self.A[23,17] = 0.
		self.A[23,18] = 0.
		self.A[23,19] = 0.
		self.A[23,20] = 0.
		self.A[23,21] = 0.
		self.A[23,22] = .4244806610219170956648818689597945762515945523075081224120777274585003570971510665439
		self.A[24,0]  = 14.54990971513478580914495526598245283940024810265087163382498896386800769812364397026
		self.A[24,1]  = -2.609444444444444444444444444444444444444444444444444444444444444444444444444444444444
		self.A[24,2]  = -2.016004609236637754870351028563643794559738431497207211298306162299623087053267335725
		self.A[24,3]  = 0.
		self.A[24,4]  = 0.
		self.A[24,5]  = -1.666875
		self.A[24,6]  = -1.840010137609049715480551028603992780751854184812582730319893278211480604766702319850
		self.A[24,7]  = 0.
		self.A[24,8]  = 112.8850026879393650927243496060343039004107661597559344830054269802155904451112935844
		self.A[24,9]  = 123.3942086822776167534198661922997568027612066045886506351026838994566451157760161079
		self.A[24,10] = -.7912126656078716671206667350992937018785263673658058027818457299775466324638082312137
		self.A[24,11] = -50.05149873558555185701853277950418028083920384013973228797374663569746882026502756132
		self.A[24,12] = 24.88778291494286023489265916235844707524170235927746532159915897878512972222657954044
		self.A[24,13] = -212.1164197057320847511194729602871445996396409264694299135694995667783941778631290781
		self.A[24,14] = -17.89082255740024165397920079197116480379826062999613139163493416384514698556677051724
		self.A[24,15] = 2.509716434086569985363077956674238286724267256774032757480361264001771307725305753519
		self.A[24,16] = .1162475886937088360535666476065014131326443901611275096836079807311057386735832665899
		self.A[24,17] = .5840328281597421751021178306186155103094570763962374837158281176580253234011237044692
		self.A[24,18] = 1.584417806712708397483834073813399023446091584920600008797782958694409008929749703075
		self.A[24,19] = 1.338635006378392645053446531474068534729248229446179562750186952887872256191439757181
		self.A[24,20] = 1.840010137609049715480551028603992780751854184812582730319893278211480604766702319850
		self.A[24,21] = 1.666875
		self.A[24,22] = 2.016004609236637754870351028563643794559738431497207211298306162299623087053267335725
		self.A[24,23] = 2.609444444444444444444444444444444444444444444444444444444444444444444444444444444444
		self.A[25,0]  = .4213659219087082450668941175898637044576077571412870755532166081975824639155816025423
		self.A[25,1]  = 0.
		self.A[25,2]  = 0.
		self.A[25,3]  = 0.
		self.A[25,4]  = 0.
		self.A[25,5]  = 2.360375290413766425107807321597798068298832670876354152068674897991709951684675057597
		self.A[25,6]  = .7887926811836902144270824477231365437442962856383491851349713361086542561144935964760e-1
		self.A[25,7]  = -1.881850641776530466652474803895333308262409829541779704822719782535259533895036456964
		self.A[25,8]  = -1.304700734906095391371228323883517431348033016096248869612640797091741378769699510853
		self.A[25,9]  = .1146971532060496506611311517441641422873299688900135018999537656144573134542692746612
		self.A[25,10] = -.5223613182942077907170609676910338480915071906000186351215047373740452623507436746018e-2
		self.A[25,11] = .7134840563194221964556259902880063405282394887795535106616674222878805188799073074497
		self.A[25,12] = 0.
		self.A[25,13] = 0.
		self.A[25,14] = 0.
		self.A[25,15] = 0.
		self.A[25,16] = 0.
		self.A[25,17] = 0.
		self.A[25,18] = 0.
		self.A[25,19] = 0.
		self.A[25,20] = 0.
		self.A[25,21] = 0.
		self.A[25,22] = 0.
		self.A[25,23] = 0.
		self.A[25,24] = 0.
		self.A[26,0]  = -1.016867684065179179311540011641152067739527559831057381007428484343486238202737995538
		self.A[26,1]  = 0.
		self.A[26,2]  = 0.
		self.A[26,3]  = 0.
		self.A[26,4]  = 0.
		self.A[26,5]  = -7.712044352285817603610736737545203003182107799250377304475316992902646516184627861475
		self.A[26,6]  = -.4034008409374858753410643280039779311650023076266296210030337872976327092081717454860
		self.A[26,7]  = 6.739165476490825275476530741799137001781411805688541615342712456896969009075069743117
		self.A[26,8]  = 6.014994643407224294180918860565568523540180411761624603152605084567461778121724411633
		self.A[26,9]  = -1.138427387973993086846707441740657423236451007119331997503516596250482644801941463866
		self.A[26,10] = .5009271973181599563449431362188397685579770251743883251930235916805543539186803311245e-1
		self.A[26,11] = -3.113250932564715585587456369457050044282973515833171733791753110672694235302567710668
		self.A[26,12] = 0.
		self.A[26,13] = 0.
		self.A[26,14] = 0.
		self.A[26,15] = 0.
		self.A[26,16] = 0.
		self.A[26,17] = 0.
		self.A[26,18] = 0.
		self.A[26,19] = 0.
		self.A[26,20] = 0.
		self.A[26,21] = 0.
		self.A[26,22] = 0.
		self.A[26,23] = 0.
		self.A[26,24] = 0.
		self.A[26,25] = 1.384086184284282287144691407184059663080846182736441247635994288225760468937471545692
		self.A[27,0]  = 1.131093475949031458408970675798323789793651098141584053672723684523337489654810813567
		self.A[27,1]  = 0.
		self.A[27,2]  = 0.
		self.A[27,3]  = 0.
		self.A[27,4]  = 0.
		self.A[27,5]  = -11.30475611955440577592346561419842170756276153921836803995513790246783417109406727002
		self.A[27,6]  = .8673508908529372037894544277195364499375277491231106383198729858086287127092489733738e-1
		self.A[27,7]  = 4.971317844154333915807514558966554931901059536397301577948574837963960265600948244094
		self.A[27,8]  = 14.86493772010299652718002500847699984298963300991290479767277341466986550270499895786
		self.A[27,9]  = -5.526130551905351405702373768620234518347747212552226033118137855980983127699224086148
		self.A[27,10] = .1017790491986200061558195486579246543940163889995857946318495764475820398967567810759
		self.A[27,11] = -5.412708567655345677389304794550103449135846140313886894023804835964789906470796078495
		self.A[27,12] = 0.
		self.A[27,13] = 0.
		self.A[27,14] = 0.
		self.A[27,15] = 0.
		self.A[27,16] = 0.
		self.A[27,17] = 0.
		self.A[27,18] = 0.
		self.A[27,19] = 0.
		self.A[27,20] = 0.
		self.A[27,21] = 0.
		self.A[27,22] = 0.
		self.A[27,23] = 0.
		self.A[27,24] = 0.
		self.A[27,25] = 2.119905903216124397337756706998226742489167533804374626293399155463227287948398852330
		self.A[27,26] = -.1603789707964253713820928925063521366431305782887091520824325014403564569409562398070
		self.A[28,0]  = 46.12864603958015905056850990838704062569763496465412763519385202873225169326091596022
		self.A[28,1]  = 0.
		self.A[28,2]  = 0.
		self.A[28,3]  = 0.
		self.A[28,4]  = 0.
		self.A[28,5]  = 27.91300163119399908845158457840426795358131126287389180101909720743727165524743997096
		self.A[28,6]  = 16.11362689862451240990975288339484000039234961533373225370151713993919933068137837655
		self.A[28,7]  = -125.4696763444318726329250646477825268481685879278990572088898486327607587168793188547
		self.A[28,8]  = 76.57182020120529497684089567511627659347626021577320126137924234653281598557298427556
		self.A[28,9]  = -48.97805558723490361747755876313897556229903489002597907918723603724236906416246304792
		self.A[28,10] = -1.242830487244052672528847080627776989497284066852925470069357906835696725179561610962
		self.A[28,11] = 18.85807213383620068645464308722546866214730809725606975205204214283143331463602851729
		self.A[28,12] = 0.
		self.A[28,13] = 0.
		self.A[28,14] = 0.
		self.A[28,15] = 0.
		self.A[28,16] = 0.
		self.A[28,17] = 0.
		self.A[28,18] = 0.
		self.A[28,19] = 0.
		self.A[28,20] = 0.
		self.A[28,21] = 0.
		self.A[28,22] = 0.
		self.A[28,23] = 0.
		self.A[28,24] = 0.
		self.A[28,25] = -8.871982194511738170929283011936752038683182063824370821380217475185779921363890310462
		self.A[28,26] = -2.069534982695615656321598541301254038890059866116256078922522295738656144906973495868
		self.A[28,27] = 2.046912691678016537956965912259391642243284658827565955103431482290288593093460219351
	
		# 12th order solution
		self.B[0]  = .2380952380952380952380952380952380952380952380952380952380952380952380952380952380952e-1
		self.B[1]  = -.11
		self.B[2]  = -.17
		self.B[3]  = 0.
		self.B[4]  = 0.
		self.B[5]  = -.19
		self.B[6]  = -.21
		self.B[7]  = 0.
		self.B[8]  = -.23
		self.B[9]  = -.27
		self.B[10] = -.29
		self.B[11] = 0.
		self.B[12] = .1384130236807829740053502031450331467488136400899412345912671194817223119377730668077
		self.B[13] = .2158726906049313117089355111406811389654720741957730511230185948039919737765126474781
		self.B[14] = .2438095238095238095238095238095238095238095238095238095238095238095238095238095238095
		self.B[15] = .2158726906049313117089355111406811389654720741957730511230185948039919737765126474781
		self.B[16] = .1384130236807829740053502031450331467488136400899412345912671194817223119377730668077
		self.B[17] = .29
		self.B[18] = .27
		self.B[19] = .23
		self.B[20] = .21
		self.B[21] = .19
		self.B[22] = .17
		self.B[23] = .11
		self.B[24] = .2380952380952380952380952380952380952380952380952380952380952380952380952380952380952e-1
		self.B[25] = 0.
		self.B[26] = 0.
		self.B[27] = 0.
		self.B[28] = 0.
	
		# 9th order solution
		self.Bhat[0]  = .1357267366422036691624508570375039921213961405197383540990729271585608575744367221422e-1
		self.Bhat[1]  = 0.
		self.Bhat[2]  = 0.
		self.Bhat[3]  = 0.
		self.Bhat[4]  = 0.
		self.Bhat[5]  = 0.
		self.Bhat[6]  = 0.
		self.Bhat[7]  = 0.
		self.Bhat[8]  = .1957242608025905233613155193355413222756712243739956539857744660017249157522413539338
		self.Bhat[9]  = .6188866347435608661616060238380345321740579185698262925072441030810461130118224626650e-1
		self.Bhat[10] = .2356461254963383884566009189531765712282459575681722654363888116993755654359289982043
		self.Bhat[11] = .9356981277656948171659125355126822887878370575133833270918675400840988929847712528189e-1
		self.Bhat[12] = 0.
		self.Bhat[13] = 0.
		self.Bhat[14] = 0.
		self.Bhat[15] = 0.
		self.Bhat[16] = 0.
		self.Bhat[17] = 0.
		self.Bhat[18] = 0.
		self.Bhat[19] = 0.
		self.Bhat[20] = 0.
		self.Bhat[21] = 0.
		self.Bhat[22] = 0.
		self.Bhat[23] = 0.
		self.Bhat[24] = 0.
		self.Bhat[25] = .2788382624223597882496809755901865993371492355322230983832374605891024297949723722505
		self.Bhat[26] = .4265887719284871852244002213010235951116369449606049493109912099005098392090245967637
		self.Bhat[27] = -.2878025166474501962999477241233861487151201542084751225511851437798538088192303308430
		self.Bhat[28] = -.1802605391747162424104685269536402054591231988681564193502526144322952773004003407203e-1
		return
	
	# Convert to a function that python can see
	def CheckTableau(self):
		'''
		Check the sanity of the Butcher's Tableau 
		of a specified Runge-Kutta scheme.
		'''
		retval = 1
		checksum = 1e-6
		
		# Check steps
		sum1, sum2 = 0., 0.
		for ii in range(self.nstep):
			sum = 0.
			for jj in range(ii):
				sum += self.A[ii, jj]
				
			if abs(self.C[ii] - sum) > checksum:
				print("Error in row %i of Butcher tableau!\n" % ii)
				retval = 0
				
			sum1 += self.B[ii]
			sum2 += self.Bhat[ii]
			
		if abs(1. - sum1) > checksum:
			print("Error in solution high of Butcher tableau!\n")
			retval = 0
			
		if abs(1. - sum2) > checksum:
			print("Error in solution low of Butcher tableau!\n")
			retval = 0
		
		return retval
	
def CheckTableau(scheme):
	rk = RKMethod(scheme)
	return rk.CheckTableau()