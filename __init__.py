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
	Last rev: 2021
'''

__VERSION__ = 3.0

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

## IMPORTS ##
from .RungeKutta import odeset, CheckTableau
#RK_SCHEMES, RKN_SCHEMES, odeset, CheckTableau
from .RungeKutta import odeRK
#, ode23, ode45

del RungeKutta