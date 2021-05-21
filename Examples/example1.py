#!/bin/env python
#
# Example 1: using the odeRK with odeset
#			 and different schemes
#
# Arnau Miro, 2018
# Last rev: 2021
from __future__ import print_function, division

import numpy as np
import pyRKIntegrator as rk

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Define the function to integrate
def odefun(x,y,n,dydx,*args):
	'''
	Function to integrate. Must be as:

	odefun(x,y,n.dydx)

	Where:
		> x: integration instant
		> y: variables (of size n)
		> n: number of variables
		> dydx: derivative of y (of size n)
	'''
	dydx[0] = np.cos(x) + np.sin(y[0])
	dydx[1] = np.sin(x) + np.cos(y[1])

# Set span and initial solution
xspan = np.array([0., 10.],np.double)
y0    = np.array([0., 0.], np.double)

# Use the odeset to generate the integrator
# parameters structure
odeset = rk.odeset() # initalize with basic parameters

# Create plot figures
plt.figure(num=1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')
ax1 = plt.subplot(2,1,1)
ax1.set_title("f'(x) = cos(x) + sin(y)")
ax1.set_ylabel('f(x)')
ax2 = plt.subplot(2,1,2)
ax2.set_title("f'(x) = sin(x) + cos(y)")
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')

# Loop all the schemes
for scheme in rk.RK_SCHEMES:
	# Set integration scheme
	odeset.set_h(xspan)  # set defaults for h0 and hmin
	# Check the sanity of the RungeKutta tableau
	if rk.CheckTableau(scheme):
		print("scheme %s, tableau ok," % scheme,end=' ')
	else:
		print("scheme %s, problems with tableau!" % scheme)
		continue
	# Launch the integrator
	x,y,err = rk.odeRK(scheme,odefun,xspan,y0,odeset)
	print("error = %.2e with %d steps" % (err,len(x)))
	# Plot
	ax1.plot(x,y[:,0],label=scheme)
	ax2.plot(x,y[:,1],label=scheme)

# Show the plot
ax1.legend(loc='lower right',fontsize='x-small',ncol=3)
ax2.legend(loc='lower right',fontsize='x-small',ncol=3)
plt.show()