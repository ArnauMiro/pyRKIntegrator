#!/bin/env python
#
# Example 4: using the odeRKN with odeset
#			 and different schemes
#
# Arnau Miro, 2018
# Last rev: 2020
from __future__ import print_function

import numpy as np
import pyRKIntegrator as rk

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Define the function to integrate
def odefun(x,y,n,dy2dx,*args):
	'''
	Function to integrate. Must be as:

	odefun(x,y,n.dy2dx)

	Where:
		> x: integration instant
		> y: variables (of size n)
		> n: number of variables
		> dy2dx: second derivative of y (of size n)
	'''
	dy2dx[0] = -np.sin(x) + np.cos(y[0])
	dy2dx[1] = np.cos(x) - np.sin(y[1])

# Set span and initial solution
xspan = np.array([0., 10.],np.double)
y0    = np.array([0., 0.], np.double)
dy0   = np.array([1., 1.], np.double)

# Use the odeset to generate the integrator
# parameters structure
odeset = rk.odeset() # initalize with basic parameters

# Create plot figures
plt.figure(num=1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')
ax1 = plt.subplot(2,1,1)
ax1.set_title("f''(x) = cos(x) + sin(y)")
ax1.set_ylabel('f(x)')
ax2 = plt.subplot(2,1,2)
ax2.set_title("f''(x) = sin(x) + cos(y)")
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')

# Loop all the schemes
for scheme in rk.RKN_SCHEMES:
	# Set integration scheme
	odeset.set_h(xspan)  # set defaults for h0 and hmin
	# Launch the integrator
	print("scheme %s, " % scheme,end=' ')
	x,y,dy,err = rk.odeRKN(scheme,odefun,xspan,y0,dy0,odeset)
	print("error = %.2e with %d steps" % (err,len(x)))
	# Plot
	ax1.plot(x,y[:,0],label=scheme)
	ax2.plot(x,y[:,1],label=scheme)

# Show the plot
ax1.legend(loc='lower right',fontsize='x-small',ncol=3)
ax2.legend(loc='lower right',fontsize='x-small',ncol=3)
plt.show()