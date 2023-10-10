#!/bin/env python
#
# Example 2: using the odeRK with parameters
#			 and output/event functions.
#
# Arnau Miro, 2018
# Last rev: 2020
from __future__ import print_function

import numpy as np, matplotlib.pyplot as plt
import pyRKIntegrator as rk


# Parameters
g    = 9.81
k    = 1.
m    = 10.
tend = 10.
h0   = 100.
v0   = 0.

# Define the function to integrate
def eqnfreefall(t,y,n,dydx,*args):
	'''
	t    = time
	y[0] = velocity
	y[1] = altitude
	'''
	v = y[0]
	h = y[1]
	dydx[0] = - g - (k/m)*v
	dydx[1] = v

# Define an output function to print the value
# of velocity and position at each successful timestep
def printvalues(t,y,n):
	'''
	Output function. Must be as:

	outputfcn(x,y,n)

	Where:
		> x: integration instant
		> y: variables (of size n)
		> n: number of variables

	Must return:
		> 1: continue
		> 0: stop
	'''
	v = y[0]
	h = y[1]
	print("time %.2f s, h = %.2f, v = %.2f" % (t,h,v))
	return 1

# Define a function to stop the integration once a
# certain criteria is met
def stoponzero(x,y,n,value,direction):
	'''
	Event function. Must be as:

	eventfcn(x,y,n,value,direction)

	Where:
		> x: integration instant
		> y: variables (of size n)
		> n: number of variables

		> value: is a mathematical expression describing the event. 
				 An event occurs when value(i) is equal to zero.
		> direction: 0 if all zeros are to be located.
					+1 locates only zeros where the event function is increasing.
					-1 locates only zeros where the event function is decreasing.

	Must return:
		> 1: continue
		> 0: stop
	'''
	value[0] = y[1]
	return 0

# Set span and initial solution
tspan = np.array([0., tend],np.double)
y0    = np.array([v0, h0], np.double)

# Use the odeset to generate the integrator
# parameters structure
odeset = rk.odeset(eventfun=stoponzero,outputfun=printvalues) # initalize with basic parameters
odeset.set_h(tspan)  # set defaults for h0 and hmin

# Launch the integrator
print('Free-fall:')
t,y,err = rk.ode45(eqnfreefall,tspan,y0,odeset)
print("end %.2f s, h = %.2f, v = %.2f" % (t[-1],y[-1,1],y[-1,0]))
print("Integration error: %.2e" % err)

# Theoretical solution
v_th = lambda t : -(m/k)*g*(1-np.exp((-k/m)*t));

# Create plot figures
plt.figure(num=1,figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')
ax1 = plt.subplot(2,1,1)
ax1.plot(t,y[:,0],'k',label='Numerical Result')
ax1.plot(t,v_th(t),'k--',label='Theoretical Result')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Velocity (m/s)')
ax1.legend()
ax2 = plt.subplot(2,1,2)
ax2.plot(t,y[:,1],'k')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Altitude (m)')

plt.show()