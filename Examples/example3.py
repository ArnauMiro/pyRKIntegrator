#!/bin/env python
#
# Example 3: Two-Body problem
#
# Arnau Miro, 2018
# Last rev: 2020
from __future__ import print_function

import numpy as np
import pyRKIntegrator as rk

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# Parameters
G  = 6.674e-11*1.e-9 # km3/kg s2
M1 = 5.97219e24      # kg
M2 = 1.9891e30       # kg

R0 = 1.5e8           # km
V0 = np.sqrt(G*M2/R0)
T  = 2*np.pi / np.sqrt(G*M2/R0/R0/R0)

# Define the function to integrate
def TwoBody(t,var,n,varp,*args):
	'''
	Body 1: Perturbated
		var[0] = rx   var[3] = vx
		var[1] = ry   var[4] = vy
		var[2] = rz   var[5] = vz
	Body 2: Perturber
		var[6] = rx   var[9]  = vx
		var[7] = ry   var[10] = vy
		var[8] = rz   var[11] = vz
	'''

	r = np.sqrt( (var[6]-var[0])*(var[6]-var[0]) + \
		         (var[7]-var[1])*(var[7]-var[1]) + \
		         (var[8]-var[2])*(var[8]-var[2]) )

	# Perturbated
	varp[0] = var[3]
	varp[1] = var[4]
	varp[2] = var[5]
	varp[3] = ( G*M2/r/r/r ) * (var[6]-var[0])
	varp[4] = ( G*M2/r/r/r ) * (var[7]-var[1])
	varp[5] = ( G*M2/r/r/r ) * (var[8]-var[2])

	# Perturber
	varp[6]  = var[9]
	varp[7]  = var[10]
	varp[8]  = var[11]
	varp[9]  = ( G*M1/r/r/r ) * (var[0]-var[6])
	varp[10] = ( G*M1/r/r/r ) * (var[1]-var[7])
	varp[11] = ( G*M1/r/r/r ) * (var[2]-var[8])

# Set span and initial solution
tspan = np.array([0., 2.*T], np.double)
y0 = np.array([R0,0,0,0,V0,0,0,0,0,0,0,0], np.double)

# Generate odeset structure
odeset = rk.odeset(h0=24.*3600.,eps=1e-6)

# Create plot figures
plt.figure(num=1,figsize=(12,6),dpi=100,facecolor='w',edgecolor='k')
ax1 = plt.subplot(1,2,1,projection='3d')
ax1.set_title('Orbit')
ax2 = plt.subplot(1,2,2)
ax2.set_title('Energy')
ax2.set_xlabel('time (days)')
ax2.set_ylabel('Ek')

# Mechanic energy must be conserved
Ek0 = 0.5*V0*V0 + G*M1/R0

# Loop all the schemes
for scheme in rk.RK_SCHEMES[3:]: # Do not run the lower order schemes
	print("scheme %s," % scheme,end=' ')
	try:
		# Launch the integrator
		t,y,err = rk.odeRK(scheme,TwoBody,tspan,y0,odeset)
		print("error = %.2e with %d steps" % (err,len(t)))
		# Plot results
		ax1.plot(y[:,0],y[:,1],y[:,2],label=scheme)
		Ek = 0.5*np.linalg.norm(y[:,3:5],axis=1)**2 + G*M1/np.linalg.norm(y[:,0:2],axis=1)
		ax2.plot(t/3600/34,(Ek-Ek0)/Ek0,label=scheme)
	except Exception as e:
		print("%s" % e)

# Show the plot
ax2.legend(loc='lower right',fontsize='x-small',ncol=3)
plt.show()