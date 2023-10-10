#!/bin/env python
#
# Example 5: Two-Body problem (RKN)
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

R0 = 1.5e8 # km
V0 = np.sqrt(G*M2/R0)
T  = 2*np.pi / np.sqrt(G*M2/R0/R0/R0)

# Define the function to integrate
def TwoBody(t,var,n,varp2,*args):
	'''
	Body 1: Perturbated
		var[0] = rx   varp[0] = vx    varp2[0] = ax
		var[1] = ry   varp[1] = vy    varp2[1] = ay
		var[2] = rz   varp[2] = vz    varp2[2] = az
	Body 2: Perturber
		var[3] = rx   varp[3] = vx    varp2[3] = ax
		var[4] = ry   varp[4] = vy    varp2[4] = ay
		var[5] = rz   varp[5] = vz    varp2[5] = az
	'''
	r = np.sqrt( (var[3]-var[0])*(var[3]-var[0]) + \
		         (var[4]-var[1])*(var[4]-var[1]) + \
		         (var[5]-var[2])*(var[5]-var[2]) )
	# Perturbated
	varp2[0] = ( G*M2/r/r/r ) * (var[3]-var[0])
	varp2[1] = ( G*M2/r/r/r ) * (var[4]-var[1])
	varp2[2] = ( G*M2/r/r/r ) * (var[5]-var[2])

	# Perturber
	varp2[3] = ( G*M1/r/r/r ) * (var[0]-var[3])
	varp2[4] = ( G*M1/r/r/r ) * (var[1]-var[4])
	varp2[5] = ( G*M1/r/r/r ) * (var[2]-var[5])

# Set span and initial solution
tspan = np.array([0., 10.*T], np.double)
y0    = np.array([R0,0,0,0,0,0], np.double)
dy0   = np.array([0,V0,0,0,0,0], np.double)

# Generate odeset structure
odeset = rk.odeset(h0=24*3600,eps=1e-6)

# Create plot figures
plt.figure(num=1,figsize=(12,6),dpi=100,facecolor='w',edgecolor='k')
ax1 = plt.subplot(1,2,1,projection='3d')
ax1.set_title('Orbit')
ax2 = plt.subplot(1,2,2)
ax2.set_title('Energy')
ax2.set_xlabel('time (days)')
ax2.set_ylabel('Ek')

# Mechanic energy that must be conserved
Ek0 = 0.5*V0*V0 + G*M1/R0

# Loop all the schemes
for scheme in rk.RKN_SCHEMES:
	print("scheme %s," % scheme,end=" ")
	try:
		# Launch the integrator
		t,y,dy,err = rk.odeRKN(scheme,TwoBody,tspan,y0,dy0,odeset)
		print("error = %.2e with %d steps" % (err,len(t)))
		# Plot results
		ax1.plot(y[:,0],y[:,1],y[:,2],label=scheme)
		Ek = 0.5*np.linalg.norm(dy[:,0:2],axis=1)**2 + G*M1/np.linalg.norm(y[:,0:2],axis=1)
		ax2.plot(t/3600/34,(Ek-Ek0)/Ek0,label=scheme)
	except Exception as e:
		print("%s" % e)

# Show the plot
ax2.legend(loc='lower right',fontsize='x-small',ncol=3)
plt.show()